import torch
import numpy as np
import os
import pathlib
import cv2
import tempfile
import logging
import signal
import time
from urllib.parse import urljoin
from contextlib import contextmanager

from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.label_interface.objects import PredictionValue
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from tqdm import tqdm

logger = logging.getLogger(__name__)

@contextmanager
def timeout_context(seconds, operation_name):
    """Context manager for operation timeouts using SIGALRM"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation '{operation_name}' timed out after {seconds}s")

    # Set up the signal handler (Unix only)
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows fallback - no timeout
        logger.warning(f'Timeout not supported on this platform for operation: {operation_name}')
        yield

def check_gpu_health():
    """Check GPU availability and memory before inference"""
    if DEVICE == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but DEVICE=cuda")

        gpu_id = 0
        props = torch.cuda.get_device_properties(gpu_id)
        total_memory = props.total_memory / 1024**3  # GB
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        cached = torch.cuda.memory_reserved(gpu_id) / 1024**3

        logger.info(f'🎮 GPU: {props.name}')
        logger.info(f'💾 GPU Memory: {allocated:.2f}GB allocated, '
                   f'{cached:.2f}GB reserved, {total_memory:.2f}GB total')

        if allocated > total_memory * 0.9:
            logger.warning(f'⚠️  GPU memory usage high: {allocated/total_memory*100:.1f}%')

        return True
    else:
        logger.info(f'💻 Running on CPU (DEVICE={DEVICE})')
        return False

DEVICE = os.getenv('DEVICE', 'cuda')
MODEL_CONFIG = os.getenv('MODEL_CONFIG', 'configs/sam2.1/sam2.1_hiera_l.yaml')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'sam2.1_hiera_large.pt')
MAX_FRAMES_TO_TRACK = int(os.getenv('MAX_FRAMES_TO_TRACK', 1000))
TRACK_FPS = float(os.getenv('TRACK_FPS', '0'))  # 0 means use original FPS (no temporal downsampling)

if DEVICE == 'cuda':
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# build path to the model checkpoint
sam2_checkpoint = str(pathlib.Path(__file__).parent / "/sam2" / "checkpoints" / MODEL_CHECKPOINT)
predictor = build_sam2_video_predictor(MODEL_CONFIG, sam2_checkpoint)


# manage cache for inference state
# TODO: make it process-safe and implement cache invalidation
_predictor_state_key = ''
_inference_state = None

def get_inference_state(video_dir):
    global _predictor_state_key, _inference_state
    if _predictor_state_key != video_dir:
        _predictor_state_key = video_dir
        _inference_state = predictor.init_state(video_path=video_dir)
    return _inference_state


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "sam2")

    def split_frames(self, video_path, temp_dir, start_frame=0, end_frame=100, stride: int = 1):
        # Open the video file
        logger.info(f'📹 Opening video file: {video_path}')

        try:
            video = cv2.VideoCapture(video_path)
        except Exception as e:
            logger.error(f'❌ Failed to open video with cv2: {e}')
            raise ValueError(f"Could not open video file: {video_path}") from e

        # check if loaded correctly
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Check available disk space before starting
        import shutil
        stat = shutil.disk_usage(temp_dir)
        free_gb = stat.free / 1024**3
        logger.info(f'💾 Available disk space: {free_gb:.2f}GB')
        if free_gb < 5:  # Less than 5GB free
            logger.warning(f'⚠️  Low disk space: {free_gb:.2f}GB free - extraction may fail')

        total_frames_in_video = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_extract = end_frame - start_frame
        logger.info(f'📊 Video has {total_frames_in_video} total frames')
        logger.info(f'🎬 Extracting frames {start_frame} to {end_frame} ({frames_to_extract} frames)')

        frame_count = 0
        extracted_count = 0
        last_heartbeat = time.time()
        last_disk_check = time.time()
        HEARTBEAT_INTERVAL = 30  # Log every 30 seconds
        DISK_CHECK_INTERVAL = 60  # Check disk space every minute

        # Optimize JPEG quality for faster writes (more aggressive for RAM disk)
        JPEG_QUALITY = 75  # Further reduced quality for maximum speed on RAM disk

        while True:
            # Heartbeat logging for long operations
            now = time.time()
            if now - last_heartbeat > HEARTBEAT_INTERVAL:
                logger.info(f'💓 Heartbeat: Extracted {extracted_count}/{frames_to_extract} frames, reading frame {frame_count}')
                last_heartbeat = now
            
            # Periodic disk space check
            if now - last_disk_check > DISK_CHECK_INTERVAL:
                stat = shutil.disk_usage(temp_dir)
                free_gb = stat.free / 1024**3
                logger.info(f'💾 Disk space check: {free_gb:.2f}GB free')
                if free_gb < 2:  # Critical threshold
                    logger.error(f'❌ Critical: Low disk space ({free_gb:.2f}GB) - stopping extraction')
                    video.release()
                    raise RuntimeError(f'Out of disk space: {free_gb:.2f}GB free')
                last_disk_check = now

            # Read a frame from the video
            try:
                success, frame = video.read()
            except Exception as e:
                logger.error(f'❌ Exception reading frame {frame_count}: {e}')
                break

            if frame_count < start_frame:
                frame_count += 1
                continue
            if frame_count >= end_frame:
                break

            # Apply temporal downsampling if stride > 1
            if stride > 1 and (frame_count - start_frame) % stride != 0:
                frame_count += 1
                continue

            # If frame is read correctly, success is True
            if not success:
                logger.error(f'❌ Failed to read frame {frame_count}')
                break

            # Generate a filename for the frame using the pattern with frame number: '%05d.jpg'
            frame_filename = os.path.join(temp_dir, f'{frame_count:05d}.jpg')
            if os.path.exists(frame_filename):
                logger.debug(f'Frame {frame_count}: {frame_filename} already exists')
                yield frame_filename, frame
            else:
                # Save the frame as an image file with optimized quality
                try:
                    # Use optimized JPEG parameters for faster writes
                    success_write = cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if not success_write:
                        logger.error(f'❌ Failed to write frame {frame_count} to {frame_filename}')
                        raise IOError(f'cv2.imwrite failed for frame {frame_count}')
                    logger.debug(f'Frame {frame_count}: {frame_filename}')
                except Exception as e:
                    logger.error(f'❌ Error writing frame {frame_count}: {e}')
                    # Check disk space on error
                    stat = shutil.disk_usage(temp_dir)
                    free_gb = stat.free / 1024**3
                    logger.error(f'💾 Disk space: {free_gb:.2f}GB free')
                    raise
                yield frame_filename, frame
                
                # Explicitly clean up frame memory to prevent memory leaks
                del frame

            extracted_count += 1
            # Log progress every 10 frames
            if extracted_count % 10 == 0:
                logger.info(f'⏳ Extracted {extracted_count}/{frames_to_extract} frames...')

            frame_count += 1

        # Release the video object
        video.release()
        logger.info(f'✅ Frame extraction complete: {extracted_count} frames extracted')

    def get_prompts(self, context) -> List[Dict]:
        logger.debug(f'Extracting keypoints from context: {context}')
        prompts = []
        for ctx in context['result']:
            # Process each video tracking object separately
            obj_id = ctx['id']
            for obj in ctx['value']['sequence']:
                x = obj['x'] / 100
                y = obj['y'] / 100
                box_width = obj['width'] / 100
                box_height = obj['height'] / 100
                frame_idx = obj['frame'] - 1

                # SAM2 video works with keypoints - convert the rectangle to the set of keypoints within the rectangle

                # bbox (x, y) is top-left corner
                kps = [
                    # center of the bbox
                    [x + box_width / 2, y + box_height / 2],
                    # half of the bbox width to the left
                    [x + box_width / 4, y + box_height / 2],
                    # half of the bbox width to the right
                    [x + 3 * box_width / 4, y + box_height / 2],
                    # half of the bbox height to the top
                    [x + box_width / 2, y + box_height / 4],
                    # half of the bbox height to the bottom
                    [x + box_width / 2, y + 3 * box_height / 4]
                ]

                points = np.array(kps, dtype=np.float32)
                labels = np.array([1] * len(kps), dtype=np.int32)
                prompts.append({
                    'points': points,
                    'labels': labels,
                    'frame_idx': frame_idx,
                    'obj_id': obj_id
                })

        return prompts

    def _get_fps(self, context):
        # get the fps from the context
        frames_count = context['result'][0]['value']['framesCount']
        duration = context['result'][0]['value']['duration']
        return frames_count, duration

    # def convert_mask_to_bbox(self, mask):
    #     # convert mask to bbox
    #     h, w = mask.shape[-2:]
    #     mask_int = mask.reshape(h, w, 1).astype(np.uint8)
    #     contours, _ = cv2.findContours(mask_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) == 0:
    #         return None
    #     x, y, w, h = cv2.boundingRect(contours[0])
    #     return {
    #         'x': x,
    #         'y': y,
    #         'width': w,
    #         'height': h
    #     }

    def convert_mask_to_bbox(self, mask):
        # squeeze
        mask = mask.squeeze()

        y_indices, x_indices = np.where(mask == 1)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        # Find the min and max indices
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)

        # Get mask dimensions
        height, width = mask.shape

        # Calculate bounding box dimensions
        box_width = xmax - xmin + 1
        box_height = ymax - ymin + 1

        # Normalize and scale to percentage
        x_pct = (xmin / width) * 100
        y_pct = (ymin / height) * 100
        width_pct = (box_width / width) * 100
        height_pct = (box_height / height) * 100

        return {
            "x": round(x_pct, 2),
            "y": round(y_pct, 2),
            "width": round(width_pct, 2),
            "height": round(height_pct, 2)
        }


    def _bbox_iou(self, box1: Dict, box2: Dict) -> float:
        """Compute IoU between two bboxes in percentage coordinates.

        Boxes use Label Studio-style percentages: x, y (top-left) and width/height in [0, 100].
        """
        x1_min = box1["x"]
        y1_min = box1["y"]
        x1_max = x1_min + box1["width"]
        y1_max = y1_min + box1["height"]

        x2_min = box2["x"]
        y2_min = box2["y"]
        x2_max = x2_min + box2["width"]
        y2_max = y2_min + box2["height"]

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_w = max(0.0, inter_x_max - inter_x_min)
        inter_h = max(0.0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        union = area1 + area2 - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union


    def dump_image_with_mask(self, frame, mask, output_file, obj_id=None, random_color=False):
        from matplotlib import pyplot as plt
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        # create an image file to display image overlayed with mask
        mask_image = (mask_image * 255).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGRA2BGR)
        mask_image = cv2.addWeighted(frame, 1.0, mask_image, 0.8, 0)
        logger.debug(f'Shapes: frame={frame.shape}, mask={mask.shape}, mask_image={mask_image.shape}')
        # save in file
        logger.debug(f'Saving image with mask to {output_file}')
        cv2.imwrite(output_file, mask_image)


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Returns the predicted mask for a smart keypoint that has been placed."""

        logger.info('='*80)
        logger.info('🎬 SAM2 VIDEO TRACKING STARTED')
        logger.info('='*80)

        # Check GPU health before starting
        try:
            check_gpu_health()
        except Exception as e:
            logger.error(f'❌ GPU health check failed: {e}')
            raise

        from_name, to_name, value = self.get_first_tag_occurence('VideoRectangle', 'Video')

        task = tasks[0]
        task_id = task['id']
        logger.info(f'📋 Processing task ID: {task_id}')

        # Get the video URL from the task
        video_url = task['data'][value]
        logger.info(f'🔗 Video URL: {video_url}')

        # Resolve relative URL if needed
        if not video_url.startswith("http") and video_url.startswith("/"):
            host = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL")
            if host:
                video_url = urljoin(host.rstrip("/"), video_url)
            else:
                logger.debug(
                    "Relative video URL %s found but LABEL_STUDIO_HOST/LABEL_STUDIO_URL is not set",
                    video_url,
                )

        # cache the video locally
        logger.info(f'⬇️  Downloading/caching video...')
        download_start = time.time()
        try:
            video_path = get_local_path(video_url, task_id=task_id)
            download_elapsed = time.time() - download_start
            logger.info(f'💾 Video cached at: {video_path} (took {download_elapsed:.2f}s)')

            # Verify file exists and is readable
            if not os.path.exists(video_path):
                raise FileNotFoundError(f'Video file not found after download: {video_path}')

            file_size_mb = os.path.getsize(video_path) / 1024**2
            logger.info(f'📦 Video file size: {file_size_mb:.2f}MB')

        except Exception as e:
            logger.error(f'❌ Video download/caching failed: {e}')
            raise

        # get prompts from context
        logger.info(f'🔍 Extracting prompts from annotation context...')
        prompts = self.get_prompts(context)
        all_obj_ids = set(p['obj_id'] for p in prompts)
        # create a map from obj_id to integer
        obj_ids = {obj_id: i for i, obj_id in enumerate(all_obj_ids)}
        # find the last frame index
        first_frame_idx = min(p['frame_idx'] for p in prompts) if prompts else 0
        last_frame_idx = max(p['frame_idx'] for p in prompts) if prompts else 0
        frames_count, duration = self._get_fps(context)
        fps = frames_count / duration

        logger.info(
            f'📍 Found {len(prompts)} prompt(s) for {len(obj_ids)} object(s), '
            f'keyframes range: [{first_frame_idx}, {last_frame_idx}]')
        logger.debug(f'Object ID mapping: {obj_ids}')

        # Temporal downsampling: determine stride between original frames fed into SAM2
        if TRACK_FPS > 0 and fps > 0:
            stride = max(1, round(fps / TRACK_FPS))
            logger.info(f'🎞️  Temporal downsampling enabled: original FPS={fps:.2f}, TRACK_FPS={TRACK_FPS:.2f}, stride={stride}')
        else:
            stride = 1
            logger.info(f'🎞️  Temporal downsampling disabled (TRACK_FPS={TRACK_FPS}, stride=1)')

        # Calculate frames to track: from first keyframe to end of video
        # If MAX_FRAMES_TO_TRACK is set (not None), use it as a hard limit
        if MAX_FRAMES_TO_TRACK > 0:
            frames_to_track = min(MAX_FRAMES_TO_TRACK, frames_count - first_frame_idx)
            logger.info(f'Tracking limited to {frames_to_track} frames (MAX_FRAMES_TO_TRACK={MAX_FRAMES_TO_TRACK})')
        else:
            frames_to_track = frames_count - first_frame_idx
            logger.info(f'Tracking full video: {frames_to_track} frames from frame {first_frame_idx} to {frames_count}')

        # Effective number of frames that SAM2 will actually see after temporal downsampling
        frames_to_track_sam2 = (frames_to_track + stride - 1) // stride
        logger.info(f'🎯 SAM2 will process approximately {frames_to_track_sam2} frames after downsampling (stride={stride})')
        
        # HARD LIMIT: Prevent SAM2 memory crashes for large videos
        # SAM2 loads all frames into memory during init_state(), estimate memory usage
        ESTIMATED_MEMORY_PER_FRAME_GB = 0.005  # ~5MB per frame at 1280x720 (more realistic)
        estimated_memory_gb = frames_to_track_sam2 * ESTIMATED_MEMORY_PER_FRAME_GB
        
        # Dynamic memory limit based on available GPU memory
        if DEVICE == 'cuda':
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # Use 90% of available GPU memory for safety
            MAX_SAFE_MEMORY_GB = total_gpu_memory * 0.9
            logger.info(f'🎮 GPU Memory Limit: {MAX_SAFE_MEMORY_GB:.1f}GB (90% of {total_gpu_memory:.1f}GB available)')
        else:
            MAX_SAFE_MEMORY_GB = 20  # Conservative limit for CPU
        
        if estimated_memory_gb > MAX_SAFE_MEMORY_GB:
            error_msg = (f'❌ SAM2 MEMORY LIMIT EXCEEDED: {frames_to_track} frames require '
                        f'~{estimated_memory_gb:.1f}GB RAM (limit: {MAX_SAFE_MEMORY_GB}GB).\n'
                        f'Solution: Set MAX_FRAMES_TO_TRACK environment variable to ≤{int(MAX_SAFE_MEMORY_GB/ESTIMATED_MEMORY_PER_FRAME_GB)} '
                        f'or process shorter video segments.')
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Split the video into frames using RAM disk for maximum performance
        # Create a unique cache directory based on video path and parameters
        import hashlib
        import platform
        
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        
        # Include frame range in hash to prevent cache contamination
        cache_params = f"{video_hash}_{first_frame_idx}_{frames_to_track}"
        cache_hash = hashlib.md5(cache_params.encode()).hexdigest()[:8]
        
        ram_disk_path = "./video_cache"
        cache_dir = os.path.join(ram_disk_path, f'video_{cache_hash}')
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f'📁 Using high-performance cache directory: {cache_dir}')
        
        # Check if frames are already cached
        cached_frames = len([f for f in os.listdir(cache_dir) if f.endswith('.jpg')])
        if cached_frames >= frames_to_track:
            logger.info(f'✅ Found {cached_frames} cached frames - skipping extraction')
        else:
            logger.info(f'📥 Extracting frames to cache (found {cached_frames} cached frames)...')
        
        # Extract frames to high-performance cache (streaming, no memory accumulation)
        # Run split_frames as generator to avoid loading all frames into memory
        # SAM2 only needs the frames on disk, not in memory
        frame_generator = self.split_frames(
            video_path, cache_dir,
            start_frame=first_frame_idx,
            end_frame=first_frame_idx + frames_to_track,
            stride=stride,
        )
        
        # Consume generator to extract frames to disk without storing in memory
        extracted_frames = []
        for frame_filename, frame_data in frame_generator:
            extracted_frames.append(frame_filename)
            # Explicitly clean up frame data to prevent memory accumulation
            del frame_data
        
        # Get first frame for dimensions (read from disk to avoid memory usage)
        if extracted_frames:
            first_frame_path = extracted_frames[0]
            first_frame = cv2.imread(first_frame_path)
            if first_frame is None:
                raise ValueError(f"Failed to read first frame: {first_frame_path}")
            height, width, _ = first_frame.shape
            del first_frame  # Clean up immediately
        else:
            raise ValueError("No frames were extracted")
            
        logger.info(f'📐 Video dimensions: {width}x{height}')

        # get inference state
        logger.info(f'🧠 Initializing SAM2 inference state...')
        init_start = time.time()
        
        # Monitor memory before SAM2 init to catch OOM issues
        if DEVICE == 'cuda':
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f'💾 GPU Memory before SAM2 init: {gpu_allocated:.2f}GB allocated, {gpu_reserved:.2f}GB reserved')
        
        try:
            # Wrap SAM2 init in timeout context and detailed logging
            with timeout_context(300, "SAM2 initialization"):  # 5 minute timeout
                inference_state = get_inference_state(cache_dir)
                predictor.reset_state(inference_state)
            
            init_elapsed = time.time() - init_start
            logger.info(f'✅ Inference state initialized (took {init_elapsed:.2f}s)')
            
            # Check memory after successful init
            if DEVICE == 'cuda':
                gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
                gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f'💾 GPU Memory after SAM2 init: {gpu_allocated:.2f}GB allocated, {gpu_reserved:.2f}GB reserved')
                
        except TimeoutError as e:
            logger.error(f'❌ SAM2 initialization timed out: {e}')
            raise RuntimeError(f'SAM2 initialization timed out after 300 seconds. Try reducing MAX_FRAMES_TO_TRACK or using a smaller video.')
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "allocate" in str(e).lower():
                logger.error(f'❌ SAM2 OOM during init: {e}')
                raise RuntimeError(f'SAM2 ran out of memory during initialization. Try reducing MAX_FRAMES_TO_TRACK from {frames_to_track} to {frames_to_track//2}.')
            else:
                logger.error(f'❌ SAM2 runtime error during init: {e}')
                raise
        except Exception as e:
            logger.error(f'❌ Unexpected error during SAM2 initialization: {e}')
            raise RuntimeError(f'SAM2 initialization failed: {e}')

        logger.info(f'📌 Adding {len(prompts)} tracking prompt(s) to SAM2...')
        for idx, prompt in enumerate(prompts, 1):
            # multiply points by the frame size
            prompt['points'][:, 0] *= width
            prompt['points'][:, 1] *= height

            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=prompt['frame_idx'],
                obj_id=obj_ids[prompt['obj_id']],
                points=prompt['points'],
                labels=prompt['labels']
            )
            logger.info(f'  ✓ Prompt {idx}/{len(prompts)}: frame={prompt["frame_idx"]}, obj_id={prompt["obj_id"]}, points={len(prompt["points"])}')

        logger.info(f'✅ All prompts added successfully')

        # Dictionary to store sequences per object (for multi-person tracking)
        from collections import defaultdict
        sequences_by_obj = defaultdict(list)

        debug_dir = './debug-frames'
        os.makedirs(debug_dir, exist_ok=True)

        logger.info(f'🚀 Starting SAM2 video propagation from frame {first_frame_idx} to {first_frame_idx + frames_to_track}')
        logger.info(f'🎯 Tracking {len(obj_ids)} object(s) across {frames_to_track} frames (SAM2 frames={frames_to_track_sam2}, stride={stride})')

        # Create progress bar for tracking (SAM2-local frames)
        pbar = tqdm(
            total=frames_to_track_sam2,
            desc="🎥 Tracking frames",
            unit="frame",
            ncols=100
        )

        last_heartbeat = time.time()
        last_memory_check = time.time()
        HEARTBEAT_INTERVAL = 30  # seconds
        MEMORY_CHECK_INTERVAL = 60  # seconds
        propagation_start = time.time()

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=frames_to_track_sam2,
        ):
            # Map SAM2-local frame index back to original video frame index
            real_frame_idx = first_frame_idx + out_frame_idx * stride

            # Heartbeat logging
            now = time.time()
            if now - last_heartbeat > HEARTBEAT_INTERVAL:
                elapsed = now - propagation_start
                logger.info(f'💓 Heartbeat: Processing original frame {real_frame_idx}, '
                           f'elapsed: {elapsed:.1f}s, progress: {out_frame_idx}/{frames_to_track_sam2}')
                last_heartbeat = now

            # Memory monitoring
            if DEVICE == 'cuda' and now - last_memory_check > MEMORY_CHECK_INTERVAL:
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f'💾 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved')
                last_memory_check = now

            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()

                # to debug, save the mask as an image
                # self.dump_image_with_mask(frames[out_frame_idx][1], mask, f'{debug_dir}/{out_frame_idx:05d}_{out_obj_id}.jpg', obj_id=out_obj_id, random_color=True)

                bbox = self.convert_mask_to_bbox(mask)
                if bbox:
                    # Append to the specific object's sequence
                    sequences_by_obj[out_obj_id].append({
                        'frame': real_frame_idx + 1,
                        'x': bbox['x'],
                        'y': bbox['y'],
                        'width': bbox['width'],
                        'height': bbox['height'],
                        'enabled': True,
                        'rotation': 0,
                        'time': real_frame_idx / fps
                    })

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'frame': real_frame_idx + 1,
                'objects': len(out_obj_ids)
            })

        pbar.close()
        propagation_elapsed = time.time() - propagation_start
        logger.info(f'✅ Video propagation complete in {propagation_elapsed:.2f}s!')

            # Create a map from obj_id (SAM2 internal ID) to original annotation ID
        # obj_ids maps original annotation ID -> SAM2 internal ID
        # We need the reverse: SAM2 internal ID -> original annotation ID
        reverse_obj_ids = {v: k for k, v in obj_ids.items()}

        # Get keyframes from context (one result per person)
        context_results = {r['id']: r for r in context['result']}

        # Build separate regions for each tracked person (multi-person tracking)
        regions = []
        for sam_obj_id, predicted_sequence in sequences_by_obj.items():
            # Get the original annotation ID
            original_obj_id = reverse_obj_ids.get(sam_obj_id)

            if original_obj_id not in context_results:
                logger.warning(f'Could not find context result for obj_id {original_obj_id}')
                continue

            # Get the original keyframes from context
            context_result = context_results[original_obj_id]
            original_keyframes = context_result['value'].get('sequence', [])

            # Get labels from context
            raw_labels = context_result['value'].get('labels')
            if isinstance(raw_labels, str):
                labels = [raw_labels.strip()] if raw_labels.strip() else []
            elif isinstance(raw_labels, list):
                labels = [lbl.strip() for lbl in raw_labels if isinstance(lbl, str) and lbl.strip()]
            else:
                labels = []
            if not labels:
                labels = ['Person']

            # Sort predicted sequence to ensure temporal order
            predicted_sequence = sorted(predicted_sequence, key=lambda x: x['frame'])

            # Downsample tracked frames into sparse keyframes using IoU threshold
            sparse_tracked = []
            last_kept_box = None

            for item in predicted_sequence:
                current_box = {
                    "x": item["x"],
                    "y": item["y"],
                    "width": item["width"],
                    "height": item["height"],
                }

                if last_kept_box is None:
                    sparse_tracked.append(item)
                    last_kept_box = current_box
                    continue

                iou = self._bbox_iou(last_kept_box, current_box)
                if iou < 0.2:
                    sparse_tracked.append(item)
                    last_kept_box = current_box

            # Always ensure we keep the last tracked frame if any tracking exists
            if predicted_sequence:
                last_tracked = predicted_sequence[-1]
                if not sparse_tracked or sparse_tracked[-1]["frame"] != last_tracked["frame"]:
                    sparse_tracked.append(last_tracked)

            # Lifecycle handling within a single region:
            # - Start enabled when the object first appears
            # - If there is a frame gap > 1, mark the previous frame as disabled
            if sparse_tracked:
                prev = None
                for i, box in enumerate(sparse_tracked):
                    # Default: object is visible on each sparsified frame
                    box["enabled"] = True

                    if prev is not None and box["frame"] - prev["frame"] > 1:
                        # Close the previous stint when there is a temporal gap
                        sparse_tracked[i - 1]["enabled"] = False

                    prev = box

            # Merge original keyframes with sparsified tracked frames
            merged_sequence = original_keyframes + sparse_tracked

            # If we have any tracking, add an explicit disabled frame right after
            # the last tracked frame so the region lifecycle ends with tracking
            if predicted_sequence:
                last_tracked_frame = predicted_sequence[-1]["frame"]
                off_frame = min(last_tracked_frame + 1, frames_count)

                if sparse_tracked:
                    last_box = sparse_tracked[-1]
                    off_item = {
                        "frame": off_frame,
                        "x": last_box["x"],
                        "y": last_box["y"],
                        "width": last_box["width"],
                        "height": last_box["height"],
                        "enabled": False,
                        "rotation": last_box.get("rotation", 0),
                        "time": (off_frame - 1) / fps,
                    }
                    merged_sequence.append(off_item)

            # Sort by frame number to maintain temporal order
            merged_sequence = sorted(merged_sequence, key=lambda x: x['frame'])

            # Calculate score (use 1.0 as default for SAM2 tracking)
            avg_score = 1.0

            region = {
                'value': {
                    'framesCount': frames_count,
                    'duration': duration,
                    'sequence': merged_sequence,
                    'labels': labels,
                },
                'from_name': from_name,
                'to_name': to_name,
                'type': 'videorectangle',
                'origin': 'manual',
                'id': original_obj_id,
                'score': avg_score,
            }
            regions.append(region)
            logger.info(
                f'Created region for person {original_obj_id}: '
                f'{len(original_keyframes)} keyframes + {len(predicted_sequence)} tracked frames = '
                f'{len(merged_sequence)} total frames'
            )

        # Calculate score (use 1.0 as default for SAM2 tracking)
        avg_score = 1.0

        prediction = {
            "result": regions,
            "score": avg_score,
            "model_version": self.model_version,
        }
        logger.debug(f'Prediction with {len(regions)} regions: {prediction}')

        logger.info('='*80)
        logger.info(f'✅ SAM2 TRACKING COMPLETE!')
        logger.info(f'📊 Summary:')
        logger.info(f'   • Objects tracked: {len(regions)}')
        logger.info(f'   • Total frames processed: {frames_to_track}')
        for region in regions:
            obj_id = region['id']
            total_frames = len(region['value']['sequence'])
            logger.info(f'   • Object {obj_id}: {total_frames} frames')
        logger.info('='*80)

        # Clean up frame cache directory for this run
        try:
            import shutil
            shutil.rmtree(cache_dir)
            logger.info(f'🧹 Deleted frame cache: {cache_dir}')
        except Exception as e:
            logger.warning(f'⚠️ Failed to delete frame cache {cache_dir}: {e}')

        return ModelResponse(predictions=[prediction])
