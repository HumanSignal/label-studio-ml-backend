import torch
import numpy as np
import os
import logging
import signal
import time
import av
from urllib.parse import urljoin
from contextlib import contextmanager
from collections import defaultdict

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from PIL import Image
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

        logger.info(f'GPU: {props.name}')
        logger.info(f'GPU Memory: {allocated:.2f}GB allocated, '
                   f'{cached:.2f}GB reserved, {total_memory:.2f}GB total')

        if allocated > total_memory * 0.9:
            logger.warning(f'GPU memory usage high: {allocated/total_memory*100:.1f}%')

        return True
    else:
        logger.info(f'Running on CPU (DEVICE={DEVICE})')
        return False


# ---------------------------------------------------------------------------
# Module-level configuration & model loading
# ---------------------------------------------------------------------------

DEVICE = os.getenv('DEVICE', 'cuda')
HINTS = os.getenv('HINTS', 'false').lower() == 'true'
MODEL_NAME = os.getenv('MODEL_NAME', 'facebook/sam3')
MAX_FRAMES_TO_TRACK = int(os.getenv('MAX_FRAMES_TO_TRACK', 1000))
TRACK_FPS = float(os.getenv('TRACK_FPS', '0'))
PROCESSING_MODE = os.getenv('PROCESSING_MODE', 'streaming')
PROMPT_TEXT = os.getenv('PROMPT_TEXT', '')

DTYPE = torch.bfloat16 if DEVICE == 'cuda' else torch.float32

if HINTS:
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    model = Sam3VideoModel.from_pretrained(MODEL_NAME).to(DEVICE, dtype=DTYPE)
    processor = Sam3VideoProcessor.from_pretrained(MODEL_NAME)
else:
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
    model = Sam3TrackerVideoModel.from_pretrained(MODEL_NAME).to(DEVICE, dtype=DTYPE)
    processor = Sam3TrackerVideoProcessor.from_pretrained(MODEL_NAME)


class NewModel(LabelStudioMLBase):
    """SAM3 Video Tracking ML Backend for Label Studio"""

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "sam3")

    # ------------------------------------------------------------------
    # Video utilities (PyAV-based, replaces split_frames / cv2 decode)
    # ------------------------------------------------------------------

    def get_video_dimensions_pyav(self, video_path):
        """Return (width, height, total_frames) using PyAV.

        Opens/closes the container quickly. Falls back to estimating
        total_frames from duration * fps if the container reports 0.
        """
        container = av.open(video_path)
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        total_frames = stream.frames
        if total_frames == 0 and stream.duration and stream.time_base:
            duration_s = float(stream.duration * stream.time_base)
            avg_fps = float(stream.average_rate) if stream.average_rate else 30.0
            total_frames = int(duration_s * avg_fps)
        container.close()
        return width, height, total_frames

    def decode_video_pyav(self, video_path, start_frame, end_frame, stride=1):
        """Yield (real_frame_idx, PIL.Image) tuples via PyAV.

        Replaces split_frames(). No disk I/O — frames stay in memory only
        long enough to be consumed.  Supports seeking for chunked batch
        mode and applies temporal downsampling via *stride*.
        """
        container = av.open(video_path)
        stream = container.streams.video[0]
        # Seek to approximate position if start_frame > 0
        if start_frame > 0 and stream.average_rate:
            avg_fps = float(stream.average_rate)
            target_ts = int(start_frame / avg_fps / stream.time_base)
            container.seek(target_ts, stream=stream)

        frame_idx = 0
        yielded = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                # Patients with seek overshoot — track absolute position
                if frame_idx < start_frame:
                    frame_idx += 1
                    continue
                if frame_idx >= end_frame:
                    container.close()
                    return
                # Apply stride
                if stride > 1 and (frame_idx - start_frame) % stride != 0:
                    frame_idx += 1
                    continue
                pil_img = frame.to_image()  # RGB PIL.Image
                yield frame_idx, pil_img
                yielded += 1
                frame_idx += 1

        container.close()
        logger.info(f'PyAV decode complete: yielded {yielded} frames '
                    f'[{start_frame}, {end_frame}) stride={stride}')

    # ------------------------------------------------------------------
    # Prompt extraction
    # ------------------------------------------------------------------

    def _get_fps(self, context):
        # get the fps from the context
        frames_count = context['result'][0]['value']['framesCount']
        duration = context['result'][0]['value']['duration']
        return frames_count, duration

    def get_prompts_native(self, context, width, height) -> List[Dict]:
        """Extract bbox prompts from LS context as xyxy pixel coords.

        Coordinate conversion follows the canonical pattern from
        initial_seeding_video_boxes._percent_xywh_to_xyxy_px():
          x1 = (x_pct / 100) * W
          y1 = (y_pct / 100) * H
          x2 = x1 + (w_pct / 100) * W
          y2 = y1 + (h_pct / 100) * H

        Frame indexing: LS uses 1-based frames (obj['frame']).
        We convert to 0-based here (frame_idx = obj['frame'] - 1).
        """
        prompts = []
        for ctx in context['result']:
            obj_id = ctx['id']
            labels = ctx['value'].get('labels', ['Person'])
            if isinstance(labels, str):
                labels = [labels.strip()] if labels.strip() else ['Person']
            for obj in ctx['value']['sequence']:
                # LS coords: top-left (x, y) + width/height, all in percent [0, 100]
                x1 = (obj['x'] / 100.0) * width
                y1 = (obj['y'] / 100.0) * height
                x2 = x1 + (obj['width'] / 100.0) * width
                y2 = y1 + (obj['height'] / 100.0) * height
                frame_idx = obj['frame'] - 1  # LS 1-based -> 0-based
                prompts.append({
                    'box': [x1, y1, x2, y2],
                    'frame_idx': frame_idx,
                    'obj_id': obj_id,
                    'labels': labels,
                })
        return prompts

    # ------------------------------------------------------------------
    # Mask / bbox utilities (preserved from SAM2)
    # ------------------------------------------------------------------

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
        """Debug helper to save frame with mask overlay using PIL (no cv2 dependency)."""
        from matplotlib import pyplot as plt
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        # Create RGBA mask overlay (values in [0, 1])
        mask_rgba = (mask_image * 255).astype(np.uint8)
        # mask_rgba is (H, W, 4) with RGBA

        # Convert frame to RGB if needed (frame may be BGR from legacy code or RGB from PIL)
        if isinstance(frame, Image.Image):
            frame_rgb = np.array(frame.convert('RGB'))
        else:
            # Assume numpy array; if 3 channels, treat as RGB
            frame_rgb = frame if frame.shape[-1] == 3 else frame[..., :3]

        # Alpha blend: out = frame * (1 - alpha) + mask_rgb * alpha
        alpha = mask_rgba[..., 3:4].astype(np.float32) / 255.0
        mask_rgb = mask_rgba[..., :3].astype(np.float32)
        frame_f = frame_rgb.astype(np.float32)
        blended = frame_f * (1.0 - alpha * 0.8) + mask_rgb * (alpha * 0.8)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        logger.debug(f'Shapes: frame={frame_rgb.shape}, mask={mask.shape}, blended={blended.shape}')
        # Save using PIL
        logger.debug(f'Saving image with mask to {output_file}')
        Image.fromarray(blended).save(output_file)

    # ------------------------------------------------------------------
    # Region-building helpers
    # ------------------------------------------------------------------

    def _xyxy_to_ls_percent(self, box_xyxy, width, height):
        """Convert pixel xyxy box to LS percent xywh dict.

        Follows the canonical pattern from seeding_common.xyxy_to_percent().
        """
        x1, y1, x2, y2 = box_xyxy
        x_pct = (x1 / width) * 100.0
        y_pct = (y1 / height) * 100.0
        w_pct = ((x2 - x1) / width) * 100.0
        h_pct = ((y2 - y1) / height) * 100.0
        return {
            "x": round(max(0.0, x_pct), 2),
            "y": round(max(0.0, y_pct), 2),
            "width": round(min(100.0, w_pct), 2),
            "height": round(min(100.0, h_pct), 2),
        }

    def _build_regions(self, sequences_by_obj, reverse_obj_ids, context,
                       from_name, to_name, frames_count, duration, fps,
                       score_map=None):
        """Build LS VideoRectangle regions from tracked sequences.

        This is the shared region-building logic preserved from the SAM2
        model.py (lines 686-834). It handles:
        - Sparsification by IoU (< 0.2)
        - Lifecycle handling (enabled/disabled based on frame gaps)
        - Merging original keyframes with tracked frames
        - Off-frame insertion after tracking ends
        """
        context_results = {r['id']: r for r in context['result']}
        regions = []

        for obj_key, predicted_sequence in sequences_by_obj.items():
            # Resolve the original annotation ID
            if reverse_obj_ids is not None:
                original_obj_id = reverse_obj_ids.get(obj_key)
            else:
                original_obj_id = obj_key

            if original_obj_id is not None and original_obj_id in context_results:
                context_result = context_results[original_obj_id]
                original_keyframes = context_result['value'].get('sequence', [])
                raw_labels = context_result['value'].get('labels')
            else:
                original_keyframes = []
                raw_labels = None

            # Normalise labels
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
                    box["enabled"] = True

                    if prev is not None and box["frame"] - prev["frame"] > 1:
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

            # Get score for this object
            avg_score = 1.0
            if score_map and obj_key in score_map:
                avg_score = score_map[obj_key]

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
                'id': original_obj_id if original_obj_id else str(obj_key),
                'score': avg_score,
            }
            regions.append(region)
            logger.info(
                f'Created region for {original_obj_id}: '
                f'{len(original_keyframes)} keyframes + {len(predicted_sequence)} tracked = '
                f'{len(merged_sequence)} total frames'
            )

        return regions

    # ------------------------------------------------------------------
    # Tracker branch (HINTS=false): Sam3TrackerVideoModel
    # ------------------------------------------------------------------

    def _predict_tracker_streaming(self, video_path, prompts, obj_ids,
                                   first_frame_idx, frames_to_track, stride,
                                   width, height, fps):
        """Streaming mode for Sam3TrackerVideoModel.

        Process frame-by-frame via PyAV — constant memory.
        """
        sequences_by_obj = defaultdict(list)
        score_accum = defaultdict(list)

        # Group prompts by their (session-local) frame index
        # Session frame idx is a monotonic counter (0, 1, 2, ...)
        # We need to map real_frame_idx -> session_frame_idx
        end_frame = first_frame_idx + frames_to_track

        # Pre-scan to know which real frames have prompts
        prompts_by_real_frame = defaultdict(list)
        for p in prompts:
            prompts_by_real_frame[p['frame_idx']].append(p)

        session = processor.init_video_session(inference_device=DEVICE, dtype=DTYPE)
        session_frame_idx = 0

        pbar = tqdm(
            total=(frames_to_track + stride - 1) // stride,
            desc="Tracking frames (streaming)",
            unit="frame",
            ncols=100,
        )

        last_heartbeat = time.time()
        propagation_start = time.time()

        for real_idx, pil_img in self.decode_video_pyav(
            video_path, start_frame=first_frame_idx, end_frame=end_frame, stride=stride
        ):
            now = time.time()
            if now - last_heartbeat > 30:
                elapsed = now - propagation_start
                logger.info(f'Heartbeat: frame {real_idx}, session_idx {session_frame_idx}, '
                           f'elapsed {elapsed:.1f}s')
                last_heartbeat = now

            # Add prompts when we reach their keyframe
            if real_idx in prompts_by_real_frame:
                for p in prompts_by_real_frame[real_idx]:
                    int_obj_id = obj_ids[p['obj_id']]
                    inputs = processor(images=pil_img, device=DEVICE, return_tensors="pt")
                    processor.add_inputs_to_inference_session(
                        session,
                        frame_idx=session_frame_idx,
                        obj_ids=[int_obj_id],
                        input_boxes=[[p['box']]],
                        original_size=inputs.original_sizes[0],
                    )

            # Process frame
            inputs = processor(images=pil_img, device=DEVICE, return_tensors="pt")
            with torch.inference_mode():
                output = model(inference_session=session, frame=inputs.pixel_values[0])

            masks = processor.post_process_masks(
                [output.pred_masks],
                original_sizes=inputs.original_sizes,
                binarize=True,
            )[0]

            # Extract bboxes from masks for each tracked object
            if output.object_ids is not None:
                for i, out_obj_id in enumerate(output.object_ids):
                    out_obj_id_int = int(out_obj_id)
                    mask_np = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]
                    bbox = self.convert_mask_to_bbox(mask_np)
                    if bbox:
                        sequences_by_obj[out_obj_id_int].append({
                            'frame': real_idx + 1,  # LS 1-based
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height'],
                            'enabled': True,
                            'rotation': 0,
                            'time': real_idx / fps,
                        })
                    # Collect scores
                    if hasattr(output, 'object_score_logits') and output.object_score_logits is not None:
                        score = torch.sigmoid(output.object_score_logits[i]).item()
                        score_accum[out_obj_id_int].append(score)

            session_frame_idx += 1
            pbar.update(1)
            pbar.set_postfix({'frame': real_idx + 1})

        pbar.close()
        logger.info(f'Streaming propagation complete in {time.time() - propagation_start:.2f}s')

        # Compute average scores per object
        score_map = {}
        for oid, scores in score_accum.items():
            score_map[oid] = sum(scores) / len(scores) if scores else 1.0

        return sequences_by_obj, score_map

    def _predict_tracker_chunked(self, video_path, prompts, obj_ids,
                                 first_frame_idx, frames_to_track, stride,
                                 width, height, fps):
        """Chunked batch mode for Sam3TrackerVideoModel.

        Decode [start_frame, end_frame] into memory, then propagate with
        bidirectional temporal context.
        """
        sequences_by_obj = defaultdict(list)
        score_accum = defaultdict(list)
        end_frame = first_frame_idx + frames_to_track

        # Decode all frames into memory
        logger.info(f'Chunked batch: decoding frames [{first_frame_idx}, {end_frame}) stride={stride}')
        frames_list = []
        frame_idx_map = {}  # session_idx -> real_frame_idx
        for real_idx, pil_img in self.decode_video_pyav(
            video_path, start_frame=first_frame_idx, end_frame=end_frame, stride=stride
        ):
            session_idx = len(frames_list)
            frame_idx_map[session_idx] = real_idx
            frames_list.append(pil_img)

        logger.info(f'Loaded {len(frames_list)} frames into memory')

        session = processor.init_video_session(
            video=frames_list, inference_device=DEVICE, dtype=DTYPE
        )

        # Build reverse map: real_frame_idx -> session_idx
        real_to_session = {v: k for k, v in frame_idx_map.items()}

        # Add all prompts at their session-local frame indices
        for p in prompts:
            local_idx = real_to_session.get(p['frame_idx'])
            if local_idx is None:
                # Prompt frame not in decoded set (e.g. stride skipped it)
                # Find nearest session frame
                nearest = min(frame_idx_map.keys(),
                              key=lambda k: abs(frame_idx_map[k] - p['frame_idx']))
                local_idx = nearest
                logger.warning(f'Prompt frame {p["frame_idx"]} not decoded; '
                              f'snapping to session frame {local_idx} (real {frame_idx_map[local_idx]})')

            int_obj_id = obj_ids[p['obj_id']]
            inputs = processor(images=frames_list[local_idx], device=DEVICE, return_tensors="pt")
            processor.add_inputs_to_inference_session(
                session,
                frame_idx=local_idx,
                obj_ids=[int_obj_id],
                input_boxes=[[p['box']]],
                original_size=inputs.original_sizes[0],
            )

        # Propagate
        pbar = tqdm(
            total=len(frames_list),
            desc="Tracking frames (chunked)",
            unit="frame",
            ncols=100,
        )

        with torch.inference_mode():
            for output in model.propagate_in_video_iterator(session):
                session_idx = output.frame_idx
                real_idx = frame_idx_map.get(session_idx, session_idx)

                masks = processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=[[height, width]],
                    binarize=True,
                )[0]

                if output.object_ids is not None:
                    for i, out_obj_id in enumerate(output.object_ids):
                        out_obj_id_int = int(out_obj_id)
                        mask_np = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]
                        bbox = self.convert_mask_to_bbox(mask_np)
                        if bbox:
                            sequences_by_obj[out_obj_id_int].append({
                                'frame': real_idx + 1,
                                'x': bbox['x'],
                                'y': bbox['y'],
                                'width': bbox['width'],
                                'height': bbox['height'],
                                'enabled': True,
                                'rotation': 0,
                                'time': real_idx / fps,
                            })
                        if hasattr(output, 'object_score_logits') and output.object_score_logits is not None:
                            score = torch.sigmoid(output.object_score_logits[i]).item()
                            score_accum[out_obj_id_int].append(score)

                pbar.update(1)
                pbar.set_postfix({'frame': real_idx + 1})

        pbar.close()

        score_map = {}
        for oid, scores in score_accum.items():
            score_map[oid] = sum(scores) / len(scores) if scores else 1.0

        return sequences_by_obj, score_map

    # ------------------------------------------------------------------
    # PCS / hints branch (HINTS=true): Sam3VideoModel
    # ------------------------------------------------------------------

    def _predict_pcs_streaming(self, video_path, prompt_text,
                               first_frame_idx, frames_to_track, stride,
                               width, height, fps):
        """Streaming mode for Sam3VideoModel (text-based detection).

        Replaces Grounding DINO — no drawn bboxes needed.
        """
        sequences_by_obj = defaultdict(list)
        score_accum = defaultdict(list)
        end_frame = first_frame_idx + frames_to_track

        session = processor.init_video_session(inference_device=DEVICE, dtype=DTYPE)
        session = processor.add_text_prompt(inference_session=session, text=prompt_text)

        pbar = tqdm(
            total=(frames_to_track + stride - 1) // stride,
            desc="PCS detection (streaming)",
            unit="frame",
            ncols=100,
        )

        last_heartbeat = time.time()
        propagation_start = time.time()

        for real_idx, pil_img in self.decode_video_pyav(
            video_path, start_frame=first_frame_idx, end_frame=end_frame, stride=stride
        ):
            now = time.time()
            if now - last_heartbeat > 30:
                elapsed = now - propagation_start
                logger.info(f'Heartbeat: frame {real_idx}, elapsed {elapsed:.1f}s')
                last_heartbeat = now

            inputs = processor(images=pil_img, device=DEVICE, return_tensors="pt")
            with torch.inference_mode():
                output = model(inference_session=session, frame=inputs.pixel_values[0], reverse=False)

            processed = processor.postprocess_outputs(
                session, output, original_sizes=inputs.original_sizes
            )

            # processed contains 'object_ids', 'scores', 'boxes' (xyxy pixels)
            obj_ids_out = processed.get('object_ids', [])
            scores_out = processed.get('scores', [])
            boxes_out = processed.get('boxes', [])

            for i, oid in enumerate(obj_ids_out):
                oid_key = int(oid) if hasattr(oid, 'item') else oid
                if i < len(boxes_out):
                    box_xyxy = boxes_out[i]
                    if hasattr(box_xyxy, 'tolist'):
                        box_xyxy = box_xyxy.tolist()
                    bbox_ls = self._xyxy_to_ls_percent(box_xyxy, width, height)
                    sequences_by_obj[oid_key].append({
                        'frame': real_idx + 1,
                        'x': bbox_ls['x'],
                        'y': bbox_ls['y'],
                        'width': bbox_ls['width'],
                        'height': bbox_ls['height'],
                        'enabled': True,
                        'rotation': 0,
                        'time': real_idx / fps,
                    })
                if i < len(scores_out):
                    s = scores_out[i]
                    score_accum[oid_key].append(float(s) if hasattr(s, 'item') else s)

            pbar.update(1)
            pbar.set_postfix({'frame': real_idx + 1, 'objects': len(obj_ids_out)})

        pbar.close()
        logger.info(f'PCS streaming complete in {time.time() - propagation_start:.2f}s')

        score_map = {}
        for oid, scores in score_accum.items():
            score_map[oid] = sum(scores) / len(scores) if scores else 1.0

        return sequences_by_obj, score_map

    def _predict_pcs_chunked(self, video_path, prompt_text,
                             first_frame_idx, frames_to_track, stride,
                             width, height, fps):
        """Chunked batch mode for Sam3VideoModel (text-based detection)."""
        sequences_by_obj = defaultdict(list)
        score_accum = defaultdict(list)
        end_frame = first_frame_idx + frames_to_track

        # Decode all frames into memory
        logger.info(f'PCS chunked: decoding frames [{first_frame_idx}, {end_frame}) stride={stride}')
        frames_list = []
        frame_idx_map = {}
        for real_idx, pil_img in self.decode_video_pyav(
            video_path, start_frame=first_frame_idx, end_frame=end_frame, stride=stride
        ):
            session_idx = len(frames_list)
            frame_idx_map[session_idx] = real_idx
            frames_list.append(pil_img)

        logger.info(f'Loaded {len(frames_list)} frames into memory')

        session = processor.init_video_session(
            video=frames_list, inference_device=DEVICE, dtype=DTYPE
        )
        session = processor.add_text_prompt(inference_session=session, text=prompt_text)

        pbar = tqdm(
            total=len(frames_list),
            desc="PCS detection (chunked)",
            unit="frame",
            ncols=100,
        )

        with torch.inference_mode():
            for output in model.propagate_in_video_iterator(
                session, max_frame_num_to_track=len(frames_list)
            ):
                session_idx = output.frame_idx
                real_idx = frame_idx_map.get(session_idx, session_idx)

                processed = processor.postprocess_outputs(session, output)

                obj_ids_out = processed.get('object_ids', [])
                scores_out = processed.get('scores', [])
                boxes_out = processed.get('boxes', [])

                for i, oid in enumerate(obj_ids_out):
                    oid_key = int(oid) if hasattr(oid, 'item') else oid
                    if i < len(boxes_out):
                        box_xyxy = boxes_out[i]
                        if hasattr(box_xyxy, 'tolist'):
                            box_xyxy = box_xyxy.tolist()
                        bbox_ls = self._xyxy_to_ls_percent(box_xyxy, width, height)
                        sequences_by_obj[oid_key].append({
                            'frame': real_idx + 1,
                            'x': bbox_ls['x'],
                            'y': bbox_ls['y'],
                            'width': bbox_ls['width'],
                            'height': bbox_ls['height'],
                            'enabled': True,
                            'rotation': 0,
                            'time': real_idx / fps,
                        })
                    if i < len(scores_out):
                        s = scores_out[i]
                        score_accum[oid_key].append(float(s) if hasattr(s, 'item') else s)

                pbar.update(1)
                pbar.set_postfix({'frame': real_idx + 1})

        pbar.close()

        score_map = {}
        for oid, scores in score_accum.items():
            score_map[oid] = sum(scores) / len(scores) if scores else 1.0

        return sequences_by_obj, score_map

    # ------------------------------------------------------------------
    # Main predict entry point
    # ------------------------------------------------------------------

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Returns the predicted bounding boxes for video tracking."""

        logger.info('=' * 80)
        logger.info('SAM3 VIDEO TRACKING STARTED')
        logger.info(f'Mode: {"PCS (hints)" if HINTS else "Tracker"}, '
                    f'Processing: {PROCESSING_MODE}')
        logger.info('=' * 80)

        # Check GPU health before starting
        try:
            check_gpu_health()
        except Exception as e:
            logger.error(f'GPU health check failed: {e}')
            raise

        from_name, to_name, value = self.get_first_tag_occurence('VideoRectangle', 'Video')

        task = tasks[0]
        task_id = task['id']
        logger.info(f'Processing task ID: {task_id}')

        # Get the video URL from the task
        video_url = task['data'][value]
        logger.info(f'Video URL: {video_url}')

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

        # Cache the video locally
        logger.info(f'Downloading/caching video...')
        download_start = time.time()
        try:
            video_path = get_local_path(video_url, task_id=task_id)
            download_elapsed = time.time() - download_start
            logger.info(f'Video cached at: {video_path} (took {download_elapsed:.2f}s)')

            # Verify file exists and is readable
            if not os.path.exists(video_path):
                raise FileNotFoundError(f'Video file not found after download: {video_path}')

            file_size_mb = os.path.getsize(video_path) / 1024**2
            logger.info(f'Video file size: {file_size_mb:.2f}MB')

        except Exception as e:
            logger.error(f'Video download/caching failed: {e}')
            raise

        # Get video dimensions via PyAV
        width, height, _ = self.get_video_dimensions_pyav(video_path)
        logger.info(f'Video dimensions: {width}x{height}')

        # Get FPS info from context
        frames_count, duration = self._get_fps(context)
        fps = frames_count / duration

        # ---------------------------------------------------------------
        # Branch: HINTS=true (PCS text detection) vs HINTS=false (Tracker)
        # ---------------------------------------------------------------
        if HINTS:
            # PCS mode: text-based detection, no drawn bboxes needed
            prompt_text = PROMPT_TEXT or 'person'
            logger.info(f'PCS mode with prompt: "{prompt_text}"')

            first_frame_idx = 0

            # Temporal downsampling
            if TRACK_FPS > 0 and fps > 0:
                stride = max(1, round(fps / TRACK_FPS))
                logger.info(f'Temporal downsampling: original FPS={fps:.2f}, '
                           f'TRACK_FPS={TRACK_FPS:.2f}, stride={stride}')
            else:
                stride = 1

            # Frames to track
            if MAX_FRAMES_TO_TRACK > 0:
                frames_to_track = min(MAX_FRAMES_TO_TRACK, frames_count)
            else:
                frames_to_track = frames_count

            logger.info(f'Processing {frames_to_track} frames, stride={stride}')

            if PROCESSING_MODE == 'chunked_batch':
                sequences_by_obj, score_map = self._predict_pcs_chunked(
                    video_path, prompt_text,
                    first_frame_idx, frames_to_track, stride,
                    width, height, fps,
                )
            else:
                sequences_by_obj, score_map = self._predict_pcs_streaming(
                    video_path, prompt_text,
                    first_frame_idx, frames_to_track, stride,
                    width, height, fps,
                )

            # For PCS, auto-assigned object IDs — no reverse_obj_ids needed
            regions = self._build_regions(
                sequences_by_obj, reverse_obj_ids=None, context=context,
                from_name=from_name, to_name=to_name,
                frames_count=frames_count, duration=duration, fps=fps,
                score_map=score_map,
            )

        else:
            # Tracker mode: requires drawn bboxes in context
            logger.info('Extracting prompts from annotation context...')
            prompts = self.get_prompts_native(context, width, height)
            all_obj_ids = set(p['obj_id'] for p in prompts)
            obj_ids = {obj_id: i for i, obj_id in enumerate(all_obj_ids)}
            reverse_obj_ids = {v: k for k, v in obj_ids.items()}

            first_frame_idx = min(p['frame_idx'] for p in prompts) if prompts else 0
            last_frame_idx = max(p['frame_idx'] for p in prompts) if prompts else 0

            logger.info(
                f'Found {len(prompts)} prompt(s) for {len(obj_ids)} object(s), '
                f'keyframes range: [{first_frame_idx}, {last_frame_idx}]')

            # Temporal downsampling
            if TRACK_FPS > 0 and fps > 0:
                stride = max(1, round(fps / TRACK_FPS))
                logger.info(f'Temporal downsampling: original FPS={fps:.2f}, '
                           f'TRACK_FPS={TRACK_FPS:.2f}, stride={stride}')
            else:
                stride = 1

            # Frames to track
            if MAX_FRAMES_TO_TRACK > 0:
                frames_to_track = min(MAX_FRAMES_TO_TRACK, frames_count - first_frame_idx)
                logger.info(f'Tracking limited to {frames_to_track} frames '
                           f'(MAX_FRAMES_TO_TRACK={MAX_FRAMES_TO_TRACK})')
            else:
                frames_to_track = frames_count - first_frame_idx
                logger.info(f'Tracking full video: {frames_to_track} frames')

            frames_to_track_sam3 = (frames_to_track + stride - 1) // stride
            logger.info(f'SAM3 will process ~{frames_to_track_sam3} frames after downsampling')

            if PROCESSING_MODE == 'chunked_batch':
                sequences_by_obj, score_map = self._predict_tracker_chunked(
                    video_path, prompts, obj_ids,
                    first_frame_idx, frames_to_track, stride,
                    width, height, fps,
                )
            else:
                sequences_by_obj, score_map = self._predict_tracker_streaming(
                    video_path, prompts, obj_ids,
                    first_frame_idx, frames_to_track, stride,
                    width, height, fps,
                )

            regions = self._build_regions(
                sequences_by_obj, reverse_obj_ids=reverse_obj_ids,
                context=context,
                from_name=from_name, to_name=to_name,
                frames_count=frames_count, duration=duration, fps=fps,
                score_map=score_map,
            )

        # ---------------------------------------------------------------
        # Build final ModelResponse
        # ---------------------------------------------------------------
        avg_score = 1.0
        if regions:
            scores = [r.get('score', 1.0) for r in regions]
            avg_score = sum(scores) / len(scores)

        prediction = {
            "result": regions,
            "score": avg_score,
            "model_version": self.model_version,
        }
        logger.debug(f'Prediction with {len(regions)} regions')

        logger.info('=' * 80)
        logger.info(f'SAM3 TRACKING COMPLETE!')
        logger.info(f'Summary:')
        logger.info(f'   Objects tracked: {len(regions)}')
        for region in regions:
            obj_id = region['id']
            total_frames = len(region['value']['sequence'])
            logger.info(f'   Object {obj_id}: {total_frames} frames')
        logger.info('=' * 80)

        return ModelResponse(predictions=[prediction])
