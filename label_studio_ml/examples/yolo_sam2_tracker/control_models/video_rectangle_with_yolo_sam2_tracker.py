import base64
import glob
import logging
import os
import pathlib
import sys
import tempfile
from collections import defaultdict
# YOLO + SAM2 related imports
from dataclasses import dataclass, field
from typing import List, Dict, Set
from typing import Literal, cast
from typing import Optional

import av
import cv2
import numpy as np
import torch
from control_models.video_rectangle import VideoRectangleModel
from pycocotools import mask as coco_mask
from ultralytics import YOLO

from label_studio_sdk.label_interface.control_tags import ControlTag

from label_studio_ml.examples.yolo.control_models.base import get_bool
from label_studio_ml.response import ModelResponse

# read the environment variables and set the paths just before importing the sam2 module
SEGMENT_ANYTHING_2_REPO_PATH = os.getenv('SEGMENT_ANYTHING_2_REPO_PATH', 'sam2')
sys.path.append(SEGMENT_ANYTHING_2_REPO_PATH)
from sam2.build_sam import build_sam2_video_predictor


# Global cache for YOLO models
_model_cache = {}
logger = logging.getLogger(__name__)

DEVICE = os.getenv('DEVICE', 'cuda')
SAM2_MODEL_CONFIG = os.getenv('MODEL_CONFIG', './configs/sam2.1/sam2.1_hiera_l.yaml')
SAM2_MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'sam2.1_hiera_large.pt')
PROMPT_TYPE = cast(Literal["box", "point"], os.getenv('PROMPT_TYPE', 'box'))
ANNOTATION_WORKAROUND = os.getenv('ANNOTATION_WORKAROUND', False)
DEBUG = os.getenv('DEBUG', False)
LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', '')
SAM2YOLOBOX_THRESHOLD = float(os.getenv('SAM2YOLOBOX_THRESHOLD', 0.6))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
VIDEO_FRAME_RATE = int(os.getenv('VIDEO_FRAME_RATE', 24))

# Set the log level
logging.basicConfig(level=LOG_LEVEL)

if DEVICE == 'cuda':
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# build path to the model checkpoint
SAM2_MODEL_CHECKPOINT_PATH = str(pathlib.Path(__file__).parent / SEGMENT_ANYTHING_2_REPO_PATH / "checkpoints" / SAM2_MODEL_CHECKPOINT)
logger.info(f'Model checkpoint: {SAM2_MODEL_CHECKPOINT}')
logger.info(f'Model config: {SAM2_MODEL_CONFIG}')
SAM2_PREDICTOR = build_sam2_video_predictor(SAM2_MODEL_CONFIG, SAM2_MODEL_CHECKPOINT_PATH)


# manage cache for inference state
# TODO: make it process-safe and implement cache invalidation
_predictor_state_key = ''
_inference_state = None

def get_inference_state(video_dir):
    """
    Get the inference state for the video directory. If the video directory is different from the previous one,
    :param video_dir:
    :return:
    """
    global _predictor_state_key, _inference_state
    if _predictor_state_key != video_dir:
        _predictor_state_key = video_dir
        _inference_state = SAM2_PREDICTOR.init_state(video_path=video_dir)
    return _inference_state


class ImageFolderSource:
    def __init__(self, folder_path: str, sorting_rule: Optional[callable] = None):
        self.folder_path = folder_path
        # Supported image extensions
        image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        if not self.image_paths:
            raise IOError(f"No images found in folder: {folder_path}")
        # Apply sorting rule
        # self.image_paths.sort(key=sorting_rule if sorting_rule else lambda x: x)

        self.image_paths = sorted(self.image_paths, key=sorting_rule)

        self.frame_count = len(self.image_paths)

    def get_frame(self, frame_index: int):
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError("Frame index out of range")
        image_path = self.image_paths[frame_index]
        frame = cv2.imread(image_path)
        if frame is None:
            raise IOError(f"Failed to read image at path: {image_path}")
        return frame

    def get_frame_count(self) -> int:
        return self.frame_count

    def release(self):
        pass  # No resource to release for image folders

    def get_image_paths(self, start_index: int, end_index: int) -> List[str]:
        """
        Returns a list of image paths between start_index (inclusive) and end_index (exclusive).
        """
        if start_index < 0 or end_index > self.frame_count or start_index >= end_index:
            raise ValueError("Invalid index range")
        return self.image_paths[start_index:end_index]


@dataclass
class Mask:
    """
    Represents a mask with an encoded string and shape.
    """

    encoded: str  # Encoded string as per pycocotools
    shape: List[int]  # [height, width]

    def to_json(self):
        json_encoded = self.encoded.copy()
        json_encoded["counts"] = base64.b64encode(self.encoded["counts"]).decode(
            "utf-8"
        )

        return {"encoded": json_encoded, "shape": self.shape}

    @classmethod
    def from_json(cls, data):
        json_encoded = data["encoded"].copy()
        json_encoded["counts"] = base64.b64decode(json_encoded["counts"])

        return cls(encoded=json_encoded, shape=data["shape"])


@dataclass
class ObjectDetection:
    """
    Stores the bounding box and class of an object.
    """

    detection_id: int
    # Normalization is based on the image resolution (width, height)
    xyxyn: List[
        float
    ]  # Normalized bounding box coordinates [x_min, y_min, x_max, y_max]
    object_class: str  # Class of the object, e.g., 'human' or 'vehicle'

    def to_json(self):
        return {
            "detection_id": self.detection_id,
            "xyxyn": self.xyxyn,
            "object_class": self.object_class,
        }

    @classmethod
    def from_json(cls, data):
        return cls(
            detection_id=data["detection_id"],
            xyxyn=data["xyxyn"],
            object_class=data["object_class"],
        )


@dataclass
class ObjectTracking:
    """
    Stores the object tracking information.
    """

    tracking_id: int
    start_frame: int
    duration_frames: int
    masks: Dict[int, Mask] = field(default_factory=dict)  # Frame ID to Mask
    original_detection_id: Dict[int, Optional[int]] = field(
        default_factory=dict
    )  # Frame ID to detection ID

    def to_json(self):
        return {
            "tracking_id": self.tracking_id,
            "start_frame": self.start_frame,
            "duration_frames": self.duration_frames,
            "masks": {
                str(frame_id): mask.to_json() for frame_id, mask in self.masks.items()
            },
            "original_detection_id": {
                str(frame_id): det_id
                for frame_id, det_id in self.original_detection_id.items()
            },
        }

    @classmethod
    def from_json(cls, data):
        return cls(
            tracking_id=data["tracking_id"],
            start_frame=data["start_frame"],
            duration_frames=data["duration_frames"],
            masks={
                int(frame_id): Mask.from_json(mask_data)
                for frame_id, mask_data in data.get("masks", {}).items()
            },
            original_detection_id={
                int(frame_id): det_id
                for frame_id, det_id in data.get("original_detection_id", {}).items()
            },
        )


@dataclass
class SingleVideoAnnotatorState:
    """
    State of the video annotator focusing on a single video.
    """

    frame_object_detections: Dict[int, List[ObjectDetection]] = field(
        default_factory=dict
    )  # Frame ID to list of detections
    object_trackings: Dict[int, ObjectTracking] = field(
        default_factory=dict
    )  # Tracking ID to ObjectTracking

    num_assigned_detections: int = 0  # Number of assigned detections
    num_assigned_trackings: int = 0  # Number of assigned trackings

    def to_json(self):
        return {
            "frame_object_detections": {
                str(frame_id): [detection.to_json() for detection in detections]
                for frame_id, detections in self.frame_object_detections.items()
            },
            "object_trackings": {
                str(tracking_id): tracking.to_json()
                for tracking_id, tracking in self.object_trackings.items()
            },
            "num_assigned_detections": self.num_assigned_detections,
            "num_assigned_trackings": self.num_assigned_trackings,
        }

    @classmethod
    def from_json(cls, data):
        frame_object_detections = {
            int(frame_id): [
                ObjectDetection.from_json(det_data) for det_data in detections
            ]
            for frame_id, detections in data.get("frame_object_detections", {}).items()
        }
        object_trackings = {
            int(tracking_id): ObjectTracking.from_json(tracking_data)
            for tracking_id, tracking_data in data.get("object_trackings", {}).items()
        }
        num_assigned_detections = data.get("num_assigned_detections", 0)
        num_assigned_trackings = data.get("num_assigned_trackings", 0)
        return cls(
            frame_object_detections=frame_object_detections,
            object_trackings=object_trackings,
            num_assigned_detections=num_assigned_detections,
            num_assigned_trackings=num_assigned_trackings,
        )


class SingleVideoAnnotatorModel:
    """
    Model of the video annotator application responsible for storing:
        - Detections
        - Single-video object tracking (without re-identification)
    The model notifies observers when its state changes and exposes an interface for the controller.
    """

    def __init__(
            self,
            object_classes: Set[str] = {"person"}
    ):
        """
        Initializes the annotator model for a single video.

        Args:
            video_id (str): Identifier for the video being annotated.
            video_source_path (str): Path to the video file or image folder.
            sorting_rule (callable, optional): Sorting function for image filenames if video_source_path is an image folder.
        """
        self.video_source = None
        self.object_classes = object_classes

        self.yolo_model = None  # Load an official Detect model

        self.state = SingleVideoAnnotatorState()

        # H, W, 3
        self.image_shape = self.get_frame(0).shape[:2] if self.video_source else None


        # SAM2 related attributes
        self.sam2_model_cfg = None
        self.sam2_model_checkpoint_path = None
        self.sam2_max_frames_to_track = None
        self.prompt_type = None
        self.annotation_workaround = None
        self.sam2_predictor = None

        self._predictor_state_key = ''
        self._inference_state = None

    @classmethod
    def load_yolo_model(cls, checkpoint) -> YOLO:
        """Load YOLO model from the file."""
        logger.info(f"Loading yolo model: {checkpoint}")
        model = YOLO(checkpoint)
        logger.info(f"Model {checkpoint} names:\n{model.names}")
        return model

    @classmethod
    def get_cached_model(cls, path: str) -> YOLO:
        if path not in _model_cache:
            _model_cache[path] = cls.load_yolo_model(path)
        return _model_cache[path]

    @staticmethod
    def get_video_fps_duration(path, fps):
        if not os.path.exists(path):
            raise ValueError(f"Video file not found: {path}")
        container = av.open(path)
        duration = container.duration / av.time_base  # Duration in seconds
        frame_count = int(duration * fps)
        logger.info(f"Video duration: {duration} seconds, {frame_count} frames, {fps} fps")
        return frame_count, duration

    def get_inference_state(self, video_dir):
        """
        Get the inference state for the video directory. If the video directory is different from the previous one,
        :param video_dir:
        :return:
        """
        if self._predictor_state_key != video_dir:
            self._predictor_state_key = video_dir
            self._inference_state = SAM2_PREDICTOR.init_state(video_path=video_dir)
        return self._inference_state

    def build_sam2_predictor(self, sam2_model_checkpoint_path, sam2_model_cfg, sam2_max_frames_to_track, prompt_type, annotation_workaround):
        """
        Update the SAM2 configuration.
        :param sam2_model_checkpoint:
        :param sam2_model_cfg:
        :param sam2_max_frames_to_track:
        :param prompt_type:
        :param annotation_workaround:
        :return:
        """

        self.sam2_model_cfg = sam2_model_cfg
        self.sam2_model_checkpoint_path = sam2_model_checkpoint_path
        self.sam2_max_frames_to_track = sam2_max_frames_to_track
        self.prompt_type = prompt_type
        self.annotation_workaround = annotation_workaround

        self.sam2_predictor = build_sam2_video_predictor(
            self.sam2_model_cfg, self.sam2_model_checkpoint_path, device="cuda:0"
        )

        return self.sam2_predictor

    def convert_mask_to_bbox(self, mask: Mask):
        """
        Function to convert a mask to a bounding box.
        Used from Label Studio ML examples.
        :param mask:
        :return:
        """
        # Decode the mask
        mask_np = coco_mask.decode(mask.encoded)

        # squeeze
        logger.debug(f"Mask shape: {mask_np.shape}")

        y_indices, x_indices = np.where(mask_np == 1)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        # Find the min and max indices
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)

        # Get mask dimensions
        height, width = mask_np.shape

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

    def split_frames(self, video_path, temp_dir, video_fps, start_frame=0, end_frame=100):
        """
        Extracts and saves frames from a video file within the specified range.

        This method processes a video file, extracts frames between the provided
        start and end frame indexes, and stores the extracted images in the specified
        temporary directory. Each frame is saved as a `.jpg` file, and the method
        yields the file path and the corresponding frame data for further processing.

        :param video_path: Path to the input video file.
        :param temp_dir: Directory to store the extracted frame images temporarily.
        :param start_frame: Index of the first frame to extract (inclusive).
        :param end_frame: Index of the last frame to extract (exclusive).
        :return: Yields a tuple containing the file path and raw frame data for each extracted frame.
        """
        logger.debug(f'Opening video file: {video_path}')

        # Get video properties using the static method
        frame_count, duration = self.get_video_fps_duration(video_path, fps=video_fps)

        # Open the video using OpenCV
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        logger.debug(f'Video duration: {duration} seconds, {frame_count} frames, {video_fps} fps')

        frame_count_current = 0
        while frame_count_current < frame_count:
            # Calculate the timestamp in seconds for the current frame
            timestamp = frame_count_current / video_fps

            # Set the video capture position to the corresponding timestamp in milliseconds
            video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            # Read the frame at the calculated timestamp
            success, frame = video.read()

            if not success:
                logger.error(f'Failed to read frame {frame_count_current}, this could be due to an empty frame.')
                break

            if frame_count_current < start_frame:
                frame_count_current += 1
                continue

            if frame_count_current >= end_frame:
                break

            frame_filename = os.path.join(temp_dir, f'{frame_count_current:05d}.jpg')

            if not os.path.exists(frame_filename):
                cv2.imwrite(frame_filename, frame)

            logger.debug(f'Frame {frame_count_current}: {frame_filename}')
            yield frame_filename, frame
            frame_count_current += 1

        video.release()

    def get_net_predictions_as_regions(self, video_path: str, video_fps: int, context: Optional[Dict] = None):
        """
        Extract predictions from the ObjectTracking object and create video rectangle regions.
        """
        logger.debug(f"Getting net predictions for video: {video_path}")
        sequences = []
        start_frame = 0

        # Get video metadata
        frame_count, duration = self.get_video_fps_duration(video_path, fps=video_fps)

        # Dictionary to store tracking sequences and labels
        tracks = defaultdict(list)
        track_labels = dict()

        # Iterate over tracking data
        for tracking_id, tracking in self.state.object_trackings.items():
            label_map = context.get('label_map', {}) if context else {}
            track_labels[tracking_id] = label_map.get(tracking_id, 'Unknown')
            for frame_id in range(tracking.start_frame, tracking.start_frame + tracking.duration_frames + 1):
                if frame_id in tracking.masks:
                    mask: Mask = tracking.masks[frame_id]
                    bbox = self.convert_mask_to_bbox(mask)
                    if bbox:
                        box = {
                            'frame': frame_id + 1,
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height'],
                            'enabled': True,
                            'rotation': 0,
                            'time': frame_id / video_fps,
                        }
                        tracks[tracking_id].append(box)

        # Process tracks to create regions
        regions = []
        for track_id, sequence in tracks.items():
            label = track_labels[track_id]
            max_score = max([frame_info.get("score", 1.0) for frame_info in sequence])  # Use 1.0 as a default score

            region = {
                "from_name": "box",
                "to_name": "video",
                "type": "videorectangle",
                "value": {
                    "framesCount": frame_count,
                    "duration": duration,
                    "sequence": sequence,
                    "labels": [label],
                },
                "score": max_score,
                "origin": "manual",
            }
            regions.append(region)

        return regions


    # state modification methods
    def add_detection(self, frame_id: int, detection: ObjectDetection):
        """
        Adds a detection to a specific frame.
        """
        detections = self.state.frame_object_detections.setdefault(frame_id, [])
        detections.append(detection)
        # self.notify_observers(
        #     frame_id=frame_id,
        #     changed="detections"
        # )

    def clear_detections(self, frame_id: int):
        """
        Clears all detections from a specific frame.
        """
        self.state.frame_object_detections.pop(frame_id, None)
        # self.notify_observers(
        #     frame_id=frame_id,
        #     changed="detections"
        # )

    def append_mask_to_tracking(self, tracking_id: int, frame_id: int, mask: Mask):
        """
        Appends a mask to an existing tracking.
        """
        if tracking_id not in self.state.object_trackings:
            self.add_tracking(
                ObjectTracking(
                    tracking_id=tracking_id, start_frame=frame_id, duration_frames=0
                )
            )

        tracking = self.state.object_trackings[tracking_id]
        tracking.masks[frame_id] = mask
        tracking.duration_frames = max(
            tracking.duration_frames, frame_id - tracking.start_frame
        )

        # self.notify_observers(
        #     frame_id=frame_id,
        #     tracking_id=tracking_id,
        #     changed="tracking"
        # )

    def add_tracking(self, tracking: ObjectTracking):
        """
        Adds a tracking object.
        """
        self.state.object_trackings[tracking.tracking_id] = tracking

    def get_YOLO_detections(self,
                            conf,
                            iou,
                            yolo_model_checkpoint,
                            video_source_path,
                            video_fps,
                            max_frames_to_track,
                            output_frames_dir="predictions/yolo"):
        """
        Run YOLO detection on the entire video, overlay the detections on each frame, and save each frame as an image.
        """
        if DEVICE == 'cuda':
            self.yolo_model = self.get_cached_model(yolo_model_checkpoint)
        else:
            logger.error("Only CUDA is supported for YOLO model")
            return

        # Open video source
        cap = cv2.VideoCapture(video_source_path)
        frame_count, duration = self.get_video_fps_duration(video_source_path, fps=video_fps)

        # Get the frame width and height from the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video dimensions: width={width}, height={height}")

        # Ensure output directory exists
        if not os.path.exists(output_frames_dir):
            os.makedirs(output_frames_dir)

        logger.info(f"Reading frames from {video_source_path} at {video_fps} FPS...")

        frame_id = 0
        frames_to_track = min(frame_count, max_frames_to_track)

        while cap.isOpened() and frame_id < frames_to_track:
            # Read from a specific timestamp based on the fps and frame_id
            timestamp = frame_id / video_fps
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            logger.debug(f"Detection frame {frame_id}")
            if not ret:
                break  # End of video

            # Run YOLO detection on the current frame,
            # Image size must be multiple of max stride 32
            img_size = min(width, height) - (min(width, height) % 32)
            detection_result = self.yolo_model.track(
                frame,
                conf=conf,
                iou=iou,
                imgsz=img_size,
                persist=True,
                show=False,
                verbose=False,
            )[0]

            detected_boxes = detection_result.boxes
            det_class_mapping = detection_result.names

            self.clear_detections(frame_id)

            for detection in detected_boxes:
                new_detection_id = self.assign_new_detection_id()
                detected_class = det_class_mapping[int(detection.cls[0].item())]

                # Check if the detected class is in the object classes
                if detected_class not in self.object_classes:
                    continue

                logger.debug(f"Detected class: {detected_class}, at {detection.xyxyn[0]} for frame {frame_id}")

                # Extract bounding box coordinates (normalize to [0, 1] range, multiply by image size)
                x1, y1, x2, y2 = detection.xyxyn[0]  # normalized coordinates
                x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
                cv2.putText(frame, detected_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Add detection to internal structure
                self.add_detection(
                    frame_id,
                    ObjectDetection(
                        detection_id=new_detection_id,
                        xyxyn=detection.xyxyn[0].tolist(),
                        object_class=detected_class,
                    ),
                )

            # Save the frame with the overlay as an image
            frame_filename = os.path.join(output_frames_dir, f"frame_{frame_id:04d}.jpg")

            try:
                # Attempt to write the frame to a file
                if not cv2.imwrite(frame_filename, frame):
                    logger.error(f"Failed to write frame {frame_id} to {frame_filename}")
                else:
                    logger.debug(f"Successfully saved frame {frame_id} to {frame_filename}")
            except Exception as e:
                logger.error(f"Error writing frame {frame_id} to {frame_filename}: {e}")

            frame_id += 1

        # Release resources
        cap.release()
        logger.info(f"Frames with detections saved to {output_frames_dir}")

    def get_sam_tracking_with_yolo_prompts(
            self, video_fps: int, frames_to_track: int, sam_batchsize: int = 100, video_source_path: str = None
    ):
        """
        Processes a video using SAM tracking with YOLO prompts.

        Args:
            video_fps (int): Frames per second for the video.
            frames_to_track (int): Total number of frames to process.
            sam_batchsize (int): Batch size for SAM processing.
            video_source_path (str): Path to the source video.
        """
        current_frame = 0
        # -1 is done to avoid out of range error as propagation only happens N -1 frames
        sam_end_frame = min(self.get_video_fps_duration(video_source_path, fps=video_fps)[0], frames_to_track) - 1

        logger.info(f"Processing video with SAM tracking for {sam_end_frame} frames...")

        with tempfile.TemporaryDirectory() as temp_img_dir:
            # temp_img_dir = '/tmp/frames'  # Use persisted directory for debugging
            # os.makedirs(temp_img_dir, exist_ok=True)

            frames = list(
                self.split_frames(
                    video_path=video_source_path, temp_dir=temp_img_dir, video_fps=video_fps, start_frame=0, end_frame=sam_end_frame + 1
                )
            )

            self.video_source = ImageFolderSource(temp_img_dir, sorting_rule=lambda x: x)
            self.image_shape = self.get_frame(0).shape[:2]
            logger.info(f"Video dimensions: width={self.image_shape[1]}, height={self.image_shape[0]}")

            with torch.autocast("cuda", torch.bfloat16):
                while current_frame < sam_end_frame - 1:
                logger.debug(f"Processing frames using SAM Tracker...")
                while current_frame < sam_end_frame:
                    logger.debug(f"Processing frame {current_frame}")

                    sam_prompts = self.prepare_sam_prompts(current_frame)
                    if not sam_prompts:
                        logger.debug(f"No SAM prompts for frame {current_frame}. Skipping.")
                        current_frame += 1
                        continue

                    logger.debug(f"Initializing inference state for frame {current_frame}")
                    inference_state = self.get_inference_state(video_dir=temp_img_dir)
                    self.sam2_predictor.reset_state(inference_state)

                    logger.debug(f"Adding {len(sam_prompts)} prompts to SAM predictor")
                    for tracking_id, mask_bbox in sam_prompts:
                        _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=current_frame,
                            obj_id=tracking_id,
                            box=mask_bbox,
                        )

                    for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(inference_state):
                        for i, out_obj_id in enumerate(out_obj_ids):
                            mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]
                            if mask.any():
                                mask_obj = Mask(
                                    encoded=coco_mask.encode(np.asfortranarray(mask)),
                                    shape=mask.shape,
                                )
                                self.append_mask_to_tracking(out_obj_id, out_frame_idx, mask_obj)

                        # unexplained_detections = self.get_unexplained_detections_at_frame(out_frame_idx)
                        # logger.debug(f"Unexplained detections at frame {out_frame_idx}: {len(unexplained_detections)}")
                        #
                        # if unexplained_detections:
                        #     logger.warning(f"Stopping propagation at frame {out_frame_idx} due to unexplained detections.")
                        #     break

                    current_frame = out_frame_idx
                    logger.info(f"Updated current frame to {current_frame}")

    def prepare_sam_prompts(self, frame_id: int):
        """
        Prepares SAM prompts for a frame.

        Args:
            frame_id (int): Frame ID for which to prepare the prompts.

        Returns:
            List[str]: List of prompts for SAM.
        """
        # For all object tracking, get the masks for the frame. These tracking ids # will be kept in the SAM prompts.
        annotated_masks = self.get_annotated_masks_at_frame(frame_id)
        unexplained_detections = self.get_unexplained_detections_at_frame(frame_id)

        # prepare the SAM prompts based on existing masks and unexplained detections. The prompts will be in the form of bounding boxes.
        sam_prompts = []

        for tracking_id, mask in annotated_masks.items():
            mask_bbox = coco_mask.toBbox(mask.encoded)

            # NOTE: this bbox is in x, y, w, h format
            mask_bbox_xyxy = np.array(
                [
                    mask_bbox[0],
                    mask_bbox[1],
                    mask_bbox[0] + mask_bbox[2],
                    mask_bbox[1] + mask_bbox[3],
                    ]
            )

            sam_prompts.append((tracking_id, mask_bbox_xyxy))

        for detection in unexplained_detections:
            detection_bbox = detection.xyxyn * np.array(
                [
                    self.image_shape[1],
                    self.image_shape[0],
                    self.image_shape[1],
                    self.image_shape[0],
                ]
            )
            sam_prompts.append((self.assign_new_tracking_id(), detection_bbox))

        return sam_prompts

    def get_annotated_masks_at_frame(self, frame_id: int):
        annotated_masks = {}  # tracking_id to mask
        for tracking_id, tracking in self.state.object_trackings.items():
            if frame_id in tracking.masks:
                mask = tracking.masks[frame_id]
                # Add the mask to the SAM prompts
                annotated_masks[tracking_id] = mask

        return annotated_masks

    def get_unexplained_detections_at_frame(self, frame_id: int):
        """
        This function returns all the bboxes that are not in the SAM propagation.
        :param frame_id:
        :return:
        """

        annotated_masks = self.get_annotated_masks_at_frame(frame_id)

        # Get the detections for the frame, try to explain them with the masks. If not possible, add the detection to the SAM prompts with a new tracking id.
        all_detections = self.state.frame_object_detections.get(frame_id, [])

        # explain the detections with the masks
        unexplained_detections = []
        for detection in all_detections:
            explained = False
            for tracking_id, mask in annotated_masks.items():
                if self.explain_detection_with_mask(detection, mask):
                    explained = True
                    break
            if not explained:
                unexplained_detections.append(detection)

        return unexplained_detections

    def explain_detection_with_mask(self, detection: ObjectDetection, mask: Mask):
        """
        Explains a detection with a mask.

        Args:
            detection (ObjectDetection): Detection to explain.
            mask (Mask): Mask to explain the detection.

        Returns:
            bool: True if the detection was explained, False otherwise.
        """

        # a detection is explained by a mask if the bounding of the mask
        # and the detection have error less then 0.1 normalized error.

        # get the bounding box of the mask
        decoded_mask = coco_mask.decode(mask.encoded)
        mask_bbox = coco_mask.toBbox(mask.encoded)  # notice! this is x, y, w, h

        detection_xyxy = np.array(
            [
                int(detection.xyxyn[0] * self.image_shape[1]),
                int(detection.xyxyn[1] * self.image_shape[0]),
                int(detection.xyxyn[2] * self.image_shape[1]),
                int(detection.xyxyn[3] * self.image_shape[0]),
            ]
        )

        mask_in_bbox = np.sum(
            decoded_mask[
            detection_xyxy[1] : detection_xyxy[3],
            detection_xyxy[0] : detection_xyxy[2],
            ]
        )
        mask_pixels = np.sum(decoded_mask)

        mask_in_bbox = mask_in_bbox / (mask_pixels + 1e-6)

        return mask_in_bbox > SAM2YOLOBOX_THRESHOLD

    def assign_new_detection_id(self):
        """
        Assigns a new detection ID for a new detection.
        """
        assigned_id = self.state.num_assigned_detections

        self.state.num_assigned_detections += 1


        return assigned_id

    def assign_new_tracking_id(self):
        """
        Assigns a new tracking ID for a new tracking.
        """
        assigned_id = self.state.num_assigned_trackings

        self.state.num_assigned_trackings += 1


        return assigned_id

    def get_frame(self, frame_index: int):
        """
        Retrieves a frame from the video source.

        Args:
            frame_index (int): Index of the frame to retrieve.

        Returns:
            The video frame as an image.
        """
        return self.video_source.get_frame(frame_index)

    def get_frame_count(self) -> int:
        """
        Returns the total number of frames in the video.

        Returns:
            int: Total number of frames.
        """
        return self.video_source.get_frame_count()

    def release_video(self):
        """
        Releases the video source resources.
        """
        self.video_source.release()

    def get_all_tracking(self):
        """
        Returns all the tracking objects.
        """
        return self.state.object_trackings

    def get_regions_from_yolo_sam2_tracker(self,
                                           conf,
                                           iou,
                                           yolo_model_checkpoint,
                                           image_size,
                                           sam2_model_checkpoint_path,
                                           sam2_model_cfg,
                                           sam2_max_frames_to_track,
                                           prompt_type,
                                           annotation_workaround,
                                           video_source_path,
                                           video_fps):
        """
        Run YOLO detection and SAM tracking on the video source.
        """

        ######### YOLO DETECTION #########
        self.get_YOLO_detections(conf=conf,
                                iou=iou,
                                yolo_model_checkpoint=yolo_model_checkpoint,
                                video_source_path=video_source_path,
                                video_fps=video_fps,
                                max_frames_to_track=max_frames_to_track)

        ######### SAM TRACKING #########
        self.build_sam2_predictor(sam2_model_checkpoint_path, sam2_model_cfg, max_frames_to_track, prompt_type, annotation_workaround)
        self.get_sam_tracking_with_yolo_prompts(video_fps=video_fps,
                                                frames_to_track=max_frames_to_track,
                                                video_source_path=video_source_path)

        # return the regions
        return self.get_net_predictions_as_regions(video_path=video_source_path, video_fps=video_fps)


class VideoRectangleWithYOLOSAM2TrackerModel(VideoRectangleModel):
    """
    Class representing a RectangleLabels (bounding boxes) control tag for YOLO model.
    """

    type = "VideoRectangleWithYOLOSAM2Tracker"
    model_path = "yolov10x.pt"

    @classmethod
    def is_control_matched(cls, control: ControlTag) -> bool:
        # check object tag type
        if control.objects[0].tag != "Video":
            return False
        if not get_bool(control.attr, "model_sam_tracker", "false"):
            return False
        return True

    def get_model_configs(self):
        """

        :return:
        """
        conf = float(self.control.attr.get("model_conf", 0.25))
        iou = float(self.control.attr.get("model_iou", 0.70))
        yolo_model = self.control.attr.get("yolo_model", "yolov10x").lower()
        yolo_model_checkpoint = yolo_model + ".pt"
        image_size = int(self.control.attr.get("model_image_size", 2560))
        frames_to_track = int(self.control.attr.get("frames_to_track", 100))
        fps = int(self.control.attr.get("fps", 24))

        return conf, iou, yolo_model_checkpoint, image_size, frames_to_track, fps

    def predict_regions(self, path) -> List[Dict]:
        """
        # track regions with YOLO SAM2 tracker
        :param path:
        :return:
        """
        single_video_annotator = SingleVideoAnnotatorModel()

        conf, iou, yolo_model_checkpoint, image_size, frames_to_track, fps = self.get_model_configs()
        regions = single_video_annotator.get_regions_from_yolo_sam2_tracker(
            conf=conf,
            iou=iou,
            yolo_model_checkpoint=yolo_model_checkpoint,
            image_size=image_size,
            sam2_model_cfg=SAM2_MODEL_CONFIG,
            sam2_model_checkpoint_path=SAM2_MODEL_CHECKPOINT_PATH,
            max_frames_to_track=frames_to_track,
            prompt_type=PROMPT_TYPE,
            annotation_workaround=ANNOTATION_WORKAROUND,
            video_source_path=path,
            video_fps=fps
        )

        return regions

# pre-load and cache default model at startup
VideoRectangleWithYOLOSAM2TrackerModel.get_cached_model(VideoRectangleWithYOLOSAM2TrackerModel.model_path)
