import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from control_models.base import ControlModel
from utils.grounding import VideoTrackingResult
from label_studio_sdk.label_interface.control_tags import ControlTag


logger = logging.getLogger(__name__)

ALLOWED_LABELS = {
    label.strip().lower()
    for label in os.getenv("GROUNDING_DINO_ALLOWED_LABELS", "person").split(",")
    if label.strip()
}

SPARSIFY_KF_FOR_SAM = os.getenv("SPARSIFY_KF_FOR_SAM", "false").lower() in ["1", "true"]
KEYFRAME_SPARSIFY_INTERVAL = max(
    1, int(os.getenv("KEYFRAME_SPARSIFY_INTERVAL", os.getenv("VIDEO_KEYFRAME_SPARSIFY_INTERVAL", "5")))
)

TRACKER_ENV_MAP = {
    "track_activation_threshold": "track_activation_threshold",
    "lost_track_buffer": "lost_track_buffer",
    "minimum_matching_threshold": "minimum_matching_threshold",
    "minimum_consecutive_frames": "minimum_consecutive_frames",
}

MODEL_THRESHOLD_ENV_MAP = {
    "model_box_threshold": "GROUNDING_DINO_BOX_THRESHOLD",
    "model_text_threshold": "GROUNDING_DINO_TEXT_THRESHOLD",
}

TRACKER_PARAM_CASTERS = {
    "lost_track_buffer": int,
    "minimum_consecutive_frames": int,
    "minimum_matching_threshold": float,
    "track_activation_threshold": float,
}


def _get_env(name: str, default: str = None) -> str:
    """Get environment variable, returning default if not set.

    When using tracking presets, env vars are set by apply_preset().
    When not using presets, env vars should be set in docker-compose.yml.
    """
    value = os.getenv(name)
    if value is None or value == "":
        if default is not None:
            return default
        raise RuntimeError(
            f"Environment variable '{name}' is not set. "
            f"Either use --preset or set {name} in docker-compose.yml."
        )
    return value


class VideoRectangleModel(ControlModel):
    """Video rectangle control using Grounding DINO detections with ByteTrack."""

    type = "VideoRectangle"
    last_tracking_result: Optional[VideoTrackingResult] = None

    @classmethod
    def is_control_matched(cls, control: ControlTag) -> bool:
        return control.objects[0].tag == "Video" and control.tag == cls.type

    @staticmethod
    def get_from_name_for_label_map(label_interface, target_name) -> str:
        target: ControlTag = label_interface.get_control(target_name)
        if not target:
            raise ValueError(f'Control tag with name "{target_name}" not found')

        for connected in label_interface.controls:
            if connected.tag == "Labels" and connected.to_name == target.to_name:
                return connected.name

        raise ValueError("VideoRectangle detected, but no connected 'Labels' tag found")

    def predict_regions(self, path, output_dir=None, save_frames=False, max_frames=None) -> List[Dict]:
        tracker_kwargs = self._build_tracker_kwargs()
        box_threshold = self._get_float_attr("model_box_threshold")
        text_threshold = self._get_float_attr("model_text_threshold")

        tracker_scenarios = self._load_tracker_scenarios()

        # Reset cached tracking result before processing the current task
        self.last_tracking_result = None

        tracking = self.inference.track_video(
            path,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            tracker_kwargs=tracker_kwargs,
            output_dir=output_dir,
            save_frames=save_frames,
            max_frames=max_frames,
            tracker_scenarios=tracker_scenarios,
        )
        self.last_tracking_result = tracking
        return self.create_video_rectangles(tracking)

    def create_video_rectangles(self, tracking_result) -> List[Dict]:
        frames_count = tracking_result.frames_count or len(tracking_result.frames)
        fps = tracking_result.fps if tracking_result.fps else 0.0
        frame_interval = (1.0 / fps) if fps else (
            tracking_result.duration / frames_count if frames_count else 0.0
        )

        tracks = defaultdict(list)
        track_labels: Dict[int, str] = {}

        for frame_info in tracking_result.frames:
            detections = frame_info.detections
            if detections.tracker_id is None:
                continue

            for bbox, score, class_id, track_id in zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id,
                detections.tracker_id,
            ):
                if track_id is None:
                    continue
                if score < self.model_score_threshold:
                    continue

                model_label = self.inference.names[int(class_id)]
                if model_label not in self.label_map:
                    continue
                output_label = self.label_map[model_label]
                if output_label.strip().lower() not in ALLOWED_LABELS:
                    continue

                track_labels[int(track_id)] = output_label

                x1, y1, x2, y2 = bbox.tolist()
                width = frame_info.width
                height = frame_info.height
                frame_number = frame_info.frame_index + 1

                box_entry = {
                    "frame": frame_number,
                    "enabled": True,
                    "rotation": 0,
                    "x": x1 / width * 100,
                    "y": y1 / height * 100,
                    "width": (x2 - x1) / width * 100,
                    "height": (y2 - y1) / height * 100,
                    "time": frame_info.frame_index * frame_interval,
                    "score": float(score),
                }
                tracks[int(track_id)].append(box_entry)

        regions: List[Dict] = []
        for track_id, sequence in tracks.items():
            sequence.sort(key=lambda item: item["frame"])
            sequence = self._sparsify_dense_keyframes(sequence)
            if SPARSIFY_KF_FOR_SAM:
                sequence = self._sparsify_sequence_for_sam(sequence)
            sequence = self.process_lifespans_enabled(sequence)

            label = track_labels.get(track_id)
            if not label:
                continue

            if len(sequence) < 2:
                logger.debug(
                    "Skipping track %s because it only has %d keyframe(s) after sparsification",
                    track_id,
                    len(sequence),
                )
                continue

            region = {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "videorectangle",
                "value": {
                    "framesCount": frames_count,
                    "duration": tracking_result.duration,
                    "sequence": sequence,
                    "labels": [label],
                },
                "score": max(frame_info["score"] for frame_info in sequence),
                "origin": "manual",
            }
            region.setdefault("meta", {})["text"] = f"id:{track_id}"
            regions.append(region)

        return regions

    @staticmethod
    def process_lifespans_enabled(sequence: List[Dict]) -> List[Dict]:
        prev = None
        for i, box in enumerate(sequence):
            if prev is None:
                prev = sequence[i]
                continue
            if box["frame"] - prev["frame"] > 1:
                sequence[i - 1]["enabled"] = False
            prev = sequence[i]

        if sequence:
            sequence[-1]["enabled"] = False
        return sequence

    @staticmethod
    def _sparsify_sequence_for_sam(sequence: List[Dict]) -> List[Dict]:
        if not sequence:
            return sequence

        chunk_size = 2000
        max_kf_per_chunk = 5

        by_chunk: Dict[int, List[Dict]] = {}
        for box in sequence:
            frame = box.get("frame")
            if frame is None:
                continue
            chunk_index = (frame - 1) // chunk_size
            if chunk_index not in by_chunk:
                by_chunk[chunk_index] = []
            by_chunk[chunk_index].append(box)

        result: List[Dict] = []
        for chunk_index in sorted(by_chunk.keys()):
            chunk_boxes = by_chunk[chunk_index]
            count = len(chunk_boxes)
            if count <= max_kf_per_chunk:
                result.extend(chunk_boxes)
                continue

            desired = max_kf_per_chunk
            if desired <= 0:
                continue

            indices = set()
            if desired == 1:
                indices.add(count // 2)
            else:
                for i in range(desired):
                    pos = int(round(i * (count - 1) / (desired - 1)))
                    indices.add(pos)

            for idx, box in enumerate(chunk_boxes):
                if idx in indices:
                    result.append(box)

        result.sort(key=lambda item: item["frame"])
        return result

    @staticmethod
    def _sparsify_dense_keyframes(sequence: List[Dict]) -> List[Dict]:
        if KEYFRAME_SPARSIFY_INTERVAL <= 1 or not sequence:
            return sequence

        thinned: List[Dict] = []
        prev_frame: Optional[int] = None
        streak_index = 0
        last_kept_idx = -1

        for idx, box in enumerate(sequence):
            frame = box.get("frame")
            if not isinstance(frame, int):
                thinned.append(box)
                prev_frame = None
                streak_index = 0
                last_kept_idx = len(thinned) - 1
                continue

            if prev_frame is None or frame - prev_frame > 1:
                streak_index = 0
            else:
                streak_index += 1

            if streak_index % KEYFRAME_SPARSIFY_INTERVAL == 0:
                thinned.append(box)
                last_kept_idx = len(thinned) - 1

            prev_frame = frame

        # Ensure the final keyframe is kept for track completeness.
        if thinned and thinned[-1] is not sequence[-1]:
            thinned.append(sequence[-1])
        elif not thinned:
            thinned.append(sequence[-1])

        return thinned

    def _build_tracker_kwargs(self) -> Dict:
        kwargs: Dict = {}

        logger.info("Loading ByteTrack parameters from environment (exact names)")
        for param, env_name in TRACKER_ENV_MAP.items():
            raw_value = os.getenv(env_name)
            if raw_value is None or raw_value == "":
                logger.info(
                    "Tracker parameter '%s' not set; using supervision default",
                    env_name,
                )
                continue

            caster = TRACKER_PARAM_CASTERS.get(param, float)
            try:
                parsed = caster(float(raw_value)) if caster is int else caster(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid value for tracker parameter '{env_name}': {raw_value}"
                ) from exc
            kwargs[param] = parsed
            logger.info("Tracker parameter %s=%s", param, parsed)

        control_threshold = self.control.attr.get("tracker_match_threshold")
        if control_threshold:
            try:
                kwargs.setdefault("minimum_matching_threshold", float(control_threshold))
            except ValueError:
                logger.warning(
                    "Invalid tracker_match_threshold value '%s'", control_threshold
                )

        return kwargs

    @staticmethod
    def _load_tracker_scenarios() -> Optional[List[Dict]]:
        raw = os.getenv("TRACKER_SCENARIOS_JSON")
        if not raw:
            return None

        try:
            scenarios = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid TRACKER_SCENARIOS_JSON: %s", exc)
            return None

        if not isinstance(scenarios, list):
            logger.warning("TRACKER_SCENARIOS_JSON must be a list of objects")
            return None

        logger.info("Loaded %d tracker comparison scenarios", len(scenarios))
        return scenarios

    def _get_float_attr(self, key: str) -> Optional[float]:
        env_name = MODEL_THRESHOLD_ENV_MAP.get(key)
        if env_name:
            env_value = _get_env(env_name)
            try:
                return float(env_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid value for environment variable '{env_name}': {env_value}"
                ) from exc

        value = self.control.attr.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            logger.warning("Invalid attribute %s='%s'", key, value)
            return None
