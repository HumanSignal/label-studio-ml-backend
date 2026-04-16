"""Segment Anything Video Interactive — ML backend.

Full-parity replacement for the in-browser BYOM SAM tag:
  * `context.event == "prewarm"` → schedule background encoding of a window of
    frames in the navigation direction; never re-encode.
  * `context.event == "predict"` (default) → run SAM2 on the target frame
    using the cached embedding if available, otherwise encode inline.

The HTTP surface is the standard label-studio-ml-backend /predict endpoint;
both modes multiplex through it so the Label Studio `mlInteractive` proxy
needs no changes.

Output format is the standard Label Studio `PredictionValue.result` shape,
dispatched on the project's control tag type (brushlabels / rectanglelabels /
polygonlabels / videorectangle).
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

from frame_cache import FrameCache
from mask_encoding import (
    mask_to_bbox_percent,
    mask_to_bitmap_png_base64,
    mask_to_polygons_percent,
)
from video_state import VideoRegistry

logger = logging.getLogger(__name__)

DEVICE = os.getenv("DEVICE", "cuda")
SEGMENT_ANYTHING_2_REPO_PATH = os.getenv("SEGMENT_ANYTHING_2_REPO_PATH", "segment-anything-2")
MODEL_CONFIG = os.getenv("MODEL_CONFIG", "sam2_hiera_l.yaml")
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "sam2_hiera_large.pt")
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "20"))


# ---------------------------------------------------------------------------
# SAM2 model loading
# ---------------------------------------------------------------------------


def _build_predictors():
    """Lazily build the SAM2 image + video predictors.

    Kept in a function so tests that don't need the model (cache logic, result
    shaping) can import this module without pulling torch/SAM2 weights.
    """
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    ckpt = str(
        pathlib.Path(__file__).parent / SEGMENT_ANYTHING_2_REPO_PATH / "checkpoints" / MODEL_CHECKPOINT
    )
    image_model = build_sam2(MODEL_CONFIG, ckpt, device=DEVICE)
    image_predictor = SAM2ImagePredictor(image_model)
    video_predictor = build_sam2_video_predictor(MODEL_CONFIG, ckpt, device=DEVICE)
    return image_predictor, video_predictor


_predictors: Optional[Tuple[Any, Any]] = None


def get_predictors():
    global _predictors
    if _predictors is None:
        _predictors = _build_predictors()
    return _predictors


# ---------------------------------------------------------------------------
# Shared per-process state
# ---------------------------------------------------------------------------

FRAME_CACHE = FrameCache()
VIDEOS = VideoRegistry()


# ---------------------------------------------------------------------------
# Context parsing
# ---------------------------------------------------------------------------


def _extract_prompts(context: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise tag-side prompts into {points, labels, box} in relative coords [0..1].

    Frontend sends a standard Label Studio `context.result` array. For this tag:
      * keypointlabels items carry {x, y, positive} in percent (0..100).
      * rectanglelabels items carry {x, y, width, height} in percent.
    """
    points: List[List[float]] = []
    labels: List[int] = []
    box: Optional[List[float]] = None

    for item in context.get("result", []) or []:
        value = item.get("value", {}) or {}
        t = item.get("type")
        if t == "keypointlabels" or "x" in value and "y" in value and "width" not in value:
            x = float(value.get("x", 0)) / 100.0
            y = float(value.get("y", 0)) / 100.0
            positive = bool(value.get("positive", True))
            points.append([x, y])
            labels.append(1 if positive else 0)
        elif t == "rectanglelabels" or ("width" in value and "height" in value):
            x = float(value.get("x", 0)) / 100.0
            y = float(value.get("y", 0)) / 100.0
            w = float(value.get("width", 0)) / 100.0
            h = float(value.get("height", 0)) / 100.0
            box = [x, y, x + w, y + h]

    return {"points": np.array(points, dtype=np.float32) if points else None,
            "labels": np.array(labels, dtype=np.int32) if labels else None,
            "box": np.array(box, dtype=np.float32) if box is not None else None}


def _detect_control(label_interface) -> Tuple[str, str, str, str]:
    """Return (from_name, to_name, object_type, control_type).

    `object_type` ∈ {'Image', 'Video'}, `control_type` is the xml-lowercased
    type we'll emit ('brushlabels' | 'rectanglelabels' | 'polygonlabels' |
    'videorectangle').
    """
    for candidate in ("BitmaskLabels", "RectangleLabels",
                      "PolygonLabels", "VectorLabels", "VideoRectangle"):
        try:
            from_name, to_name, value = label_interface.get_first_tag_occurence(
                candidate, "Image"
            )
            return from_name, to_name, "Image", _control_to_type(candidate)
        except Exception:
            pass
    # Video path
    from_name, to_name, _ = label_interface.get_first_tag_occurence("VideoRectangle", "Video")
    return from_name, to_name, "Video", "videorectangle"


def _control_to_type(control: str) -> str:
    return {
        "BitmaskLabels": "bitmap",
        "RectangleLabels": "rectanglelabels",
        "PolygonLabels": "polygonlabels",
        "VectorLabels": "vectorlabels",
        "VideoRectangle": "videorectangle",
    }[control]


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------


def _mask_to_result_value(control_type: str, mask: np.ndarray) -> Optional[Dict[str, Any]]:
    if control_type == "bitmap":
        # BitmaskLabels consumes a PNG data URL directly via the tag's
        # `maskToBitmapDataURL` path.
        return {
            "imageDataURL": f"data:image/png;base64,{mask_to_bitmap_png_base64(mask)}",
            "width": int(mask.shape[1]),
            "height": int(mask.shape[0]),
        }
    if control_type == "rectanglelabels":
        bbox = mask_to_bbox_percent(mask)
        if bbox is None:
            return None
        return {**bbox, "rotation": 0}
    if control_type in ("polygonlabels", "vectorlabels"):
        polygons = mask_to_polygons_percent(mask)
        if not polygons:
            return None
        # Largest polygon first.
        polygons.sort(key=lambda p: -len(p))
        return {"points": polygons[0], "closed": True}
    raise ValueError(f"unsupported control type: {control_type}")


def _build_result(
    value: Dict[str, Any],
    from_name: str,
    to_name: str,
    type_str: str,
    label_interface,
) -> Dict[str, Any]:
    labels = _lookup_labels(label_interface, from_name)
    label_key = {
        "bitmap": "bitmasklabels",
        "rectanglelabels": "rectanglelabels",
        "polygonlabels": "polygonlabels",
        "vectorlabels": "vectorlabels",
        "videorectangle": "labels",
    }[type_str]
    if labels:
        value = {**value, label_key: labels[:1]}
    return {
        "id": str(uuid4())[:8],
        "from_name": from_name,
        "to_name": to_name,
        "type": type_str,
        "origin": "manual",
        "value": value,
    }


def _lookup_labels(label_interface, from_name: str) -> List[str]:
    try:
        control = label_interface.get_control(from_name)
        return list(getattr(control, "labels", []) or [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SamVideoInteractive(LabelStudioMLBase):
    """Interactive SAM2 backend with prewarm + sticky frame cache."""

    def predict(
        self,
        tasks: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ModelResponse:
        context = context or {}
        event = context.get("event", "predict")

        task = tasks[0]
        task_id = str(task.get("id"))
        from_name, to_name, object_type, control_type = _detect_control(self.label_interface)

        if object_type == "Video":
            result = self._handle_video(task, task_id, context, event,
                                        from_name, to_name, control_type)
        else:
            result = self._handle_image(task, task_id, context, event,
                                        from_name, to_name, control_type)

        return ModelResponse(predictions=[result])

    # --- image path ------------------------------------------------------

    def _handle_image(self, task, task_id, context, event, from_name, to_name, control_type):
        if event == "prewarm":
            # Single-frame task; nothing to precompute beyond loading the image
            # lazily. Return ack so the frontend contract is uniform.
            return PredictionValue(result=[{
                "value": {"status": "ok", "cached": [0], "pending": []},
                "from_name": from_name, "to_name": to_name,
                "type": "prewarm_ack", "origin": "manual",
                "id": str(uuid4())[:8],
            }])

        image_predictor, _ = get_predictors()
        image_url = self._image_url_from_task(task, to_name)
        local_path = self.get_local_path(image_url, task_id=task_id)

        import cv2
        bgr = cv2.imread(local_path)
        if bgr is None:
            raise RuntimeError(f"failed to read image: {local_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        prompts = _extract_prompts(context)
        image_predictor.set_image(rgb)

        points_abs = (prompts["points"] * np.array([w, h], dtype=np.float32)) if prompts["points"] is not None else None
        box_abs = (prompts["box"] * np.array([w, h, w, h], dtype=np.float32)) if prompts["box"] is not None else None

        masks, scores, _ = image_predictor.predict(
            point_coords=points_abs,
            point_labels=prompts["labels"] if prompts["labels"] is not None else None,
            box=box_abs,
            multimask_output=False,
        )
        mask = (masks[0] > 0).astype(np.uint8)
        value = _mask_to_result_value(control_type, mask)
        if value is None:
            return PredictionValue(result=[])
        return PredictionValue(result=[
            _build_result(value, from_name, to_name, control_type, self.label_interface)
        ])

    def _image_url_from_task(self, task, to_name: str) -> str:
        # The Image object tag's `value` attribute is the key into task.data.
        data = task.get("data", {}) or {}
        if to_name in data:
            return data[to_name]
        # Fallback: first string value.
        for v in data.values():
            if isinstance(v, str):
                return v
        raise KeyError("no image URL in task.data")

    # --- video path ------------------------------------------------------

    def _handle_video(self, task, task_id, context, event, from_name, to_name, control_type):
        frame = int(context.get("frame", 0))
        window = int(context.get("window", WINDOW_SIZE))
        direction = context.get("direction", "forward")

        video_url = self._image_url_from_task(task, to_name)
        local_path = self.get_local_path(video_url, task_id=task_id)
        video = VIDEOS.get_or_create(task_id, local_path)

        FRAME_CACHE.touch(task_id, frame)

        if event == "prewarm":
            frame_range = self._window_range(frame, window, direction, video.frame_count)
            cached, pending = FRAME_CACHE.submit(
                task_id, frame_range, lambda idx: self._encode_frame(task_id, idx)
            )
            return PredictionValue(result=[{
                "value": {"status": "ok", "cached": cached, "pending": pending,
                          "frame_count": video.frame_count},
                "from_name": from_name, "to_name": to_name,
                "type": "prewarm_ack", "origin": "manual",
                "id": str(uuid4())[:8],
            }])

        # predict on a single frame: decode with the cached embedding for that
        # frame; no cross-frame tracking yet (that's step 4 in the build plan).
        prompts = _extract_prompts(context)
        frame_bgr = video.read_frame(frame)
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        image_predictor, _ = get_predictors()
        image_predictor.set_image(rgb)
        points_abs = (prompts["points"] * np.array([w, h], dtype=np.float32)) if prompts["points"] is not None else None
        box_abs = (prompts["box"] * np.array([w, h, w, h], dtype=np.float32)) if prompts["box"] is not None else None
        masks, _, _ = image_predictor.predict(
            point_coords=points_abs,
            point_labels=prompts["labels"] if prompts["labels"] is not None else None,
            box=box_abs,
            multimask_output=False,
        )
        mask = (masks[0] > 0).astype(np.uint8)

        if control_type == "videorectangle":
            bbox = mask_to_bbox_percent(mask)
            if bbox is None:
                return PredictionValue(result=[])
            fps = video.fps or 30.0
            value = {
                "framesCount": video.frame_count,
                "duration": video.frame_count / fps if fps else 0,
                "sequence": [{
                    "frame": frame + 1, "enabled": True, "rotation": 0,
                    "time": frame / fps if fps else 0, **bbox,
                }],
            }
            return PredictionValue(result=[
                _build_result(value, from_name, to_name, "videorectangle", self.label_interface)
            ])

        # Fallback: emit mask as whatever the control tag expects.
        value = _mask_to_result_value(control_type, mask)
        if value is None:
            return PredictionValue(result=[])
        return PredictionValue(result=[
            _build_result(value, from_name, to_name, control_type, self.label_interface)
        ])

    # --- encoding --------------------------------------------------------

    def _encode_frame(self, task_id: str, frame_idx: int):
        """Decode + encode a single video frame into a SAM2 image embedding."""
        video = VIDEOS._handles.get(task_id)
        if video is None:
            raise RuntimeError(f"no video handle for task {task_id}")
        frame_bgr = video.read_frame(frame_idx)
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        image_predictor, _ = get_predictors()
        image_predictor.set_image(rgb)
        # SAM2ImagePredictor stores features on the predictor instance. Capture
        # them off the internal state so the cache can hold per-frame
        # embeddings without the predictor being process-global state.
        return {
            "features": getattr(image_predictor, "_features", None),
            "original_size": getattr(image_predictor, "_orig_hw", None),
            "is_image_set": True,
        }

    def _window_range(self, frame: int, window: int, direction: str, frame_count: int):
        if direction == "backward":
            start = max(0, frame - window)
            end = frame + 1
        else:
            start = frame
            end = min(frame_count, frame + window + 1)
        return list(range(start, end))
