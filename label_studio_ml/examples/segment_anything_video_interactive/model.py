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
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

from control_detect import control_to_type, detect_control
from frame_cache import FrameCache
from frame_resolve import resolve_frame_index
from ls_auth import ls_auth_headers
from mask_encoding import (
    mask_to_bbox_percent,
    mask_to_bitmap_png_base64,
    mask_to_polygons_percent,
)
from video_state import VideoRegistry

logger = logging.getLogger(__name__)

DEVICE = os.getenv("DEVICE", "cuda")
SEGMENT_ANYTHING_2_REPO_PATH = os.getenv("SEGMENT_ANYTHING_2_REPO_PATH", "segment-anything-2")
# SAM 2.1 by default — `download_ckpts.sh` now fetches only sam2.1_* weights,
# and the matching configs live under the `configs/sam2.1/` Hydra group. Config
# size must match the checkpoint size.
MODEL_CONFIG = os.getenv("MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "sam2.1_hiera_large.pt")
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "20"))
MAX_FRAMES_TO_TRACK = int(os.getenv("MAX_FRAMES_TO_TRACK", "300"))

# How long a single `track_progress` call may hold the HTTP request waiting
# for the *first* new frame before returning an empty batch. Long polling
# eliminates idle-state spam: the FE's next request won't fire until data
# is ready or this timeout hits.
#
# Tune downward if gunicorn sync workers become scarce (each idle tracker
# ties up one worker for up to this many seconds at a time).
TRACK_PROGRESS_WAIT_SECONDS = float(os.getenv("TRACK_PROGRESS_WAIT_SECONDS", "5.0"))

# Once the first frame is ready, keep the response open for a short grace
# window so rapidly-produced frames pile up in the same response. Without
# this, active tracking spams at SAM2's per-frame cadence (~20-30 req/s)
# because each poll drains 1 frame and returns immediately.
#
# Total frames per response ≈ BATCH_WINDOW_SECONDS × frame_rate. Larger
# window = fewer requests, slightly higher per-frame latency.
TRACK_PROGRESS_BATCH_WINDOW_SECONDS = float(
    os.getenv("TRACK_PROGRESS_BATCH_WINDOW_SECONDS", "0.25")
)
TRACK_PROGRESS_MAX_BATCH = int(os.getenv("TRACK_PROGRESS_MAX_BATCH", "32"))

# Stop-tracking thresholds (see _run_tracking).
# SAM2's mask decoder emits a per-frame object_score_logits; convention is
# `> 0` = object present, `<= 0` = occluded/absent. We debounce across a few
# frames so a single brief occlusion doesn't terminate the track.
MIN_OBJECT_SCORE = float(os.getenv("SAM_MIN_OBJECT_SCORE", "0.0"))
OBJECT_LOST_DEBOUNCE = int(os.getenv("SAM_OBJECT_LOST_DEBOUNCE", "3"))
MAX_FOREGROUND_RATIO = float(os.getenv("SAM_MAX_FOREGROUND_RATIO", "0.7"))


# ---------------------------------------------------------------------------
# SAM2 model loading
# ---------------------------------------------------------------------------


def _build_models():
    """Lazily build the SAM2 models (weights only, shared across threads).

    Returns (image_model, video_predictor). The image_model is wrapped in a
    per-call SAM2ImagePredictor at use sites — no shared mutable state.
    """
    from sam2.build_sam import build_sam2, build_sam2_video_predictor

    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    ckpt = str(
        pathlib.Path(__file__).parent / SEGMENT_ANYTHING_2_REPO_PATH / "checkpoints" / MODEL_CHECKPOINT
    )
    image_model = build_sam2(MODEL_CONFIG, ckpt, device=DEVICE)
    video_predictor = build_sam2_video_predictor(MODEL_CONFIG, ckpt, device=DEVICE)

    # Pre-warm torch.jit.script by creating one throwaway predictor on the
    # main thread. Subsequent SAM2ImagePredictor() calls reuse the cached
    # JIT compilation and are fast + thread-safe.
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2ImagePredictor(image_model)

    return image_model, video_predictor


_models: Optional[Tuple[Any, Any]] = None
_models_lock = threading.Lock()


def get_models():
    global _models
    with _models_lock:
        if _models is None:
            _models = _build_models()
        return _models


def make_image_predictor():
    """Create a per-call image predictor. The underlying model weights are
    shared (read-only); only the predictor wrapper (features, state) is
    per-call, so no lock is needed."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    image_model, _ = get_models()
    return SAM2ImagePredictor(image_model)


def get_video_predictor():
    _, video_predictor = get_models()
    return video_predictor


# ---------------------------------------------------------------------------
# Shared per-process state
# ---------------------------------------------------------------------------

FRAME_CACHE = FrameCache()
VIDEOS = VideoRegistry()

# Pre-initialize SAM2 predictors on module load (main thread) so background
# threads never hit torch.jit.script which is not thread-safe.
try:
    get_models()
    logger.info("SAM2 predictors pre-initialized on startup")
except Exception as e:
    logger.warning("SAM2 pre-init failed (will retry on first request): %s", e)


# ---------------------------------------------------------------------------
# Tracking sessions — background SAM2 propagation with poll-based progress
# ---------------------------------------------------------------------------

from collections import deque


class TrackingSession:
    """Holds state for an async tracking job. The background thread appends
    frame masks; the frontend polls for new ones via track_progress."""

    def __init__(self, session_id: str, total_frames: int, producers: int = 1):
        self.session_id = session_id
        self.total_frames = total_frames
        self.frames: List[Dict[str, Any]] = []
        self.cursor = 0  # how many frames the client has fetched
        self.done = False
        self.error: Optional[str] = None
        self.cancelled = False
        # Bidirectional tracking runs two producer threads sharing this
        # session; we only flip `done` after all of them report in.
        self._remaining_producers = max(1, producers)
        self.lock = threading.RLock()
        # Signaled whenever there's something new for a poller to see:
        # a frame was appended, the session finished, errored, or was cancelled.
        self.new_data_event = threading.Event()

    def append_frame(self, frame_data: Dict[str, Any]):
        with self.lock:
            self.frames.append(frame_data)
        self.new_data_event.set()

    def drain_new(self) -> Tuple[List[Dict[str, Any]], int, bool]:
        """Return (new_frames, total_produced, is_done)."""
        with self.lock:
            new = self.frames[self.cursor:]
            self.cursor = len(self.frames)
            return new, len(self.frames), self.done

    def finish(self):
        """Single-producer shorthand: mark the session done immediately."""
        with self.lock:
            self.done = True
            self._remaining_producers = 0
        self.new_data_event.set()

    def producer_done(self):
        """One of N parallel producers reports completion. The session only
        flips to `done=True` after every producer has called this — the FE
        long-poller keeps getting fresh frames from any still-running
        direction until then."""
        with self.lock:
            self._remaining_producers = max(0, self._remaining_producers - 1)
            if self._remaining_producers == 0:
                self.done = True
        self.new_data_event.set()

    def cancel(self):
        self.cancelled = True
        self.new_data_event.set()

    def wait_for_new_data(self, timeout: float) -> bool:
        """Block until `new_data_event` fires or `timeout` elapses. Returns
        True if signalled, False on timeout. Callers drain afterwards."""
        signalled = self.new_data_event.wait(timeout)
        # Clear immediately so the next wait starts fresh. It's fine if we
        # miss a set here — the caller drains after wake and the next wait
        # will re-fire as soon as the producer signals again.
        self.new_data_event.clear()
        return signalled


_tracking_sessions: Dict[str, TrackingSession] = {}
_tracking_lock = threading.RLock()


def _resolve_be_frame(context: Dict[str, Any], video) -> int:
    """Translate a frontend-supplied timestamp (ms) or 1-indexed frame into
    the BE's 0-indexed frame space using the video's own fps.

    Delegates to the dependency-free :func:`resolve_frame_index` so the
    conversion can be unit-tested without torch/cv2 (see test_frame_resolve.py).
    """
    return resolve_frame_index(
        context.get("time_ms"),
        context.get("frame", 1),
        video.fps,
        video.frame_count,
    )


def _object_score(inference_state, obj_idx: int, frame_idx: int) -> Optional[float]:
    """Return SAM2's per-frame object-presence logit, or None if unavailable.

    SAM2 convention: `> 0` => object present, `<= 0` => occluded/absent.
    """
    try:
        per_obj = inference_state["output_dict_per_obj"][obj_idx]
        frame_out = per_obj["non_cond_frame_outputs"].get(frame_idx)
        if frame_out is None:
            return None
        score = frame_out.get("object_score_logits")
        if score is None:
            return None
        return float(score.item() if hasattr(score, "item") else score)
    except (KeyError, AttributeError, TypeError):
        return None


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


# Control-tag detection lives in the dependency-free `control_detect` module so
# it can be unit-tested without torch/cv2 (see test_control_detect.py).
_detect_control = detect_control
_control_to_type = control_to_type


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
        "videovectorlabels": "labels",
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


_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _pick_best_mask(masks: np.ndarray, scores: Optional[np.ndarray]) -> np.ndarray:
    """Select the highest-scoring candidate from a `multimask_output=True`
    prediction. SAM typically returns 3 masks at (small / medium / large)
    granularity; the FE fragments badly when the small-scale one wins on a
    high-detail object, so we go by the model's own IoU-style score."""
    if masks.ndim == 2:
        return masks
    if scores is None or len(scores) == 0:
        return masks[0]
    idx = int(np.argmax(scores))
    return masks[idx]


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Post-process a raw SAM binary mask before sending it to the FE:

    - Morphological close with a 5×5 ellipse seals pinhole gaps SAM
      occasionally leaves on textured objects.
    - Keep only the largest 8-connected component so stray speckles outside
      the main object don't survive. Works as a belt with FE's
      `keepLargestComponent` for when older FE clients consume the mask.
    """
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return binary
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num <= 1:
        return closed
    # Skip background (label 0); pick the largest foreground component by area.
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + int(np.argmax(areas))
    return (labels == best_label).astype(np.uint8)


def _is_loopback_host(host: str) -> bool:
    """LS sends its own `settings.HOSTNAME` in `/setup` — if the LS admin
    didn't set it, LS's own api_connector falls through to `http://localhost:<port>`.
    That value is never useful to an ML backend (even on the same box,
    operators set `LABEL_STUDIO_URL` explicitly), so we refuse to cache it."""
    if not host:
        return False
    h = host.lower()
    return "localhost" in h or "127.0.0.1" in h or "://0.0.0.0" in h


def _capture_ls_context_from_request() -> None:
    """Cache `hostname` + `access_token` from a `/setup` payload.

    Label Studio sends both fields in every `/setup` request, but the
    `label-studio-ml-backend` base class only persists `extra_params` — so
    by default the ML backend has no way to learn the LS URL except via the
    `LABEL_STUDIO_URL` env var. Reading the Flask request here makes LS a
    fallback source of truth when the env var isn't set (env var still wins
    in `_ls_host_token`).
    """
    try:
        from flask import request, has_request_context
    except Exception:
        return
    if not has_request_context():
        return
    data = request.get_json(silent=True) or {}
    host = (data.get("hostname") or "").rstrip("/")
    token = data.get("access_token") or ""
    if host and not _is_loopback_host(host):
        LS_CONTEXT["url"] = host
    elif host:
        logger.info(
            "ignoring loopback hostname from /setup (%s) — set LABEL_STUDIO_URL "
            "on the ML backend to point at a reachable LS URL",
            host,
        )
    if token:
        LS_CONTEXT["token"] = token


# Module-level cache populated by `/setup` and read by `_resolve_video_source`.
LS_CONTEXT: Dict[str, Optional[str]] = {"url": None, "token": None}


class SamVideoInteractive(LabelStudioMLBase):
    """Interactive SAM2 backend with prewarm + sticky frame cache."""

    def setup(self) -> None:
        # Base `LabelStudioMLBase.setup` is an empty hook; this override runs
        # on every model instantiation (the backend creates a fresh instance
        # per HTTP request, so this fires on both `/setup` and `/predict`).
        # `/setup` carries the LS hostname + token; `/predict` doesn't, so
        # we just keep whatever was cached on the last setup call.
        _capture_ls_context_from_request()

    def predict(
        self,
        tasks: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ModelResponse:
        context = context or {}
        event = context.get("event", "predict")

        # `capabilities` is task-agnostic: the FE polls it once per backend
        # to discover which control tags this model can drive, so that SAM
        # interactions can be wired automatically instead of requiring a
        # <SegmentAnything> tag in the config.
        if event == "capabilities":
            return ModelResponse(predictions=[self._handle_capabilities()])

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

    # --- capability discovery -------------------------------------------

    def _handle_capabilities(self) -> PredictionValue:
        """Static capability declaration so LS can auto-bind this backend to
        any compatible control tags in the project without user config.

        The FE reads this once per backend and builds `InteractiveBinding`s
        for each (backend × control-tag) pair where `tag` matches a control
        present in the annotation config."""
        return PredictionValue(result=[{
            "id": str(uuid4())[:8],
            "type": "capabilities",
            "origin": "manual",
            "value": {
                "prompts": ["point", "box"],
                "targets": [
                    {"tag": "BitmaskLabels",      "output": "mask"},
                    {"tag": "RectangleLabels",    "output": "bbox"},
                    {"tag": "PolygonLabels",      "output": "polygon"},
                    {"tag": "VectorLabels",       "output": "polygon"},
                    {"tag": "VideoRectangle",     "output": "bbox",    "features": ["track"]},
                    {"tag": "VideoVectorLabels",  "output": "polygon", "features": ["track"]},
                ],
                "model_info": {
                    "name": "SAM2",
                    "version": MODEL_CHECKPOINT,
                },
            },
        }])

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

        prompts = _extract_prompts(context)
        if prompts["points"] is None and prompts["box"] is None:
            return PredictionValue(result=[])

        image_predictor = make_image_predictor()
        image_url = self._image_url_from_task(task, to_name)
        ls_host, ls_token = self._ls_host_token()
        local_path = self.get_local_path(
            image_url, task_id=task_id, ls_host=ls_host, ls_access_token=ls_token,
        )

        bgr = cv2.imread(local_path)
        if bgr is None:
            raise RuntimeError(f"failed to read image: {local_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        image_predictor.set_image(rgb)

        points_abs = (prompts["points"] * np.array([w, h], dtype=np.float32)) if prompts["points"] is not None else None
        box_abs = (prompts["box"] * np.array([w, h, w, h], dtype=np.float32)) if prompts["box"] is not None else None

        masks, scores, _ = image_predictor.predict(
            point_coords=points_abs,
            point_labels=prompts["labels"] if prompts["labels"] is not None else None,
            box=box_abs,
            multimask_output=True,
        )
        mask = _clean_mask(_pick_best_mask(masks, scores))
        return self._mask_response(mask, w, h, from_name, to_name)

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
        # Lightweight events — no video setup needed
        if event == "release":
            FRAME_CACHE.drop_task(task_id)
            VIDEOS.drop(task_id)
            logger.info("release: cleared cache for task %s", task_id)
            return PredictionValue(result=[{
                "value": {"status": "released"},
                "from_name": from_name, "to_name": to_name,
                "type": "release_ack", "origin": "manual",
                "id": str(uuid4())[:8],
            }])

        if event == "track_progress":
            return self._handle_track_progress(context, from_name, to_name)

        if event == "track_cancel":
            return self._handle_track_cancel(context, from_name, to_name)

        # Events below need video handle
        window = int(context.get("window", WINDOW_SIZE))
        direction = context.get("direction", "forward")

        raw_url = self._image_url_from_task(task, to_name)
        source, headers = self._resolve_video_source(raw_url, task_id)
        try:
            video = VIDEOS.get_or_create(task_id, source, headers=headers)
        except Exception as e:
            logger.warning("streaming failed (%s), falling back to download", e)
            # A previously-resolved handle for the same task is still usable
            # — don't force another probe / download cycle (which can 429 on
            # the LS token-refresh endpoint under load).
            cached = VIDEOS._handles.get(task_id)
            if cached is not None:
                video = cached
            else:
                ls_host, ls_token = self._ls_host_token()
                local_path = self.get_local_path(
                    raw_url, task_id=task_id, ls_host=ls_host, ls_access_token=ls_token,
                )
                video = VIDEOS.get_or_create(task_id, local_path)

        # Prefer `time` (seconds) — the only quantity FE and BE can agree on
        # without knowing each other's fps. Fall back to `frame` (FE 1-indexed)
        # for legacy callers that don't send time.
        frame = _resolve_be_frame(context, video)

        FRAME_CACHE.touch(task_id, frame)

        if event == "prewarm":
            frame_range = self._window_range(frame, window, direction, video.frame_count)
            # Fetch the whole window in ONE ffmpeg pass (one HTTP request to LS)
            # rather than one request per frame, which trips LS's rate limit.
            # The encode step pulls decoded frames from this buffer, falling
            # back to a per-frame read only for any the prefetch missed.
            prefetched = self._prefetch_window(video, frame_range)

            def _encode(idx, _video=video, _frames=prefetched):
                bgr = _frames.get(idx)
                if bgr is None:
                    bgr = _video.read_frame(idx)
                return self._encode_bgr(bgr)

            cached, pending = FRAME_CACHE.submit(task_id, frame_range, _encode)
            return PredictionValue(result=[{
                "value": {"status": "ok", "cached": cached, "pending": pending,
                          "frame_count": video.frame_count},
                "from_name": from_name, "to_name": to_name,
                "type": "prewarm_ack", "origin": "manual",
                "id": str(uuid4())[:8],
            }])

        if event == "track":
            return self._handle_track(
                task_id, context, video, frame,
                from_name, to_name, control_type,
            )

        # Single-frame predict: return mask PNG for the frontend preview.
        return self._predict_single_frame(
            task_id, context, video, frame,
            from_name, to_name,
        )

    def _predict_single_frame(self, task_id, context, video, frame,
                              from_name, to_name):
        prompts = _extract_prompts(context)
        if prompts["points"] is None and prompts["box"] is None:
            return PredictionValue(result=[])

        image_predictor = make_image_predictor()
        h, w = video.height, video.width

        # Try to restore from frame cache (pre-encoded by prewarm)
        cached = FRAME_CACHE.get(task_id, frame)
        if cached is not None and cached.get("features") is not None:
            image_predictor._features = cached["features"]
            image_predictor._orig_hw = cached["original_size"]
            image_predictor._is_image_set = True
            logger.debug("predict: cache hit task=%s frame=%s", task_id, frame)
        else:
            frame_bgr = video.read_frame(frame)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            image_predictor.set_image(rgb)
            features_snapshot = {
                "features": image_predictor._features,
                "original_size": image_predictor._orig_hw,
                "is_image_set": True,
            }
            FRAME_CACHE.submit(task_id, [frame], lambda idx: features_snapshot)
            logger.debug("predict: cache miss task=%s frame=%s, encoded + cached", task_id, frame)

        points_abs = (prompts["points"] * np.array([w, h], dtype=np.float32)) if prompts["points"] is not None else None
        box_abs = (prompts["box"] * np.array([w, h, w, h], dtype=np.float32)) if prompts["box"] is not None else None
        masks, scores, _ = image_predictor.predict(
            point_coords=points_abs,
            point_labels=prompts["labels"] if prompts["labels"] is not None else None,
            box=box_abs,
            multimask_output=True,
        )
        mask = _clean_mask(_pick_best_mask(masks, scores))
        return self._mask_response(mask, w, h, from_name, to_name)

    def _ls_host_token(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (ls_host, ls_token) using env vars first, then the cache
        populated from `/setup`. Either may be None; callers pass them into
        `self.get_local_path(ls_host=..., ls_access_token=...)` which lets
        the SDK skip its `http://localhost:8000` default fallback.

        LS's own `/setup` payload can carry `http://localhost:<port>` when
        the LS side doesn't have `HOSTNAME` configured — that's useless to a
        remote ML backend. The env-var-first ordering means an operator can
        always override LS's guess via `.env` / `docker-compose`.
        """
        env_host = (os.getenv("LABEL_STUDIO_URL") or "").rstrip("/")
        env_token = os.getenv("LABEL_STUDIO_API_KEY") or ""
        cached_host = (LS_CONTEXT.get("url") or "").rstrip("/")
        cached_token = LS_CONTEXT.get("token") or ""
        host = env_host or cached_host
        token = env_token or cached_token
        return (host or None, token or None)

    def _resolve_video_source(self, raw_url: str, task_id: str):
        """Resolve a task video URL to a streamable source + auth headers.

        * Cloud storage URLs (don't look like LS's own host) → stream directly,
          no headers.
        * LS-hosted URLs, whether absolute or relative → attach the LS API
          token; LS guards the /data/upload/* path and returns 401 without it.
        * No LS url / key configured → fall back to local download.

        Resolution order for the LS hostname:
          1. `LABEL_STUDIO_URL` env var (explicit operator override)
          2. Cached value from the last `/setup` payload (LS is the source
             of truth when the env var isn't set — avoids the SDK's
             `http://localhost:8000` default).
        """
        ls_url_opt, api_key_opt = self._ls_host_token()
        ls_url = ls_url_opt or ""
        api_key = api_key_opt or ""

        def _auth_headers():
            # Handles both legacy tokens (`Token <key>`) and Personal Access
            # Tokens (exchanged for a short-lived `Bearer <access>` JWT).
            return ls_auth_headers(ls_url, api_key)

        def _points_at_ls(url: str) -> bool:
            if not ls_url:
                return False
            return url.startswith(f"{ls_url}/") or url == ls_url

        if raw_url.startswith("http://") or raw_url.startswith("https://"):
            # Absolute URL — attach auth iff it's an LS-hosted URL.
            return raw_url, _auth_headers() if _points_at_ls(raw_url) else None

        if ls_url and api_key:
            full_url = f"{ls_url}{raw_url}" if raw_url.startswith("/") else f"{ls_url}/{raw_url}"
            return full_url, _auth_headers()

        logger.warning("streaming not available (no LABEL_STUDIO_URL), falling back to download")
        local_path = self.get_local_path(
            raw_url, task_id=task_id, ls_host=ls_url_opt, ls_access_token=api_key_opt,
        )
        return local_path, None

    def _mask_response(self, mask, w, h, from_name, to_name):
        """Return a mask PNG as a prediction using `bitmask` — a recognized
        LS result type — so the editor won't reject it."""
        return PredictionValue(result=[{
            "id": str(uuid4())[:8],
            "from_name": from_name,
            "to_name": to_name,
            "type": "bitmask",
            "origin": "manual",
            "value": {
                "imageDataURL": f"data:image/png;base64,{mask_to_bitmap_png_base64(mask)}",
                "width": int(w),
                "height": int(h),
            },
        }])

    def _handle_track(self, task_id, context, video, prompt_frame,
                      from_name, to_name, control_type):
        """Start async SAM2 video propagation. Returns a session_id immediately;
        the frontend polls track_progress to get results incrementally.

        `direction` supports "forward", "backward", and "both". For "both"
        we spawn two independent producer threads sharing the session —
        each direction has its own `propagate_in_video` iterator and its
        own auto-stop state, so one side can terminate on object loss while
        the other continues until it also loses the object or hits the end
        of the video.
        """
        prompts = _extract_prompts(context)
        max_duration_ms = context.get("max_duration_ms")
        if max_duration_ms is not None and video.fps:
            max_frames = int(round((float(max_duration_ms) / 1000.0) * video.fps))
        else:
            max_frames = int(context.get("max_frames", MAX_FRAMES_TO_TRACK))
        direction = context.get("direction", "forward")
        if direction not in ("forward", "backward", "both"):
            direction = "forward"

        h, w = video.height, video.width

        # Each direction gets its own frame range. "both" covers the full
        # [prompt - max_frames, prompt + max_frames] span.
        fwd_start, fwd_end = prompt_frame, min(video.frame_count, prompt_frame + max_frames + 1)
        bwd_start, bwd_end = max(0, prompt_frame - max_frames), prompt_frame + 1

        if direction == "forward":
            ranges = [("forward", fwd_start, fwd_end)]
        elif direction == "backward":
            ranges = [("backward", bwd_start, bwd_end)]
        else:  # "both"
            ranges = [("forward", fwd_start, fwd_end), ("backward", bwd_start, bwd_end)]

        total = sum(end - start for _, start, end in ranges)
        session_id = str(uuid4())[:12]
        session = TrackingSession(session_id, total, producers=len(ranges))

        with _tracking_lock:
            _tracking_sessions[session_id] = session

        for d, start_frame, end_frame in ranges:
            t = threading.Thread(
                target=self._run_tracking,
                args=(session, video, prompts, start_frame, end_frame,
                      prompt_frame, max_frames, w, h, d),
                daemon=True,
            )
            t.start()

        logger.info("track: started session=%s task=%s direction=%s ranges=%s",
                     session_id, task_id, direction, ranges)

        return PredictionValue(result=[{
            "id": str(uuid4())[:8],
            "from_name": from_name, "to_name": to_name,
            "type": "track_started",
            "origin": "manual",
            "value": {
                "session_id": session_id,
                "total_frames": total,
                "fps": video.fps,
                "duration_ms": (video.frame_count * 1000.0 / video.fps) if video.fps else 0.0,
            },
        }])

    def _run_tracking(self, session, video, prompts, start_frame, end_frame,
                      prompt_frame, max_frames, w, h, direction="forward"):
        """Background thread: extract frames, run SAM2 propagation, push
        mask PNGs into the session as they're produced."""
        try:
            video_predictor = get_video_predictor()
            frame_count_needed = end_frame - start_frame

            with tempfile.TemporaryDirectory() as frame_dir:
                written = video.write_frame_range_as_jpegs(
                    start_frame, frame_count_needed, frame_dir)
                logger.info("track bg: extracted %d frames to %s", written, frame_dir)

                # async_loading_frames=True makes SAM2 return immediately from
                # init_state and decode/normalize JPEGs in a background thread
                # as propagate_in_video walks through them — the wait for
                # "encode all frames upfront" becomes "encode frame 0".
                #
                # offload_video_to_cpu=True keeps the loaded frames on CPU so
                # only the propagation thread touches the GPU. Without this
                # the async loader pushes frames to the device with
                # non_blocking=True while propagation is concurrently running
                # the backbone — the two streams race and SAM2 sometimes
                # processes partly-copied tensors, which shows up as the
                # mask "jumping" onto a different object.
                inference_state = video_predictor.init_state(
                    video_path=frame_dir,
                    async_loading_frames=True,
                    offload_video_to_cpu=True,
                )
                video_predictor.reset_state(inference_state)

                relative_prompt_frame = prompt_frame - start_frame
                if prompts["points"] is not None:
                    points_abs = prompts["points"] * np.array([w, h], dtype=np.float32)
                    video_predictor.add_new_points(
                        inference_state=inference_state,
                        frame_idx=relative_prompt_frame,
                        obj_id=0,
                        points=points_abs,
                        labels=prompts["labels"],
                    )
                elif prompts["box"] is not None:
                    box_abs = (prompts["box"] * np.array([w, h, w, h], dtype=np.float32)).astype(np.float32)
                    # Prefer the modern SAM2 API which takes `box` as a
                    # dedicated parameter. The legacy "two points with
                    # labels [2, 3]" encoding is interpreted as two
                    # unknown-label points by current SAM2 builds — that
                    # was producing a degenerate mask at the prompt
                    # frame, which then propagated as a tiny region at
                    # the top-left of the video.
                    if hasattr(video_predictor, "add_new_points_or_box"):
                        video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=relative_prompt_frame,
                            obj_id=0,
                            box=box_abs,
                        )
                    else:
                        x1, y1, x2, y2 = box_abs
                        points_abs = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                        box_labels = np.array([2, 3], dtype=np.int32)
                        video_predictor.add_new_points(
                            inference_state=inference_state,
                            frame_idx=relative_prompt_frame,
                            obj_id=0,
                            points=points_abs,
                            labels=box_labels,
                        )

                total_pixels = w * h
                consecutive_lost = 0

                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=relative_prompt_frame,
                    max_frame_num_to_track=max_frames,
                    reverse=(direction == "backward"),
                ):
                    if session.cancelled:
                        logger.info("track bg: cancelled session=%s", session.session_id)
                        break

                    real_frame_idx = out_frame_idx + start_frame
                    stop = False

                    for i, _ in enumerate(out_obj_ids):
                        # SAM2-native presence check: the decoder's object_score_logits
                        # is the model's own "is this object here?" prediction.
                        score = _object_score(inference_state, i, out_frame_idx)
                        mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                        fg_count = int(mask.sum())

                        object_lost = (
                            (score is not None and score < MIN_OBJECT_SCORE)
                            or fg_count == 0
                        )

                        if object_lost:
                            consecutive_lost += 1
                            if consecutive_lost >= OBJECT_LOST_DEBOUNCE:
                                logger.info(
                                    "track bg: object lost (score=%s, fg=%d) for %d frames, "
                                    "stopping at frame %d",
                                    f"{score:.2f}" if score is not None else "n/a",
                                    fg_count, consecutive_lost, real_frame_idx,
                                )
                                stop = True
                                break
                            # Skip emitting this low-confidence frame; keep propagating.
                            continue

                        consecutive_lost = 0

                        fg_ratio = fg_count / total_pixels
                        if fg_ratio > MAX_FOREGROUND_RATIO:
                            logger.info(
                                "track bg: foreground ratio %.2f > %.2f at frame %d, stopping",
                                fg_ratio, MAX_FOREGROUND_RATIO, real_frame_idx,
                            )
                            stop = True
                            break

                        session.append_frame({
                            "frame": real_frame_idx,
                            # Emit time (ms) so the FE can recompute its own
                            # frame index without knowing the BE's fps —
                            # avoids frame drift when FE and BE see different
                            # fps for the same video.
                            "time_ms": (real_frame_idx * 1000.0 / video.fps) if video.fps else 0.0,
                            "imageDataURL": f"data:image/png;base64,{mask_to_bitmap_png_base64(mask)}",
                            "width": int(w),
                            "height": int(h),
                        })

                    if stop:
                        logger.info("track bg: stopped at frame %d (produced %d frames)",
                                    real_frame_idx, len(session.frames))
                        break

        except Exception as e:
            logger.exception("track bg: error session=%s direction=%s", session.session_id, direction)
            session.error = str(e)
        finally:
            session.producer_done()
            logger.info("track bg: finished session=%s direction=%s frames=%d",
                         session.session_id, direction, len(session.frames))

    def _handle_track_progress(self, context, from_name, to_name):
        session_id = context.get("session_id", "")
        with _tracking_lock:
            session = _tracking_sessions.get(session_id)
        if not session:
            return PredictionValue(result=[{
                "id": str(uuid4())[:8],
                "from_name": from_name, "to_name": to_name,
                "type": "track_progress", "origin": "manual",
                "value": {"error": "session not found", "done": True},
            }])

        # Long polling: drain once; if there's nothing yet and the session is
        # still running, block up to TRACK_PROGRESS_WAIT_SECONDS for the
        # background thread to signal (frame produced, finished, errored, or
        # cancelled), then drain again.
        new_frames, total_produced, done = session.drain_new()
        if not new_frames and not done and session.error is None:
            session.wait_for_new_data(TRACK_PROGRESS_WAIT_SECONDS)
            new_frames, total_produced, done = session.drain_new()

        # Micro-batch: once we have at least one frame and the session is
        # still running, hold the response for a short window so more frames
        # coalesce into this response. Without this, active tracking yields
        # ~SAM2_fps requests/s because each poll drains 1 frame and returns
        # immediately. With a 0.25 s window we send ~4 responses/s instead,
        # each carrying multiple frames.
        if new_frames and not done and session.error is None:
            deadline = time.monotonic() + TRACK_PROGRESS_BATCH_WINDOW_SECONDS
            while len(new_frames) < TRACK_PROGRESS_MAX_BATCH:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if not session.wait_for_new_data(remaining):
                    break  # timeout
                more, total_produced, done = session.drain_new()
                new_frames.extend(more)
                if done or session.error is not None or session.cancelled:
                    break

        if done:
            with _tracking_lock:
                _tracking_sessions.pop(session_id, None)

        return PredictionValue(result=[{
            "id": str(uuid4())[:8],
            "from_name": from_name, "to_name": to_name,
            "type": "track_progress", "origin": "manual",
            "value": {
                "frames": new_frames,
                "produced": total_produced,
                "total": session.total_frames,
                "done": done,
                "error": session.error,
            },
        }])

    def _handle_track_cancel(self, context, from_name, to_name):
        session_id = context.get("session_id", "")
        with _tracking_lock:
            session = _tracking_sessions.get(session_id)
        if session:
            session.cancel()
        return PredictionValue(result=[{
            "id": str(uuid4())[:8],
            "from_name": from_name, "to_name": to_name,
            "type": "track_cancel_ack", "origin": "manual",
            "value": {"status": "cancelled"},
        }])

    # --- helpers ---------------------------------------------------------

    def _encode_frame(self, task_id: str, frame_idx: int):
        """Decode + encode a single video frame into a SAM2 image embedding."""
        video = VIDEOS._handles.get(task_id)
        if video is None:
            raise RuntimeError(f"no video handle for task {task_id}")
        return self._encode_bgr(video.read_frame(frame_idx))

    def _encode_bgr(self, frame_bgr):
        """Encode an already-decoded BGR frame into a SAM2 image embedding.
        Split out from `_encode_frame` so prewarm can fetch a whole window in
        one ffmpeg pass (one HTTP request) and feed the frames in here, instead
        of one network read per frame (which trips LS rate limits)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_predictor = make_image_predictor()
        image_predictor.set_image(rgb)
        return {
            "features": image_predictor._features,
            "original_size": image_predictor._orig_hw,
            "is_image_set": True,
        }

    def _prefetch_window(self, video, frame_range: List[int]) -> Dict[int, Any]:
        """Fetch a contiguous frame window in a single ffmpeg pass to collapse
        one-HTTP-request-per-frame into one request for the whole window.

        Best-effort: returns {frame_idx: bgr_frame}; on any failure returns the
        frames it managed to get (possibly empty) and the encode step falls back
        to per-frame reads for the rest.
        """
        if not frame_range:
            return {}
        start = min(frame_range)
        count = max(frame_range) - start + 1
        try:
            frames = video.read_frame_range(start, count)
        except Exception as e:
            logger.warning("window prefetch failed (%s); falling back to per-frame", e)
            return {}
        return {start + i: f for i, f in enumerate(frames)}

    def _window_range(self, frame: int, window: int, direction: str, frame_count: int):
        if direction == "backward":
            start = max(0, frame - window)
            end = frame + 1
        else:
            start = frame
            end = min(frame_count, frame + window + 1)
        return list(range(start, end))
