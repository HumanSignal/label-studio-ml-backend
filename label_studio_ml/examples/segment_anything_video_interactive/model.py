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

import gc
import logging
import os
import pathlib
import queue
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
from PIL import Image

from control_detect import control_to_type, detect_control
from frame_cache import FrameCache
from frame_resolve import resolve_frame_index
from ls_auth import ls_token_for_sdk
from url_auth import should_attach_ls_auth
from mask_encoding import (
    mask_to_bbox_percent,
    mask_to_bitmap_png_base64,
    mask_to_polygons_percent,
)
from video_state import VideoRegistry, video_is_readable

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

# Schemes LS stores verbatim in task data for cloud-backed projects; kept in
# sync with the SDK's own check in `label_studio_tools.core.utils.io`.
CLOUD_URI_SCHEMES = ("s3://", "s3a://", "gs://", "azure-blob://")

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
TRACK_SESSION_TTL_SECONDS = float(os.getenv("TRACK_SESSION_TTL_SECONDS", "300"))
TRACK_SESSION_MAX_AGE_SECONDS = float(os.getenv("TRACK_SESSION_MAX_AGE_SECONDS", "1800"))
TRACKING_WORKERS = int(os.getenv("TRACKING_WORKERS", "1"))
TRACKING_RELEASE_MEMORY = os.getenv("TRACKING_RELEASE_MEMORY", "1").lower() in {"1", "true", "yes", "on"}
TRACKING_ON_DEMAND_FRAMES = os.getenv("TRACKING_ON_DEMAND_FRAMES", "1").lower() in {"1", "true", "yes", "on"}
TRACKING_FRAME_CACHE_SIZE = int(os.getenv("TRACKING_FRAME_CACHE_SIZE", "2"))

# Authenticated LS-hosted videos are intentionally downloaded through the SDK.
# ffmpeg forwards custom headers across redirects, so streaming them directly
# can leak the LS Authorization header to a redirected origin. External HTTP(S)
# videos still stream without LS auth.

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

_autocast_context: Optional[Any] = None


def _build_models():
    """Lazily build the SAM2 model (weights only, shared across threads).

    Returns (image_model, video_predictor). The image predictor can use the
    video predictor because SAM2VideoPredictor is a SAM2Base subclass. Sharing
    that one module avoids keeping two full SAM2-L copies resident.
    """
    from sam2.build_sam import build_sam2_video_predictor

    global _autocast_context
    if DEVICE == "cuda" and torch.cuda.is_available():
        if _autocast_context is None:
            _autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            _autocast_context.__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    ckpt = str(
        pathlib.Path(__file__).parent / SEGMENT_ANYTHING_2_REPO_PATH / "checkpoints" / MODEL_CHECKPOINT
    )
    video_predictor = build_sam2_video_predictor(MODEL_CONFIG, ckpt, device=DEVICE)
    image_model = video_predictor

    # Pre-warm torch.jit.script by creating one throwaway predictor on the
    # main thread. Subsequent SAM2ImagePredictor() calls reuse the cached
    # JIT compilation and are fast + thread-safe.
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2ImagePredictor(image_model)

    return image_model, video_predictor


_models: Optional[Tuple[Any, Any]] = None
_models_lock = threading.Lock()
_ACCELERATOR = threading.BoundedSemaphore(1)


@contextmanager
def _accelerator_slot():
    """Serialize GPU/MPS-heavy SAM2 calls across request and worker threads."""
    _ACCELERATOR.acquire()
    try:
        yield
    finally:
        _ACCELERATOR.release()


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


def _load_sam2_jpeg_as_tensor(img_path: str, image_size: int) -> Tuple[torch.Tensor, int, int]:
    """Load one extracted JPEG like SAM2's utility, but as float32."""
    with Image.open(img_path) as img_pil:
        video_width, video_height = img_pil.size
        img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype != np.uint8:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img_np = img_np.astype(np.float32) / 255.0
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    return img, video_height, video_width


class _OnDemandSam2FrameLoader:
    """Small LRU frame loader for SAM2 tracking.

    Upstream AsyncVideoFrameLoader eventually stores every resized frame tensor
    in memory. For tracking, SAM2 only needs the current image; temporal memory
    comes from maskmem outputs, not old RGB frames. This loader keeps a tiny LRU
    of normalized frame tensors and decodes the rest on demand.
    """

    def __init__(
        self,
        img_paths: List[str],
        image_size: int,
        offload_video_to_cpu: bool,
        img_mean: torch.Tensor,
        img_std: torch.Tensor,
        compute_device,
        cache_size: int,
    ):
        if not img_paths:
            raise RuntimeError("no images found for SAM2 tracking")
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        self.compute_device = compute_device
        self.cache_size = max(0, cache_size)
        self._cache = OrderedDict()
        self._closed = False
        self._lock = threading.RLock()
        with Image.open(img_paths[0]) as img_pil:
            self.video_width, self.video_height = img_pil.size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        with self._lock:
            if self._closed:
                raise RuntimeError("SAM2 frame loader is closed")
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached

        img, video_height, video_width = _load_sam2_jpeg_as_tensor(
            self.img_paths[index], self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)

        with self._lock:
            if self._closed:
                return img
            if self.cache_size > 0:
                self._cache[index] = img
                self._cache.move_to_end(index)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return img

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._cache.clear()


def _sam2_jpeg_paths(frame_dir: str) -> List[str]:
    names = [
        name for name in os.listdir(frame_dir)
        if os.path.splitext(name)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    names.sort(key=lambda name: int(os.path.splitext(name)[0]))
    return [os.path.join(frame_dir, name) for name in names]


def _init_tracking_inference_state(video_predictor, frame_dir: str, use_on_demand_frames: bool):
    inference_state = video_predictor.init_state(
        video_path=frame_dir,
        async_loading_frames=use_on_demand_frames or (DEVICE != "mps"),
        offload_video_to_cpu=True,
    )
    if not use_on_demand_frames:
        return inference_state

    old_images = inference_state.get("images")
    compute_device = video_predictor.device
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    images = _OnDemandSam2FrameLoader(
        _sam2_jpeg_paths(frame_dir),
        video_predictor.image_size,
        offload_video_to_cpu=True,
        img_mean=img_mean,
        img_std=img_std,
        compute_device=compute_device,
        cache_size=TRACKING_FRAME_CACHE_SIZE,
    )
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    inference_state["video_height"] = images.video_height
    inference_state["video_width"] = images.video_width
    inference_state["cached_features"] = {}
    _stop_sam2_frame_loader(old_images)
    return inference_state


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



class TrackingSession:
    """Holds state for an async tracking job. The background thread appends
    frame masks; the frontend polls for new ones via track_progress."""

    def __init__(self, session_id: str, total_frames: int, producers: int = 1, task_id: Optional[str] = None):
        self.session_id = session_id
        self.task_id = task_id
        self.total_frames = total_frames
        self.frames: queue.Queue[Dict[str, Any]] = queue.Queue(
            maxsize=max(1, TRACK_PROGRESS_MAX_BATCH * 4)
        )
        self.produced = 0
        self.created_at = time.time()
        self.last_access = self.created_at
        self.completed_at: Optional[float] = None
        self.done = False
        self.error: Optional[str] = None
        self.cancelled = False
        self.futures: List[Future] = []
        # Bidirectional tracking runs two producer threads sharing this
        # session; we only flip `done` after all of them report in.
        self._remaining_producers = max(1, producers)
        self.lock = threading.RLock()
        # Signaled whenever there's something new for a poller to see:
        # a frame was appended, the session finished, errored, or was cancelled.
        self.new_data_event = threading.Event()

    def append_frame(self, frame_data: Dict[str, Any]):
        while True:
            with self.lock:
                if self.cancelled:
                    return
            try:
                self.frames.put(frame_data, timeout=0.5)
                break
            except queue.Full:
                continue
        with self.lock:
            self.produced += 1
        self.new_data_event.set()

    def drain_new(self) -> Tuple[List[Dict[str, Any]], int, bool]:
        """Return (new_frames, total_produced, is_done), dropping drained payloads."""
        new: List[Dict[str, Any]] = []
        while len(new) < TRACK_PROGRESS_MAX_BATCH:
            try:
                new.append(self.frames.get_nowait())
            except queue.Empty:
                break
        with self.lock:
            self.last_access = time.time()
            done = self.done and self.frames.empty()
            return new, self.produced, done

    def finish(self):
        """Single-producer shorthand: mark the session done immediately."""
        with self.lock:
            self.done = True
            self._remaining_producers = 0
            self.completed_at = time.time()
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
                self.completed_at = time.time()
        self.new_data_event.set()

    def add_future(self, future: Future) -> None:
        with self.lock:
            if self.cancelled:
                future.cancel()
                return
            self.futures.append(future)

    def has_running_future(self) -> bool:
        with self.lock:
            return any(future.running() for future in self.futures)

    def cancel(self):
        with self.lock:
            self.cancelled = True
            self.done = True
            self._remaining_producers = 0
            self.completed_at = time.time()
            self.last_access = self.completed_at
            futures = list(self.futures)
        for future in futures:
            future.cancel()
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
_tracking_executor = ThreadPoolExecutor(
    max_workers=max(1, TRACKING_WORKERS),
    thread_name_prefix="sam2-track",
)
def _cleanup_tracking_sessions() -> None:
    now = time.time()
    expired: List[str] = []
    to_cancel: List[TrackingSession] = []
    with _tracking_lock:
        for session_id, session in list(_tracking_sessions.items()):
            with session.lock:
                idle_for = now - session.last_access
                age = now - session.created_at
                done_ttl = session.done and idle_for > TRACK_SESSION_TTL_SECONDS
                cancelled_ttl = session.cancelled and idle_for > TRACK_SESSION_TTL_SECONDS
                max_age = age > TRACK_SESSION_MAX_AGE_SECONDS
                if done_ttl or cancelled_ttl or max_age:
                    if max_age and not session.done:
                        to_cancel.append(session)
                    expired.append(session_id)
        for session_id in expired:
            _tracking_sessions.pop(session_id, None)
    for session in to_cancel:
        session.cancel()
    FRAME_CACHE.expire_idle()
    VIDEOS.expire_idle()
    if expired:
        logger.info("cleaned up %d tracking sessions", len(expired))


def _tracking_cleanup_loop() -> None:
    interval = max(30.0, min(TRACK_SESSION_TTL_SECONDS, 300.0))
    while True:
        time.sleep(interval)
        try:
            _cleanup_tracking_sessions()
        except Exception:
            logger.exception("tracking session cleanup failed")


threading.Thread(target=_tracking_cleanup_loop, name="tracking-session-cleanup", daemon=True).start()


def _stop_sam2_frame_loader(images: Any) -> None:
    close = getattr(images, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.debug("failed to close SAM2 frame loader", exc_info=True)
        return

    if hasattr(images, "images"):
        # Upstream AsyncVideoFrameLoader does not expose close(). Signal the
        # thread to stop, wait briefly, then clear loaded tensors.
        try:
            images.exception = RuntimeError("SAM2 frame loader closed")
        except Exception:
            pass
        thread = getattr(images, "thread", None)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        if thread is None or not thread.is_alive():
            try:
                images.images = []
            except Exception:
                logger.debug("failed to clear SAM2 async frame loader", exc_info=True)


def _release_sam2_inference_state(video_predictor: Any, inference_state: Optional[Dict[str, Any]]) -> None:
    """Drop all per-track SAM2 state, including async-loaded frame tensors.

    SAM2's video predictor keeps the resized video frames, cached backbone
    features, and per-frame memory outputs inside the caller-owned
    inference_state. In normal Python control flow those objects are released
    when _run_tracking returns, but explicit teardown matters here because the
    async frame loader can otherwise continue filling its tensor list after we
    have already stopped tracking.
    """
    if not inference_state:
        return

    try:
        if video_predictor is not None:
            with _accelerator_slot():
                video_predictor.reset_state(inference_state)
    except Exception:
        logger.debug("failed to reset SAM2 tracking state before release", exc_info=True)

    _stop_sam2_frame_loader(inference_state.get("images"))

    for key in (
        "images",
        "cached_features",
        "constants",
        "point_inputs_per_obj",
        "mask_inputs_per_obj",
        "output_dict_per_obj",
        "temp_output_dict_per_obj",
        "frames_tracked_per_obj",
    ):
        value = inference_state.get(key)
        if hasattr(value, "clear"):
            try:
                value.clear()
            except Exception:
                pass
        inference_state.pop(key, None)
    inference_state.clear()


def _release_tracking_allocator_memory() -> None:
    """Best-effort return of allocator-held memory after a large tracking run."""
    if not TRACKING_RELEASE_MEMORY:
        return

    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        logger.debug("failed to empty CUDA cache after tracking", exc_info=True)

    try:
        mps = getattr(torch, "mps", None)
        if mps is not None and getattr(mps, "is_available", lambda: False)():
            mps.empty_cache()
    except Exception:
        logger.debug("failed to empty MPS cache after tracking", exc_info=True)



def _release_task_runtime_state(task_id: Optional[str], reason: str) -> None:
    """Drop non-model runtime state associated with a task/video asset."""
    if not task_id:
        return
    cache_stats = FRAME_CACHE.stats()
    task_stats = cache_stats.get("per_task", {}).get(task_id, {})
    FRAME_CACHE.drop_task(task_id)
    VIDEOS.drop(task_id)
    logger.info(
        "released task runtime state task=%s reason=%s frame_cache_frames=%s frame_cache_bytes=%s",
        task_id,
        reason,
        task_stats.get("frames", 0),
        task_stats.get("bytes", 0),
    )
    _release_tracking_allocator_memory()


def _task_has_active_tracking_session(task_id: Optional[str]) -> bool:
    if not task_id:
        return False
    with _tracking_lock:
        return any(
            session.task_id == task_id and not session.done
            for session in _tracking_sessions.values()
        )


def _contiguous_runs(indices: List[int]) -> List[List[int]]:
    if not indices:
        return []
    runs: List[List[int]] = [[indices[0]]]
    for idx in indices[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs


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

        image_url = self._image_url_from_task(task, to_name)
        ls_host, ls_token = self._ls_host_token()
        local_path = self.get_local_path(
            image_url,
            task_id=task_id,
            ls_host=ls_host,
            ls_access_token=ls_token_for_sdk(ls_host, ls_token),
        )

        bgr = cv2.imread(local_path)
        if bgr is None:
            raise RuntimeError(f"failed to read image: {local_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        points_abs = (prompts["points"] * np.array([w, h], dtype=np.float32)) if prompts["points"] is not None else None
        box_abs = (prompts["box"] * np.array([w, h, w, h], dtype=np.float32)) if prompts["box"] is not None else None

        with _accelerator_slot():
            image_predictor = make_image_predictor()
            image_predictor.set_image(rgb)
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
            _release_task_runtime_state(task_id, reason="release")
            return PredictionValue(result=[{
                "value": {"status": "released"},
                "from_name": from_name, "to_name": to_name,
                "type": "release_ack", "origin": "manual",
                "id": str(uuid4())[:8],
            }])

        if event == "track_progress":
            return self._handle_track_progress(context, from_name, to_name)

        if event == "track_cancel":
            return self._handle_track_cancel(task_id, context, from_name, to_name)

        # Events below need video handle
        window = int(context.get("window", WINDOW_SIZE))
        direction = context.get("direction", "forward")

        raw_url = self._image_url_from_task(task, to_name)
        with VIDEOS.acquire(
            task_id,
            raw_url,
            lambda: self._open_video_handle(task_id, raw_url),
        ) as video:
            # Prefer `time` (seconds) — the only quantity FE and BE can agree on
            # without knowing each other's fps. Fall back to `frame` (FE 1-indexed)
            # for legacy callers that don't send time.
            frame = _resolve_be_frame(context, video)

            FRAME_CACHE.touch(task_id, frame)

            if event == "prewarm":
                frame_range = self._window_range(frame, window, direction, video.frame_count)
                # Reserve missing frames before any decode/network I/O so concurrent
                # prewarms share the same pending work instead of duplicating it.
                cached, pending = FRAME_CACHE.schedule_missing(
                    task_id,
                    frame_range,
                    lambda indices, _video=video: self._encode_frame_batch(_video, indices),
                    on_scheduled=lambda _video=video: VIDEOS.retain(_video),
                )
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

        h, w = video.height, video.width

        # Try to restore from frame cache (pre-encoded by prewarm)
        cached = FRAME_CACHE.get(task_id, frame)
        frame_rgb = None
        if cached is None or cached.get("features") is None:
            frame_bgr = video.read_frame(frame)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

        points_abs = (prompts["points"] * np.array([w, h], dtype=np.float32)) if prompts["points"] is not None else None
        box_abs = (prompts["box"] * np.array([w, h, w, h], dtype=np.float32)) if prompts["box"] is not None else None
        with _accelerator_slot():
            image_predictor = make_image_predictor()
            if cached is not None and cached.get("features") is not None:
                image_predictor._features = cached["features"]
                image_predictor._orig_hw = cached["original_size"]
                image_predictor._is_image_set = True
                logger.debug("predict: cache hit task=%s frame=%s", task_id, frame)
            else:
                image_predictor.set_image(frame_rgb)
                features_snapshot = {
                    "features": image_predictor._features,
                    "original_size": image_predictor._orig_hw,
                    "is_image_set": True,
                }
                FRAME_CACHE.submit(task_id, [frame], lambda idx: features_snapshot)
                logger.debug("predict: cache miss task=%s frame=%s, encoded + cached", task_id, frame)

            masks, scores, _ = image_predictor.predict(
                point_coords=points_abs,
                point_labels=prompts["labels"] if prompts["labels"] is not None else None,
                box=box_abs,
                multimask_output=True,
            )
        mask = _clean_mask(_pick_best_mask(masks, scores))
        return self._mask_response(mask, w, h, from_name, to_name)

    def _ls_host_token(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (ls_host, ls_token) for authenticated LS asset fetches.

        Host/token resolution is env first, then the cache populated from
        `/setup`. Either may be None; callers pass them into
        `self.get_local_path(ls_host=..., ls_access_token=...)` which lets the
        SDK skip its `http://localhost:8000` default fallback.

        LS's own `/setup` payload can carry `http://localhost:<port>` when the
        LS side doesn't have `HOSTNAME` configured — that's useless to a remote
        ML backend. Keep both host and token env-var-first so an operator can
        fix credentials through the process environment / `docker-compose` and
        restart the backend; otherwise an old `/setup` access token can shadow a
        freshly configured `LABEL_STUDIO_API_KEY` and cause 401s on uploaded
        files.
        """
        env_host = (os.getenv("LABEL_STUDIO_URL") or os.getenv("LABEL_STUDIO_HOST") or "").rstrip("/")
        env_token = (
            os.getenv("LABEL_STUDIO_API_KEY")
            or os.getenv("LABEL_STUDIO_ACCESS_TOKEN")
            or ""
        )
        cached_host = (LS_CONTEXT.get("url") or "").rstrip("/")
        cached_token = LS_CONTEXT.get("token") or ""
        host = env_host or cached_host
        token = env_token or cached_token
        return (host or None, token or None)

    def _open_video_handle(self, task_id: str, raw_url: str):
        source, headers = self._resolve_video_source(raw_url, task_id)
        try:
            return VIDEOS.open_handle(task_id, source, headers=headers, raw_url=raw_url)
        except Exception as e:
            logger.warning("video open failed (%s), falling back to SDK download", e)
            local_path = self._download_valid_video(raw_url, task_id)
            return VIDEOS.open_handle(task_id, local_path, raw_url=raw_url)

    def _download_valid_video(self, raw_url: str, task_id: str) -> str:
        """Download via the LS SDK, but verify the file actually decodes.

        `get_local_path` caches downloads by URL hash with no integrity check,
        so a truncated download of a large video gets reused forever and the
        decoder then dies with an opaque "failed to open video" (surfaced as a
        503). On a bad cache hit, drop the file and re-download once.
        """
        ls_host, ls_token = self._ls_host_token()
        sdk_token = ls_token_for_sdk(ls_host, ls_token)
        for attempt in range(2):
            local_path = self.get_local_path(
                raw_url, task_id=task_id, ls_host=ls_host, ls_access_token=sdk_token,
            )
            ok, reason = video_is_readable(local_path)
            if ok:
                return local_path
            logger.warning(
                "cached video unreadable (attempt %d/2), re-downloading: %s — %s",
                attempt + 1, local_path, reason,
            )
            try:
                os.remove(local_path)
            except OSError:
                pass
        raise RuntimeError(
            f"video failed to decode after re-download: {raw_url} ({reason})"
        )

    def _resolve_video_source(self, raw_url: str, task_id: str):
        """Resolve a task video URL to a streamable source + auth headers.

        * External HTTP(S) URLs (don't look like LS's own host) → stream
          directly, no headers.
        * LS-hosted URLs / paths → authenticated local download through the
          LS SDK. We deliberately do not pass LS Authorization headers to
          ffmpeg because it forwards custom headers across redirects.
        * Cloud-storage URIs → authenticated local download through the LS SDK.

        Resolution order for the LS hostname:
          1. `LABEL_STUDIO_URL` env var (explicit operator override)
          2. Cached value from the last `/setup` payload (LS is the source
             of truth when the env var isn't set — avoids the SDK's
             `http://localhost:8000` default).
        """
        ls_url_opt, api_key_opt = self._ls_host_token()
        ls_url = ls_url_opt or ""
        api_key = api_key_opt or ""

        if raw_url.startswith("http://") or raw_url.startswith("https://"):
            # Absolute URL — attach auth iff its host matches the known LS host.
            # Never attach to any other host: task data can carry external /
            # presigned cloud URLs and we must not leak the LS token to them.
            attach = should_attach_ls_auth(raw_url, ls_url, bool(api_key))
            if attach:
                logger.info(
                    "LS-hosted video: downloading once instead of streaming (%s)",
                    raw_url,
                )
                return self._download_valid_video(raw_url, task_id), None
            return raw_url, None

        if raw_url.startswith(CLOUD_URI_SCHEMES):
            # A cloud-storage URI is not a URL — only LS knows which storage it
            # belongs to and holds the credentials, so it can't be streamed and
            # must not be joined onto the LS host (that yields
            # `https://<ls-host>/s3://bucket/…`, which 404s). The SDK resolves
            # it through `/tasks/<id>/presign/` and downloads the result.
            return self._download_valid_video(raw_url, task_id), None

        if not ls_url:
            logger.warning(
                "streaming not available (no LABEL_STUDIO_URL), falling back to download"
            )
        else:
            logger.info("video source requires Label Studio resolution; downloading once")
        local_path = self._download_valid_video(raw_url, task_id)
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
        _cleanup_tracking_sessions()

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
        session = TrackingSession(session_id, total, producers=len(ranges), task_id=task_id)

        with _tracking_lock:
            _tracking_sessions[session_id] = session

        for d, start_frame, end_frame in ranges:
            release_video = VIDEOS.retain(video)
            future = _tracking_executor.submit(
                self._run_tracking,
                session, video, release_video, prompts, start_frame, end_frame,
                prompt_frame, max_frames, w, h, d,
            )
            future.add_done_callback(
                lambda done_future, release=release_video: release()
                if done_future.cancelled() else None
            )
            session.add_future(future)

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

    def _run_tracking(self, session, video, release_video, prompts, start_frame, end_frame,
                      prompt_frame, max_frames, w, h, direction="forward"):
        """Background thread: extract frames, run SAM2 propagation, push
        mask PNGs into the session as they're produced."""
        video_predictor = None
        inference_state = None
        propagation = None
        out_mask_logits = None
        mask = None
        try:
            if session.cancelled:
                logger.info("track bg: skipped cancelled session=%s", session.session_id)
                return

            video_predictor = get_video_predictor()
            frame_count_needed = end_frame - start_frame

            with tempfile.TemporaryDirectory() as frame_dir:
                if session.cancelled:
                    logger.info("track bg: skipped cancelled session=%s", session.session_id)
                    return

                written = video.write_frame_range_as_jpegs(
                    start_frame, frame_count_needed, frame_dir)
                logger.info("track bg: extracted %d frames to %s", written, frame_dir)

                if session.cancelled:
                    logger.info("track bg: cancelled after extraction session=%s", session.session_id)
                    return

                # Use a small on-demand JPEG loader by default. Upstream SAM2's
                # async loader eventually stores every resized frame tensor in
                # RAM; for long tracking ranges that alone can be multi-GB.
                # Tracking only needs the current image because temporal state
                # lives in maskmem outputs, so a tiny LRU is enough.
                with _accelerator_slot():
                    inference_state = _init_tracking_inference_state(
                        video_predictor,
                        frame_dir,
                        use_on_demand_frames=TRACKING_ON_DEMAND_FRAMES,
                    )

                if session.cancelled:
                    logger.info("track bg: cancelled after SAM2 init session=%s", session.session_id)
                    return

                relative_prompt_frame = prompt_frame - start_frame
                with _accelerator_slot():
                    video_predictor.reset_state(inference_state)
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

                if session.cancelled:
                    logger.info("track bg: cancelled before propagation session=%s", session.session_id)
                    return

                total_pixels = w * h
                consecutive_lost = 0

                propagation = video_predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=relative_prompt_frame,
                    max_frame_num_to_track=max_frames,
                    reverse=(direction == "backward"),
                )
                while True:
                    if session.cancelled:
                        logger.info("track bg: cancelled session=%s", session.session_id)
                        break
                    try:
                        with _accelerator_slot():
                            out_frame_idx, out_obj_ids, out_mask_logits = next(propagation)
                    except StopIteration:
                        break

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
                                mask = None
                                break
                            # Skip emitting this low-confidence frame; keep propagating.
                            mask = None
                            continue

                        consecutive_lost = 0

                        fg_ratio = fg_count / total_pixels
                        if fg_ratio > MAX_FOREGROUND_RATIO:
                            logger.info(
                                "track bg: foreground ratio %.2f > %.2f at frame %d, stopping",
                                fg_ratio, MAX_FOREGROUND_RATIO, real_frame_idx,
                            )
                            stop = True
                            mask = None
                            break

                        image_data_url = f"data:image/png;base64,{mask_to_bitmap_png_base64(mask)}"
                        session.append_frame({
                            "frame": real_frame_idx,
                            # Emit time (ms) so the FE can recompute its own
                            # frame index without knowing the BE's fps —
                            # avoids frame drift when FE and BE see different
                            # fps for the same video.
                            "time_ms": (real_frame_idx * 1000.0 / video.fps) if video.fps else 0.0,
                            "imageDataURL": image_data_url,
                            "width": int(w),
                            "height": int(h),
                        })
                        mask = None

                    out_mask_logits = None
                    if stop:
                        logger.info("track bg: stopped at frame %d (produced %d frames)",
                                    real_frame_idx, session.produced)
                        break

        except Exception as e:
            logger.exception("track bg: error session=%s direction=%s", session.session_id, direction)
            session.error = str(e)
        finally:
            if propagation is not None:
                try:
                    propagation.close()
                except Exception:
                    logger.debug("failed to close SAM2 propagation iterator", exc_info=True)
            propagation = None
            out_mask_logits = None
            mask = None
            _release_sam2_inference_state(video_predictor, inference_state)
            inference_state = None
            video_predictor = None
            _release_tracking_allocator_memory()
            session.producer_done()
            release_video()
            logger.info("track bg: finished session=%s direction=%s frames=%d",
                         session.session_id, direction, session.produced)

    def _handle_track_progress(self, context, from_name, to_name):
        _cleanup_tracking_sessions()
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

    def _handle_track_cancel(self, task_id, context, from_name, to_name):
        session_id = context.get("session_id", "")
        with _tracking_lock:
            session = _tracking_sessions.pop(session_id, None)
        if session:
            session.cancel()
            # If a worker is already inside video extraction/propagation, let
            # its finally block close the task state. Closing the VideoHandle
            # here can race cv2/ffmpeg fallback reads.
            release_task_id = session.task_id or task_id
            if not session.has_running_future() and not _task_has_active_tracking_session(release_task_id):
                _release_task_runtime_state(release_task_id, reason="track_cancel")
        return PredictionValue(result=[{
            "id": str(uuid4())[:8],
            "from_name": from_name, "to_name": to_name,
            "type": "track_cancel_ack", "origin": "manual",
            "value": {"status": "cancelled"},
        }])

    # --- helpers ---------------------------------------------------------

    def _encode_frame(self, task_id: str, frame_idx: int):
        """Decode + encode a single video frame into a SAM2 image embedding."""
        video = VIDEOS.get(task_id)
        if video is None:
            raise RuntimeError(f"no video handle for task {task_id}")
        return self._encode_bgr(video.read_frame(frame_idx))

    def _encode_bgr(self, frame_bgr):
        """Encode an already-decoded BGR frame into a SAM2 image embedding."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with _accelerator_slot():
            image_predictor = make_image_predictor()
            image_predictor.set_image(rgb)
            return {
                "features": image_predictor._features,
                "original_size": image_predictor._orig_hw,
                "is_image_set": True,
            }

    def _encode_frame_batch(self, video, frame_indices: List[int]) -> Dict[int, Any]:
        """Decode sparse frame indices in contiguous runs, then encode them."""
        encoded: Dict[int, Any] = {}
        for run in _contiguous_runs(sorted(set(frame_indices))):
            start = run[0]
            try:
                frames = video.read_frame_range(start, len(run))
            except Exception as e:
                logger.warning("batch frame read failed (%s); falling back to per-frame", e)
                frames = [video.read_frame(idx) for idx in run]
            for idx, frame_bgr in zip(run, frames):
                encoded[idx] = self._encode_bgr(frame_bgr)
        return encoded

    def _window_range(self, frame: int, window: int, direction: str, frame_count: int):
        if direction == "backward":
            start = max(0, frame - window)
            end = frame + 1
        else:
            start = frame
            end = min(frame_count, frame + window + 1)
        return list(range(start, end))
