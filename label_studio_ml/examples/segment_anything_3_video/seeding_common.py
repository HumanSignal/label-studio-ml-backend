"""
Shared helpers for SAM3 video seeding CLI tools.

Provides:
- Label Studio I/O (client, fetch, upload, video path resolution)
- SAM3 image embeddings for keyframe detection (replaces SAM2)
- Text-based object detection via Sam3VideoModel (replaces Grounding DINO)
- Coordinate conversion utilities
- Change-detection math (cosine distance, smoothing, keyframe selection)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import av
import numpy as np
import requests
import torch
from joblib import Memory
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.client import LabelStudio
from PIL import Image

logger = logging.getLogger(__name__)


class InitialSeedingError(Exception):
    pass


@dataclass
class KeyframeDetection:
    frame_idx: int
    xyxy: np.ndarray
    score: float
    label: str
    track_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = os.getenv('MODEL_NAME', 'facebook/sam3')
DTYPE = torch.bfloat16 if DEVICE == 'cuda' else torch.float32


# ---------------------------------------------------------------------------
# Lazy-loaded SAM3 model singletons
# ---------------------------------------------------------------------------

_sam3_image_model = None
_sam3_image_processor = None
_sam3_video_model = None
_sam3_video_processor = None
_sam3_tracker_model = None
_sam3_tracker_processor = None


def _get_sam3_image_model():
    """Lazy-load Sam3Model for image embeddings (keyframe detection, teacher confirmation)."""
    global _sam3_image_model, _sam3_image_processor
    if _sam3_image_model is None:
        from transformers import Sam3Model, Sam3Processor
        logger.info("Loading Sam3Model (image) from %s ...", MODEL_NAME)
        _sam3_image_model = Sam3Model.from_pretrained(MODEL_NAME).to(DEVICE, dtype=DTYPE)
        _sam3_image_processor = Sam3Processor.from_pretrained(MODEL_NAME)
        logger.info("Sam3Model loaded on %s", DEVICE)
    return _sam3_image_model, _sam3_image_processor


def _get_sam3_video_model():
    """Lazy-load Sam3VideoModel for text-based detection (replaces Grounding DINO)."""
    global _sam3_video_model, _sam3_video_processor
    if _sam3_video_model is None:
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        logger.info("Loading Sam3VideoModel (PCS) from %s ...", MODEL_NAME)
        _sam3_video_model = Sam3VideoModel.from_pretrained(MODEL_NAME).to(DEVICE, dtype=DTYPE)
        _sam3_video_processor = Sam3VideoProcessor.from_pretrained(MODEL_NAME)
        logger.info("Sam3VideoModel loaded on %s", DEVICE)
    return _sam3_video_model, _sam3_video_processor


def _get_sam3_tracker_model():
    """Lazy-load Sam3TrackerVideoModel for box-prompted tracking."""
    global _sam3_tracker_model, _sam3_tracker_processor
    if _sam3_tracker_model is None:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        logger.info("Loading Sam3TrackerVideoModel from %s ...", MODEL_NAME)
        _sam3_tracker_model = Sam3TrackerVideoModel.from_pretrained(MODEL_NAME).to(DEVICE, dtype=DTYPE)
        _sam3_tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(MODEL_NAME)
        logger.info("Sam3TrackerVideoModel loaded on %s", DEVICE)
    return _sam3_tracker_model, _sam3_tracker_processor


# ---------------------------------------------------------------------------
# Text+Box Refinement (Hybrid approach for imperfect boxes)
# ---------------------------------------------------------------------------

def refine_box_with_text_prompt(
    image: Image.Image,
    box_xyxy: np.ndarray,
    text_label: str,
    search_scale: float = 1.3,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Refine a bounding box using Sam3Model with combined text+box prompts.

    This hybrid approach handles boxes that are too large OR too small:
    - The expanded box (search region) ensures the target is likely contained
    - The text prompt tells SAM3 WHAT to segment (e.g., "person")

    Args:
        image: PIL Image containing the frame
        box_xyxy: Original bounding box in xyxy pixel coordinates
        text_label: Text prompt describing the object (e.g., "person", "Player")
        search_scale: Factor to expand the search region (default 1.3 = 30% larger)
        threshold: Confidence threshold for accepting refinement (default 0.5)

    Returns:
        Tuple of (refined_box_xyxy, confidence_score)
        If refinement fails, returns (original_box, 0.0)
    """
    sam3_model, sam3_processor = _get_sam3_image_model()

    w, h = image.size
    x0, y0, x1, y1 = box_xyxy

    # Compute expanded search region centered on original box
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = x1 - x0
    bh = y1 - y0

    half_w = 0.5 * search_scale * bw
    half_h = 0.5 * search_scale * bh

    # Clamp to image bounds
    x0_search = max(0.0, cx - half_w)
    y0_search = max(0.0, cy - half_h)
    x1_search = min(float(w), cx + half_w)
    y1_search = min(float(h), cy + half_h)

    search_box = [int(round(x0_search)), int(round(y0_search)),
                  int(round(x1_search)), int(round(y1_search))]

    if search_box[2] <= search_box[0] or search_box[3] <= search_box[1]:
        logger.debug("Search box invalid after clamping, returning original")
        return box_xyxy.copy(), 0.0

    try:
        # Run SAM3 with combined text + positive box prompt
        inputs = sam3_processor(
            images=image,
            text=text_label,
            input_boxes=[[search_box]],
            input_boxes_labels=[[1]],  # 1 = positive box
            return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
            outputs = sam3_model(**inputs)

        # Post-process using instance segmentation
        results = sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results.get("masks", [])
        scores = results.get("scores", [])
        boxes = results.get("boxes", [])

        if len(masks) == 0:
            logger.debug("No masks returned for text='%s', returning original box", text_label)
            return box_xyxy.copy(), 0.0

        # Select best mask by confidence score
        if len(scores) > 0:
            score_vals = [s.item() if hasattr(s, 'item') else float(s) for s in scores]
            best_idx = int(np.argmax(score_vals))
            best_score = score_vals[best_idx]
        else:
            best_idx = 0
            best_score = 0.5

        # Extract refined box
        if len(boxes) > best_idx:
            best_box = boxes[best_idx]
            if hasattr(best_box, 'tolist'):
                best_box = best_box.tolist()
            elif hasattr(best_box, 'cpu'):
                best_box = best_box.cpu().numpy().tolist()
            refined_xyxy = np.array(best_box, dtype=np.float32)
        else:
            # Fall back to extracting bbox from mask
            mask = masks[best_idx]
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()

            ys, xs = np.where(mask > 0)
            if xs.size == 0 or ys.size == 0:
                return box_xyxy.copy(), 0.0

            refined_xyxy = np.array([
                int(xs.min()), int(ys.min()),
                int(xs.max()) + 1, int(ys.max()) + 1
            ], dtype=np.float32)

        # Clamp to image bounds
        refined_xyxy[0] = max(0, min(w - 1, refined_xyxy[0]))
        refined_xyxy[1] = max(0, min(h - 1, refined_xyxy[1]))
        refined_xyxy[2] = max(0, min(w, refined_xyxy[2]))
        refined_xyxy[3] = max(0, min(h, refined_xyxy[3]))

        if refined_xyxy[2] <= refined_xyxy[0] or refined_xyxy[3] <= refined_xyxy[1]:
            return box_xyxy.copy(), 0.0

        logger.debug(
            "Refined box: [%.1f,%.1f,%.1f,%.1f] -> [%.1f,%.1f,%.1f,%.1f] (score=%.3f)",
            x0, y0, x1, y1,
            refined_xyxy[0], refined_xyxy[1], refined_xyxy[2], refined_xyxy[3],
            best_score,
        )
        return refined_xyxy, best_score

    except Exception as exc:
        logger.warning("Text+box refinement failed: %s", exc)
        return box_xyxy.copy(), 0.0


def refine_boxes_batch(
    image: Image.Image,
    boxes_xyxy: List[np.ndarray],
    text_labels: List[str],
    search_scale: float = 1.3,
    threshold: float = 0.5,
) -> List[Tuple[np.ndarray, float]]:
    """Refine multiple boxes on the same frame using text+box prompts.

    Processes boxes one at a time (batching with different text prompts per box
    is complex with the current API).

    Args:
        image: PIL Image containing the frame
        boxes_xyxy: List of bounding boxes in xyxy pixel coordinates
        text_labels: List of text prompts (one per box)
        search_scale: Factor to expand search regions
        threshold: Confidence threshold

    Returns:
        List of (refined_box_xyxy, confidence_score) tuples
    """
    results = []
    for box, label in zip(boxes_xyxy, text_labels):
        refined, score = refine_box_with_text_prompt(
            image, box, label, search_scale, threshold
        )
        results.append((refined, score))
    return results


# ---------------------------------------------------------------------------
# LS I/O helpers (preserved from SAM2 seeding_common)
# ---------------------------------------------------------------------------

def _ensure_meta_text_placeholder(result: Dict[str, Any]) -> None:
    meta = result.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        result["meta"] = meta
    raw_text = meta.get("text")
    texts: List[str] = []
    if isinstance(raw_text, str):
        texts = [raw_text]
    elif isinstance(raw_text, list):
        texts = [t for t in raw_text if isinstance(t, str)]
    if not texts or all(not t.strip() for t in texts):
        meta["text"] = "id:"


def _build_ls_client(ls_url: str, ls_api_key: str):
    if not ls_api_key or ls_api_key.strip() == "" or ls_api_key == "your_api_key":
        raise InitialSeedingError(
            "LABEL_STUDIO_API_KEY is required. "
            "Provide it via --ls-api-key or the LABEL_STUDIO_API_KEY env var."
        )

    os.environ.setdefault("LABEL_STUDIO_URL", ls_url)
    os.environ.setdefault("LABEL_STUDIO_API_KEY", ls_api_key)

    logger.info("Connecting to Label Studio at %s", ls_url)
    client = LabelStudio(base_url=ls_url, api_key=ls_api_key, timeout=600)
    logger.info("Connected to Label Studio")
    return client


def _fetch_task(ls, project_id: int, task_id: int) -> Dict[str, Any]:
    logger.info("Fetching task %s from project %s", task_id, project_id)
    task_obj = ls.tasks.get(task_id)
    if not task_obj:
        raise InitialSeedingError(f"Task {task_id} not found")

    task_project = getattr(task_obj, "project", None)
    if task_project is not None and task_project != project_id:
        logger.warning(
            "Task %s belongs to project %s (not %s)",
            getattr(task_obj, "id", task_id),
            task_project,
            project_id,
        )

    task = {"id": task_obj.id, "data": getattr(task_obj, "data", {})}
    logger.info("Task fetched: %s", task.get("id"))
    return task


def _fetch_annotation(ls, annotation_id: int) -> Any:
    logger.info("Fetching annotation %s", annotation_id)
    ann = ls.annotations.get(id=annotation_id)
    if not ann:
        raise InitialSeedingError(f"Annotation {annotation_id} not found")

    result = getattr(ann, "result", None)
    if result is None:
        raise InitialSeedingError(f"Annotation {annotation_id} has no regions")

    logger.info(
        "Annotation fetched: id=%s with %d regions", getattr(ann, "id", annotation_id), len(result or [])
    )
    return ann


def _detect_video_key(task_data: Dict[str, Any]) -> Tuple[str, str]:
    preferred_keys = ["video", "video_url", "video_path"]
    for key in preferred_keys:
        if key in task_data and isinstance(task_data[key], str):
            return key, task_data[key]

    for key, value in task_data.items():
        if not isinstance(value, str):
            continue
        lower = value.lower()
        if lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            return key, value

    raise InitialSeedingError(
        "Could not detect video field in task data. "
        "Ensure your task has a field like 'video' with a video URL/path."
    )


def _manual_download_video(url: str, dest_path: str) -> None:
    """Manually download video with Authorization header if needed."""
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Token {api_key}"

    logger.info("Starting manual download from %s to %s", url, dest_path)
    try:
        with requests.get(url, headers=headers, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Manual download completed")
    except Exception as e:
        logger.error("Manual download failed: %s", e)
        # Clean up partial file
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise


def _get_video_path(task: Dict[str, Any]) -> Tuple[str, str]:
    data = task.get("data") or {}
    key, video_url = _detect_video_key(data)
    logger.info("Using video field '%s' with URL %s", key, video_url)

    if not video_url.startswith("http") and video_url.startswith("/"):
        host = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL")
        if host:
            from urllib.parse import urljoin

            video_url = urljoin(host.rstrip("/"), video_url)
            logger.info("Resolved relative video URL to %s", video_url)

    logger.info("Downloading/caching video via get_local_path...")
    local_path = get_local_path(video_url, task_id=task["id"])

    # Check for empty or missing file
    if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
        logger.warning("Cached video file is empty (0 bytes). Removing and attempting manual download...")
        try:
            os.remove(local_path)
        except OSError:
            pass

        try:
            _manual_download_video(video_url, local_path)
        except Exception:
            pass

    if not os.path.exists(local_path):
        raise InitialSeedingError(f"Video file not found after download: {local_path}")

    size_mb = os.path.getsize(local_path) / 1024**2
    logger.info("Video cached at: %s (%.2f MB)", local_path, size_mb)

    if size_mb == 0:
        raise InitialSeedingError(f"Video file is empty (0 bytes) after download attempts: {local_path}")

    return local_path, key


# ---------------------------------------------------------------------------
# PyAV video utilities (replaces cv2.VideoCapture)
# ---------------------------------------------------------------------------

def _get_video_info_pyav(video_path: str) -> Tuple[int, int, int, float]:
    """Get (width, height, frames_count, fps) via PyAV."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        frames_count = stream.frames
        if not frames_count and stream.duration and stream.time_base:
            frames_count = int(float(stream.duration * stream.time_base) * fps)
        return width, height, max(frames_count, 0), fps
    finally:
        container.close()


def _read_frame_pyav(video_path: str, frame_idx: int) -> Optional[Image.Image]:
    """Read a single frame by index via PyAV, return as PIL RGB Image."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]

        # Seek near target frame
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        if frame_idx > 0 and stream.time_base:
            target_ts = int(frame_idx / fps / stream.time_base)
            container.seek(target_ts, stream=stream)

        current_idx = 0
        for frame in container.decode(video=0):
            if current_idx >= frame_idx:
                pil_img = frame.to_image()
                return pil_img
            current_idx += 1

        return None
    finally:
        container.close()


def _decode_frames_pyav(
    video_path: str,
    start_frame: int,
    end_frame: int,
    stride: int = 1,
) -> List[Tuple[int, Image.Image]]:
    """Decode a range of frames to [(frame_idx, PIL.Image)] via PyAV."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]

        if start_frame > 0 and stream.average_rate and stream.time_base:
            avg_fps = float(stream.average_rate)
            target_ts = int(start_frame / avg_fps / stream.time_base)
            container.seek(target_ts, stream=stream)

        results = []
        frame_idx = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame_idx < start_frame:
                    frame_idx += 1
                    continue
                if frame_idx >= end_frame:
                    return results
                if stride > 1 and (frame_idx - start_frame) % stride != 0:
                    frame_idx += 1
                    continue
                results.append((frame_idx, frame.to_image()))
                frame_idx += 1

        return results
    finally:
        container.close()


# ---------------------------------------------------------------------------
# SAM3 image embeddings (replaces SAM2 embedding pipeline)
# ---------------------------------------------------------------------------

def _global_pool_embed(embed: torch.Tensor) -> torch.Tensor:
    if embed.ndim == 4:
        return embed.mean(dim=[2, 3])
    if embed.ndim == 3:
        return embed.mean(dim=[1, 2])
    return embed


def _extract_sam3_image_embedding(
    sam3_model,
    sam3_processor,
    pil_image: Image.Image,
) -> torch.Tensor:
    """Extract image embedding from Sam3Model's vision encoder."""
    inputs = sam3_processor(images=pil_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
        outputs = sam3_model.get_image_embeddings(inputs.pixel_values)
    return _global_pool_embed(outputs)


def _embed_batch_sam3(
    sam3_model,
    sam3_processor,
    frames: List[Image.Image],
) -> np.ndarray:
    """Embed a batch of PIL frames using Sam3Model."""
    out: List[np.ndarray] = []
    for frame in frames:
        embed = _extract_sam3_image_embedding(sam3_model, sam3_processor, frame)
        out.append(embed.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _compute_sam3_frame_embeddings(
    video_id: str,
    video_path: str,
    batch_size: int,
    cache_dir: str,
) -> np.ndarray:
    """Compute per-frame SAM3 image embeddings with joblib caching and PyAV decode."""
    memory = Memory(cache_dir, verbose=0)

    @memory.cache(ignore=["video_path_arg", "batch_size_arg"])
    def _cached_compute(video_id_key: str, video_path_arg: str, batch_size_arg: int) -> np.ndarray:
        from tqdm import tqdm

        sam3_model, sam3_processor = _get_sam3_image_model()

        container = av.open(video_path_arg)
        stream = container.streams.video[0]
        total_frames = stream.frames
        if not total_frames and stream.duration and stream.time_base:
            fps_est = float(stream.average_rate) if stream.average_rate else 30.0
            total_frames = int(float(stream.duration * stream.time_base) * fps_est)

        # Suppress logging during batch embedding
        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_handler_levels = [(h, h.level) for h in root_logger.handlers]
        root_logger.setLevel(logging.WARNING)
        for h in root_logger.handlers:
            h.setLevel(logging.WARNING)

        try:
            embeds: List[np.ndarray] = []
            frames_batch: List[Image.Image] = []

            with tqdm(total=total_frames, desc="Embedding frames", unit="frame") as pbar:
                for frame in container.decode(video=0):
                    pil_img = frame.to_image()
                    frames_batch.append(pil_img)

                    if len(frames_batch) >= batch_size_arg:
                        embeds.append(_embed_batch_sam3(sam3_model, sam3_processor, frames_batch))
                        pbar.update(len(frames_batch))
                        frames_batch = []

                if frames_batch:
                    embeds.append(_embed_batch_sam3(sam3_model, sam3_processor, frames_batch))
                    pbar.update(len(frames_batch))

            if not embeds:
                raise InitialSeedingError("No frames read from video for embedding computation")

            stacked = np.concatenate(embeds, axis=0).astype("float16")
            logger.info(
                "Computed SAM3 embeddings for %d frames (shape=%s)",
                stacked.shape[0],
                stacked.shape,
            )
            return stacked
        finally:
            container.close()
            root_logger.setLevel(original_level)
            for h, lvl in original_handler_levels:
                h.setLevel(lvl)

    return _cached_compute(video_id, video_path, batch_size)


# ---------------------------------------------------------------------------
# Change detection & keyframe selection (preserved, pure numpy)
# ---------------------------------------------------------------------------

def compute_change_scores(embeds: np.ndarray) -> np.ndarray:
    if embeds.ndim != 2:
        raise InitialSeedingError(f"Expected embeddings with shape [T, D], got {embeds.shape}")
    norm = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8
    norm_embeds = embeds / norm
    T_len = norm_embeds.shape[0]
    diff = np.zeros(T_len, dtype=np.float32)
    diff[1:] = np.linalg.norm(norm_embeds[1:] - norm_embeds[:-1], axis=1)
    return diff


def _median_filter_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size < 1:
        raise InitialSeedingError(f"kernel_size must be positive, got {kernel_size}")
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    try:
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
        return np.median(windows, axis=-1).astype(values.dtype, copy=False)
    except AttributeError:
        filtered = np.empty_like(values)
        for i in range(len(values)):
            start = i
            end = i + kernel_size
            filtered[i] = np.median(padded[start:end])
        return filtered


def smooth_change_scores(diff: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return _median_filter_1d(diff, kernel_size=kernel_size)


def uniform_indices(T_len: int, K: int) -> List[int]:
    return sorted({int(round(i * T_len / K)) for i in range(max(K, 1))})


def top_change_indices(smooth_diff: np.ndarray, max_candidates: int, min_spacing: int) -> List[int]:
    idx_sorted = np.argsort(-smooth_diff)
    chosen: List[int] = []
    for idx in idx_sorted:
        if len(chosen) >= max_candidates:
            break
        if all(abs(idx - c) >= min_spacing for c in chosen):
            chosen.append(int(idx))
    return sorted(chosen)


def select_keyframes(T_len: int, frac: float, smooth_diff: np.ndarray, min_spacing: int = 30) -> List[int]:
    K = max(1, int(frac * T_len))
    base = set(uniform_indices(T_len, K))
    changed = set(top_change_indices(smooth_diff, max_candidates=3 * K, min_spacing=min_spacing))
    merged = sorted(base.union(changed))
    if len(merged) > K:
        idx = np.linspace(0, len(merged) - 1, num=K, dtype=int)
        merged = [merged[i] for i in idx]
    return sorted(set(merged))


def _detect_keyframes(
    video_path: str,
    cache_dir: str,
    cache_key: str,
    embedding_batch: int,
    keyframe_frac: float,
    min_spacing: int,
) -> Tuple[List[int], int, int, int, float]:
    width, height, frames_count, fps = _get_video_info_pyav(video_path)

    embeds = _compute_sam3_frame_embeddings(cache_key, video_path, embedding_batch, cache_dir)
    if embeds.shape[0] != frames_count:
        logger.warning(
            "Embedding frame count (%d) does not match video frames (%d); proceeding with min length",
            embeds.shape[0],
            frames_count,
        )
        frames_count = min(frames_count, embeds.shape[0])
        embeds = embeds[:frames_count]

    diff = compute_change_scores(embeds)
    smooth = smooth_change_scores(diff, kernel_size=5)
    keyframes = select_keyframes(frames_count, keyframe_frac, smooth, min_spacing=min_spacing)
    logger.info("Selected %d keyframes out of %d total frames", len(keyframes), frames_count)
    return keyframes, width, height, frames_count, fps


# ---------------------------------------------------------------------------
# Sam3TextDetector (replaces GroundingDINOHelper)
# ---------------------------------------------------------------------------

class Sam3TextDetector:
    """Text-prompted object detection using Sam3VideoModel.

    Replaces GroundingDINOHelper. Uses SAM3's built-in text-based detection
    to find objects in individual frames.
    """

    def __init__(self):
        self.model, self.processor = _get_sam3_video_model()
        self.prompt = self._resolve_prompt()

    @staticmethod
    def _resolve_prompt() -> str:
        """Resolve text prompt from env vars."""
        prompt = os.getenv('PROMPT_TEXT', '')
        if not prompt:
            # Backward compat: fall back to GDINO env vars during transition
            prompt = os.getenv('GROUNDING_DINO_LABELS', 'person')
        return prompt.strip()

    def infer_frame(
        self,
        frame: Image.Image,
        *,
        prompt: Optional[str] = None,
    ) -> List[KeyframeDetection]:
        """Detect objects in a single frame using text prompt.

        Returns list of KeyframeDetection with xyxy pixel coords and scores.
        """
        prompt_text = prompt or self.prompt

        # Single-frame inference via Sam3VideoModel
        session = self.processor.init_video_session(
            inference_device=DEVICE, dtype=DTYPE
        )
        session = self.processor.add_text_prompt(
            inference_session=session, text=prompt_text
        )

        inputs = self.processor(images=frame, device=DEVICE, return_tensors="pt")
        with torch.no_grad():
            output = self.model(
                inference_session=session,
                frame=inputs.pixel_values[0],
                reverse=False,
            )

        processed = self.processor.postprocess_outputs(
            session, output, original_sizes=inputs.original_sizes
        )

        detections: List[KeyframeDetection] = []
        obj_ids = processed.get('object_ids', [])
        scores = processed.get('scores', [])
        boxes = processed.get('boxes', [])  # xyxy pixel coords

        for i in range(len(obj_ids)):
            box = boxes[i] if i < len(boxes) else None
            score = float(scores[i]) if i < len(scores) else 0.0
            if box is None:
                continue
            if hasattr(box, 'tolist'):
                box = box.tolist()
            xyxy = np.array(box, dtype=np.float32)
            detections.append(
                KeyframeDetection(
                    frame_idx=-1,  # set by caller
                    xyxy=xyxy,
                    score=score,
                    label=prompt_text,
                )
            )

        return detections


def _run_text_detection_on_keyframes(
    video_path: str,
    keyframes: List[int],
    prompt: Optional[str],
) -> Dict[int, List[KeyframeDetection]]:
    """Run Sam3VideoModel text detection on selected keyframes (replaces GDINO)."""
    from tqdm import tqdm

    detector = Sam3TextDetector()
    detections: Dict[int, List[KeyframeDetection]] = {}

    for frame_idx in tqdm(keyframes, desc="Text detection on keyframes", unit="kf"):
        pil_frame = _read_frame_pyav(video_path, frame_idx)
        if pil_frame is None:
            logger.warning("Failed to read keyframe %d", frame_idx)
            continue
        dets = detector.infer_frame(pil_frame, prompt=prompt)
        for d in dets:
            d.frame_idx = frame_idx
        detections[frame_idx] = dets

    return detections


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def xyxy_to_percent(xyxy: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
    # Guard against division by zero
    if width <= 0 or height <= 0:
        logger.warning("Invalid image dimensions (width=%s, height=%s), returning zeros", width, height)
        return (0.0, 0.0, 0.0, 0.0)
    x0, y0, x1, y1 = xyxy
    x0 = max(0.0, min(float(width - 1), float(x0)))
    y0 = max(0.0, min(float(height - 1), float(y0)))
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    return (x0 / width) * 100.0, (y0 / height) * 100.0, (w / width) * 100.0, (h / height) * 100.0


def _percent_xywh_to_xyxy_px(
    x_pct: float, y_pct: float, w_pct: float, h_pct: float,
    img_w: int, img_h: int,
) -> np.ndarray:
    """Convert LS percent coords to pixel xyxy. Canonical pattern."""
    x1 = (x_pct / 100.0) * img_w
    y1 = (y_pct / 100.0) * img_h
    x2 = x1 + (w_pct / 100.0) * img_w
    y2 = y1 + (h_pct / 100.0) * img_h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ---------------------------------------------------------------------------
# Prediction building & upload
# ---------------------------------------------------------------------------

def _build_prediction(
    tracks: List[Dict[str, Any]],
    width: int,
    height: int,
    frames_count: int,
    fps: float,
) -> Dict[str, Any]:
    duration = frames_count / fps if fps > 0 else 0.0
    results: List[Dict[str, Any]] = []
    for tr in tracks:
        seq_items = []
        for item in tr["sequence"]:
            x_pct, y_pct, w_pct, h_pct = xyxy_to_percent(item["xyxy"], width, height)
            frame_num = int(item["frame"])
            seq_items.append(
                {
                    "frame": frame_num,
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                    "enabled": item.get("enabled", True),
                    "rotation": 0,
                    "time": (frame_num - 1) / fps if fps > 0 else 0.0,
                }
            )

        if not seq_items:
            continue

        results.append(
            {
                "id": f"auto-track-{tr['track_id']}",
                "type": "videorectangle",
                "from_name": "box",
                "to_name": "video",
                "score": 1.0,
                "origin": "manual",
                "value": {
                    "sequence": seq_items,
                    "framesCount": frames_count,
                    "duration": duration,
                    "labels": [tr.get("label") or "object"],
                },
                "meta": {"text": "id:"},
            }
        )
        _ensure_meta_text_placeholder(results[-1])

    prediction = {"result": results, "score": 1.0, "model_version": "sam3-init-seed"}
    return prediction


def _upload_prediction(ls, task_id: int, prediction: Dict[str, Any]):
    try:
        result = ls.predictions.create(
            task=task_id,
            score=prediction.get("score", 0.0),
            model_version=prediction.get("model_version", "sam3-init-seed"),
            result=prediction.get("result", []),
        )
        pred_id = getattr(result, "id", None)
        if pred_id is not None:
            logger.info("Upload complete, prediction id=%s", pred_id)
        else:
            logger.info("Upload request completed (no prediction id in response)")
    except Exception as exc:  # pragma: no cover
        msg = str(exc)
        err_no = getattr(exc, "errno", None)
        if "504" in msg:
            logger.warning("Received 504 from LS during prediction upload; assuming it succeeded.")
        else:
            if err_no is not None:
                logger.error("Failed to upload prediction (errno=%s): %s", err_no, msg)
            else:
                logger.error("Failed to upload prediction: %s", msg)
