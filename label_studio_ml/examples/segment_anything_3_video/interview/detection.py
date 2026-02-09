"""Detection pipeline for Interview UI.

Implements text-based object detection using Sam3Model (image mode) with
Promptable Concept Segmentation (PCS). Detections are produced per sampled
keyframe, filtered through NMS, padded, and stored as CropData objects on
the InterviewSession.

Key components:
- Sam3TextBasedDetector: wraps Sam3Model for text-prompted instance
  segmentation.  Unlike the Sam3TextDetector in seeding_common.py (which
  uses Sam3VideoModel), this class works with the image-only model and
  supports multi-prompt reuse via cached pixel_values.
- nms_numpy: greedy non-maximum suppression (pure numpy, no cv2/torchvision).
- pad_boxes: expand detections by a configurable fraction, clamped to frame
  bounds.
- run_detection_pipeline: end-to-end entry point called from routes.py.
- run_recall_strategy: additional detection passes to close recall gaps.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Parent-directory imports (seeding_common lives one level up)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import av  # noqa: E402

from seeding_common import (  # noqa: E402
    _get_sam3_image_model,
    _read_frame_pyav,
    _get_video_info_pyav,
    _compute_sam3_frame_embeddings,
    _do_embed_all_frames,
    compute_change_scores,
    smooth_change_scores,
    select_keyframes,
    DEVICE,
    DTYPE,
)

from .state import CropData, CropLabel, CropSource, InterviewSession, Phase  # noqa: E402
from .cache_manager import save_session  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Keyframe sampling
DEFAULT_KEYFRAME_FRAC = float(os.getenv("INTERVIEW_KEYFRAME_FRAC", "0.04"))
DEFAULT_MIN_SPACING = int(os.getenv("INTERVIEW_MIN_SPACING", "30"))
DEFAULT_EMBEDDING_BATCH = int(os.getenv("INTERVIEW_EMBEDDING_BATCH", "32"))
EMBEDDING_CACHE_DIR = os.getenv("INTERVIEW_EMBEDDING_CACHE", "/tmp/interview_embed_cache")
INITIAL_KEYFRAME_COUNT = int(os.getenv("INTERVIEW_INITIAL_KEYFRAMES", "40"))

# Batch detection
DEFAULT_DETECT_BATCH = int(os.getenv("INTERVIEW_DETECT_BATCH", "8"))

# Detection
DEFAULT_DETECTION_THRESHOLD = float(os.getenv("INTERVIEW_DETECT_THRESHOLD", "0.3"))
DEFAULT_MASK_THRESHOLD = float(os.getenv("INTERVIEW_MASK_THRESHOLD", "0.5"))
DEFAULT_NMS_IOU_THRESHOLD = float(os.getenv("INTERVIEW_NMS_IOU", "0.5"))
DEFAULT_PAD_FRAC = float(os.getenv("INTERVIEW_PAD_FRAC", "0.1"))

# Deduplication
DEFAULT_DEDUP_IOU_THRESHOLD = float(os.getenv("INTERVIEW_DEDUP_IOU", "0.5"))

# Minimum box area in pixels to keep a detection
MIN_BOX_AREA_PX = int(os.getenv("INTERVIEW_MIN_BOX_AREA", "100"))


# ===========================================================================
# NMS (pure numpy)
# ===========================================================================

def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of xyxy boxes.

    Args:
        boxes_a: (N, 4) array of [x1, y1, x2, y2].
        boxes_b: (M, 4) array of [x1, y1, x2, y2].

    Returns:
        (N, M) IoU matrix.
    """
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)  # (N, M)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter_area
    iou = np.where(union > 0, inter_area / union, 0.0)
    return iou


def nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Greedy non-maximum suppression (pure numpy, no cv2/torchvision).

    Sorts detections by score in descending order.  Keeps each box only if
    its IoU with every previously kept box is below ``iou_threshold``.

    Args:
        boxes:  (N, 4) float array of [x1, y1, x2, y2] coordinates.
        scores: (N,) float array of confidence scores.
        iou_threshold: Suppress boxes with IoU >= this value against a
            higher-scoring kept box.

    Returns:
        (K,) int array of indices into the original arrays that survive NMS.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    keep: List[int] = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    suppressed = np.zeros(len(boxes), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))

        # Compute IoU of this box against all remaining unsuppressed boxes
        xx1 = np.maximum(x1[idx], x1)
        yy1 = np.maximum(y1[idx], y1)
        xx2 = np.minimum(x2[idx], x2)
        yy2 = np.minimum(y2[idx], y2)

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union = areas[idx] + areas - inter_area
        iou = np.where(union > 0, inter_area / union, 0.0)

        suppressed |= iou >= iou_threshold

    return np.array(keep, dtype=np.int64)


# ===========================================================================
# Box padding
# ===========================================================================

def pad_boxes(
    boxes: np.ndarray,
    width: int,
    height: int,
    pad_frac: float = 0.1,
) -> np.ndarray:
    """Expand each box by ``pad_frac`` on all sides, clamped to frame bounds.

    For a 10 % pad (the default), a 100 px-wide box gains 10 px on the left
    and 10 px on the right (total 120 px).

    Args:
        boxes:    (N, 4) float array of [x1, y1, x2, y2] in pixel coords.
        width:    Frame width in pixels.
        height:   Frame height in pixels.
        pad_frac: Fraction of box width/height to add on each side.

    Returns:
        (N, 4) padded boxes, clamped to [0, width] x [0, height].
    """
    if len(boxes) == 0:
        return boxes.copy()

    padded = boxes.copy().astype(np.float32)
    bw = padded[:, 2] - padded[:, 0]
    bh = padded[:, 3] - padded[:, 1]

    dx = bw * pad_frac
    dy = bh * pad_frac

    padded[:, 0] -= dx
    padded[:, 1] -= dy
    padded[:, 2] += dx
    padded[:, 3] += dy

    # Clamp to frame bounds
    padded[:, 0] = np.clip(padded[:, 0], 0, width)
    padded[:, 1] = np.clip(padded[:, 1], 0, height)
    padded[:, 2] = np.clip(padded[:, 2], 0, width)
    padded[:, 3] = np.clip(padded[:, 3], 0, height)

    return padded


# ===========================================================================
# Batch decode + detect helpers
# ===========================================================================

def uniform_indices(total: int, k: int) -> List[int]:
    """Return *k* uniformly-spaced frame indices in [0, total).

    Always includes the first and last frame when k >= 2.
    """
    if total <= 0 or k <= 0:
        return []
    if k >= total:
        return list(range(total))
    if k == 1:
        return [total // 2]
    return [int(round(i * (total - 1) / (k - 1))) for i in range(k)]


_MAX_DECODE_AFTER_SEEK = int(os.getenv("INTERVIEW_MAX_DECODE_AFTER_SEEK", "500"))


def _decode_frames_sequential(
    video_path: str,
    frame_indices: List[int],
    max_decode_after_seek: int = _MAX_DECODE_AFTER_SEEK,
) -> Dict[int, Image.Image]:
    """Decode specific frames using keyframe-seeking for widely-spaced targets.

    For each target frame, seeks to the nearest prior keyframe and then
    decodes forward to the exact frame.  This avoids decoding every frame
    in the video (which for 30K frames takes ~10 minutes) when only ~40
    uniformly-spaced frames are needed.

    A safety limit (``max_decode_after_seek``) caps how many frames we
    decode after each seek.  If the target isn't found within that window
    the frame is skipped and a warning is logged.

    Args:
        video_path:             Path to video file.
        frame_indices:          List of 0-based frame indices to decode (need not be sorted).
        max_decode_after_seek:  Maximum frames to decode after each seek before giving up.

    Returns:
        Dict mapping frame_idx -> PIL Image.
    """
    if not frame_indices:
        return {}

    result: Dict[int, Image.Image] = {}
    sorted_targets = sorted(set(frame_indices))

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        time_base = stream.time_base

        for target_idx in sorted_targets:
            # Seek to just before the target frame.  PyAV seeks to the
            # nearest prior keyframe, then we decode forward.
            target_pts = int(target_idx / fps / time_base)
            container.seek(target_pts, stream=stream)

            frame_count_after_seek = None
            decoded_count = 0
            found = False
            for av_frame in container.decode(video=0):
                decoded_count += 1
                # After seek, the first decoded frame is the keyframe at or
                # before our target.  We figure out which frame index it is
                # from its pts, then count forward.
                if frame_count_after_seek is None:
                    if av_frame.pts is not None and time_base:
                        frame_count_after_seek = int(
                            round(float(av_frame.pts * time_base) * fps)
                        )
                    else:
                        # Fallback: assume seek landed on target (best effort)
                        frame_count_after_seek = target_idx

                if frame_count_after_seek == target_idx:
                    result[target_idx] = av_frame.to_image()
                    found = True
                    break
                elif frame_count_after_seek > target_idx:
                    # Overshot (rare, can happen with some codecs) — take it
                    result[target_idx] = av_frame.to_image()
                    found = True
                    break

                frame_count_after_seek += 1

                if decoded_count >= max_decode_after_seek:
                    break

            if not found:
                logger.warning(
                    "Could not find frame %d after decoding %d frames post-seek; skipping",
                    target_idx, decoded_count,
                )

        logger.info(
            "Decoded %d / %d target frames via seek from %s",
            len(result), len(sorted_targets), video_path,
        )
    finally:
        container.close()

    return result


def _detect_batch(
    detector: "Sam3TextBasedDetector",
    frames: Dict[int, Image.Image],
    prompt: str,
    width: int,
    height: int,
    batch_size: int = DEFAULT_DETECT_BATCH,
    nms_iou: float = DEFAULT_NMS_IOU_THRESHOLD,
    pad_frac: float = DEFAULT_PAD_FRAC,
) -> List[CropData]:
    """Run batched detection on pre-decoded frames.

    Groups frames into batches of ``batch_size`` and runs the Sam3 model
    in a single forward pass per batch.  The processor requires ``text``
    to be a **list of prompts** (one per image) for batched inference —
    we replicate the same prompt for every image in the batch.

    Falls back to per-frame inference on OOM.

    Args:
        detector:   Sam3TextBasedDetector instance.
        frames:     Dict of frame_idx -> PIL Image (from _decode_frames_sequential).
        prompt:     Text prompt.
        width:      Video width in pixels.
        height:     Video height in pixels.
        batch_size: Frames per GPU forward pass.
        nms_iou:    NMS IoU threshold.
        pad_frac:   Box padding fraction.

    Returns:
        List of CropData across all frames.
    """
    all_crops: List[CropData] = []
    sorted_indices = sorted(frames.keys())

    for batch_start in range(0, len(sorted_indices), batch_size):
        batch_indices = sorted_indices[batch_start:batch_start + batch_size]
        batch_images = [frames[idx] for idx in batch_indices]

        # Key: text must be a list of prompts, one per image
        text_prompts = [prompt] * len(batch_images)

        try:
            inputs = detector.processor(
                images=batch_images, text=text_prompts, return_tensors="pt",
            ).to(DEVICE)

            with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
                outputs = detector.model(**inputs)

            # Use original_sizes from processor if available, else compute
            if inputs.get("original_sizes") is not None:
                target_sizes = inputs.get("original_sizes").tolist()
            else:
                target_sizes = [[img.height, img.width] for img in batch_images]

            batch_results = detector.processor.post_process_instance_segmentation(
                outputs,
                threshold=detector.threshold,
                mask_threshold=detector.mask_threshold,
                target_sizes=target_sizes,
            )
        except torch.cuda.OutOfMemoryError:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            logger.warning(
                "OOM during batch detection (batch_size=%d), falling back to per-frame",
                len(batch_images),
            )
            for idx in batch_indices:
                crops = _detect_single_frame(
                    detector, frames[idx], prompt, idx, width, height,
                    nms_iou=nms_iou, pad_frac=pad_frac,
                )
                all_crops.extend(crops)
            continue

        # Post-process each image in the batch
        for i, frame_idx in enumerate(batch_indices):
            results_i = batch_results[i]
            dets = Sam3TextBasedDetector._parse_results(results_i, prompt)

            if not dets:
                continue

            boxes = np.array([d["xyxy"] for d in dets], dtype=np.float32)
            scores = np.array([d["score"] for d in dets], dtype=np.float32)

            keep_idx = nms_numpy(boxes, scores, iou_threshold=nms_iou)
            if len(keep_idx) == 0:
                continue

            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            boxes = pad_boxes(boxes, width, height, pad_frac=pad_frac)

            for j in range(len(boxes)):
                crop = CropData(
                    crop_id=str(uuid.uuid4())[:12],
                    frame_idx=frame_idx,
                    xyxy=boxes[j],
                    score=float(scores[j]),
                    label=CropLabel.PENDING,
                    source=CropSource.TEXT_DETECT,
                    prompt=prompt,
                )
                all_crops.append(crop)

    return all_crops


# ===========================================================================
# Sam3TextBasedDetector (image-mode PCS)
# ===========================================================================

class Sam3TextBasedDetector:
    """Text-prompted instance segmentation via Sam3Model (image mode).

    Unlike ``Sam3TextDetector`` in seeding_common.py which wraps
    ``Sam3VideoModel``, this class uses the lighter-weight ``Sam3Model``
    (image-only) and supports caching of preprocessed pixel values so that
    multiple text prompts can be run against the same frame cheaply.

    Typical usage::

        detector = Sam3TextBasedDetector(threshold=0.3)
        detector.set_frame(pil_image)
        dets = detector.detect("person")
        more = detector.detect("bicycle")  # reuses cached frame
    """

    def __init__(
        self,
        threshold: float = DEFAULT_DETECTION_THRESHOLD,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
    ):
        self.model, self.processor = _get_sam3_image_model()
        self.threshold = threshold
        self.mask_threshold = mask_threshold

        # Cached per-frame data
        self._cached_pil: Optional[Image.Image] = None
        self._cached_pixel_values: Optional[torch.Tensor] = None
        self._cached_original_sizes: Optional[List[List[int]]] = None

    # ------------------------------------------------------------------
    # Frame caching
    # ------------------------------------------------------------------

    def set_frame(self, pil_image: Image.Image) -> None:
        """Pre-process and cache pixel values for a frame.

        Calling :meth:`detect` afterwards will reuse these tensors, saving
        the image encoding cost when running multiple prompts on the same
        frame.
        """
        if pil_image is self._cached_pil:
            return  # already cached

        inputs = self.processor(images=pil_image, return_tensors="pt").to(DEVICE)
        self._cached_pil = pil_image
        self._cached_pixel_values = inputs.pixel_values
        # original_sizes may be present depending on processor version
        self._cached_original_sizes = (
            inputs.get("original_sizes").tolist()
            if inputs.get("original_sizes") is not None
            else [[pil_image.height, pil_image.width]]
        )

    def clear_cache(self) -> None:
        """Release cached frame tensors to free GPU memory."""
        self._cached_pil = None
        self._cached_pixel_values = None
        self._cached_original_sizes = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        prompt: str,
        pil_image: Optional[Image.Image] = None,
    ) -> List[Dict[str, Any]]:
        """Run text-based detection (PCS) on a frame.

        If ``pil_image`` is provided it replaces any cached frame.  If the
        frame was previously set via :meth:`set_frame` and ``pil_image`` is
        ``None``, the cached tensors are reused.

        Args:
            prompt:    Text describing the target class (e.g. "person").
            pil_image: Optional PIL Image.  If omitted, uses cached frame.

        Returns:
            List of dicts, each with keys:
                - ``xyxy``: np.ndarray of [x1, y1, x2, y2] in pixel coords.
                - ``score``: float confidence.
                - ``label``: str (the text prompt used).
        """
        if pil_image is not None:
            self.set_frame(pil_image)

        if self._cached_pil is None:
            raise RuntimeError(
                "No frame cached. Call set_frame() or pass pil_image."
            )

        # Full forward pass with text prompt.
        # NOTE: We re-run the full model here rather than passing pre-
        # computed vision features because the Sam3Processor text+image
        # encoding is tightly coupled.  A future optimisation can split
        # the vision backbone call from the prompt encoder call.
        inputs = self.processor(
            images=self._cached_pil,
            text=prompt,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=DTYPE):
            outputs = self.model(**inputs)

        # Determine target sizes for post-processing
        if inputs.get("original_sizes") is not None:
            target_sizes = inputs.get("original_sizes").tolist()
        else:
            target_sizes = self._cached_original_sizes or [
                [self._cached_pil.height, self._cached_pil.width]
            ]

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )[0]

        return self._parse_results(results, prompt)

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_results(
        results: Dict[str, Any],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """Convert processor output to a list of detection dicts.

        Handles both tensor and plain-list formats returned by various
        versions of the Sam3Processor post-processing API.
        """
        masks = results.get("masks", [])
        scores_raw = results.get("scores", [])
        boxes_raw = results.get("boxes", [])
        labels_raw = results.get("labels", [])

        detections: List[Dict[str, Any]] = []

        n_detections = max(len(scores_raw), len(boxes_raw), len(masks))
        if n_detections == 0:
            return detections

        for i in range(n_detections):
            # --- score ---
            if i < len(scores_raw):
                s = scores_raw[i]
                score = float(s.item() if hasattr(s, "item") else s)
            else:
                score = 0.0

            # --- box ---
            box = None
            if i < len(boxes_raw):
                b = boxes_raw[i]
                if hasattr(b, "tolist"):
                    b = b.tolist()
                elif hasattr(b, "cpu"):
                    b = b.cpu().numpy().tolist()
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    box = np.array(b[:4], dtype=np.float32)

            # Fall back: derive box from mask
            if box is None and i < len(masks):
                mask = masks[i]
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                elif not isinstance(mask, np.ndarray):
                    mask = np.asarray(mask)
                ys, xs = np.where(mask > 0)
                if xs.size > 0 and ys.size > 0:
                    box = np.array(
                        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                        dtype=np.float32,
                    )

            if box is None:
                continue

            # Sanity: skip degenerate boxes
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            if bw <= 0 or bh <= 0 or (bw * bh) < MIN_BOX_AREA_PX:
                continue

            # --- label ---
            if i < len(labels_raw):
                lbl = labels_raw[i]
                label = str(lbl.item() if hasattr(lbl, "item") else lbl)
            else:
                label = prompt

            detections.append({
                "xyxy": box,
                "score": score,
                "label": label,
            })

        return detections


# ===========================================================================
# Keyframe sampling helpers
# ===========================================================================

def _sample_keyframes(
    session: InterviewSession,
    progress: Any,
    keyframe_frac: float = DEFAULT_KEYFRAME_FRAC,
    min_spacing: int = DEFAULT_MIN_SPACING,
    embedding_batch: int = DEFAULT_EMBEDDING_BATCH,
) -> List[int]:
    """Compute or reuse sampled keyframes for a session.

    If the session already has ``sampled_frames`` populated (e.g. from a
    previous detection pass or a cached session), those are returned
    directly.  Otherwise, SAM3 frame embeddings are computed and
    change-detection-based keyframe selection is performed.

    Args:
        session:         The interview session (must have video_path set).
        progress:        JobProgress object to update UI.
        keyframe_frac:   Fraction of total frames to sample.
        min_spacing:     Minimum frame gap between keyframes.
        embedding_batch: Batch size for SAM3 embedding computation.

    Returns:
        Sorted list of 0-based frame indices.
    """
    if session.sampled_frames:
        logger.info(
            "Reusing %d previously sampled keyframes", len(session.sampled_frames)
        )
        return session.sampled_frames

    video_path = session.video_path
    if not video_path:
        raise RuntimeError("Session has no video_path set. Call video_info first.")

    progress.step = "Computing frame embeddings for keyframe selection..."
    progress.current = 0

    cache_key = session.cache_key
    embeds = _compute_sam3_frame_embeddings(
        cache_key, video_path, embedding_batch, EMBEDDING_CACHE_DIR
    )

    # Use actual embedding count as ground truth for frame count
    frames_count = embeds.shape[0]
    if session.frames_count and session.frames_count != frames_count:
        logger.warning(
            "Session frames_count=%d but embeddings have %d frames; using min",
            session.frames_count,
            frames_count,
        )
        frames_count = min(session.frames_count, frames_count)
        embeds = embeds[:frames_count]

    progress.step = "Selecting keyframes via change detection..."
    diff = compute_change_scores(embeds)
    smooth = smooth_change_scores(diff, kernel_size=5)
    keyframes = select_keyframes(
        frames_count, keyframe_frac, smooth, min_spacing=min_spacing
    )

    logger.info(
        "Selected %d keyframes from %d total frames (frac=%.3f)",
        len(keyframes), frames_count, keyframe_frac,
    )

    # Persist on session
    with session._lock:
        session.sampled_frames = keyframes
        session.touch()

    return keyframes


# ===========================================================================
# Deduplication helpers
# ===========================================================================

def _deduplicate_against_existing(
    new_boxes: np.ndarray,
    new_scores: np.ndarray,
    existing_boxes: np.ndarray,
    iou_threshold: float = DEFAULT_DEDUP_IOU_THRESHOLD,
) -> np.ndarray:
    """Return indices of ``new_boxes`` that do NOT overlap with existing.

    Args:
        new_boxes:      (N, 4) candidate detections.
        new_scores:     (N,) confidence scores (unused but kept for API symmetry).
        existing_boxes: (M, 4) already-accepted detections.
        iou_threshold:  Max IoU a new box may have with any existing box.

    Returns:
        (K,) int array of surviving indices into ``new_boxes``.
    """
    if len(new_boxes) == 0 or len(existing_boxes) == 0:
        return np.arange(len(new_boxes), dtype=np.int64)

    iou = _compute_iou_matrix(new_boxes, existing_boxes)  # (N, M)
    max_iou_per_new = iou.max(axis=1)  # (N,)
    keep_mask = max_iou_per_new < iou_threshold
    return np.where(keep_mask)[0].astype(np.int64)


# ===========================================================================
# Detection on a single frame
# ===========================================================================

def _detect_single_frame(
    detector: Sam3TextBasedDetector,
    pil_image: Image.Image,
    prompt: str,
    frame_idx: int,
    width: int,
    height: int,
    nms_iou: float = DEFAULT_NMS_IOU_THRESHOLD,
    pad_frac: float = DEFAULT_PAD_FRAC,
) -> List[CropData]:
    """Run detection + NMS + padding on one frame and return CropData list.

    Args:
        detector:   Sam3TextBasedDetector instance.
        pil_image:  PIL Image of the frame.
        prompt:     Text prompt for detection.
        frame_idx:  0-based frame index.
        width:      Video frame width.
        height:     Video frame height.
        nms_iou:    IoU threshold for NMS.
        pad_frac:   Fraction for box padding.

    Returns:
        List of CropData objects for this frame.
    """
    raw_dets = detector.detect(prompt, pil_image)

    if not raw_dets:
        return []

    # Stack into arrays for vectorised NMS / padding
    boxes = np.array([d["xyxy"] for d in raw_dets], dtype=np.float32)
    scores = np.array([d["score"] for d in raw_dets], dtype=np.float32)

    # NMS
    keep_idx = nms_numpy(boxes, scores, iou_threshold=nms_iou)
    if len(keep_idx) == 0:
        return []

    boxes = boxes[keep_idx]
    scores = scores[keep_idx]

    # Padding
    boxes = pad_boxes(boxes, width, height, pad_frac=pad_frac)

    # Build CropData
    crops: List[CropData] = []
    for i in range(len(boxes)):
        crop = CropData(
            crop_id=str(uuid.uuid4())[:12],
            frame_idx=frame_idx,
            xyxy=boxes[i],
            score=float(scores[i]),
            label=CropLabel.PENDING,
            source=CropSource.TEXT_DETECT,
            prompt=prompt,
        )
        crops.append(crop)

    return crops


# ===========================================================================
# Main pipeline entry point
# ===========================================================================

def run_detection_pipeline(
    session: InterviewSession,
    prompt: str,
    progress: Any,
) -> Dict[str, Any]:
    """Run the full detection pipeline and populate session crops.

    Called from ``routes.detect_start`` via the background job executor.

    Steps:
        1. Sample keyframes (or reuse existing).
        2. For each keyframe, read frame, detect with Sam3Model, NMS, pad.
        3. Store CropData objects on the session.
        4. Advance session phase to DETECTION.
        5. Persist session to disk.

    Args:
        session:  InterviewSession with video_path already set.
        prompt:   Text prompt for detection (e.g. "person").
        progress: JobProgress object (updated for UI polling).

    Returns:
        Summary dict with detection statistics.
    """
    t0 = time.time()

    # Record prompt
    if prompt not in session.prompts:
        session.prompts.append(prompt)

    # Step 1: sample keyframes
    progress.step = "Sampling keyframes..."
    progress.total = 3  # rough phases: sample, detect, finalise
    progress.current = 0

    keyframes = _sample_keyframes(session, progress)
    progress.current = 1

    # Step 2: detect on each keyframe
    progress.step = "Running text-based detection on keyframes..."
    progress.total = len(keyframes) + 2  # +2 for sample + finalise
    progress.current = 1

    detector = Sam3TextBasedDetector()
    width = session.width
    height = session.height
    total_crops = 0

    for i, frame_idx in enumerate(keyframes):
        progress.step = f"Detecting on frame {frame_idx} ({i + 1}/{len(keyframes)})..."
        progress.current = i + 2  # offset by 1 for sampling phase

        pil_image = _read_frame_pyav(session.video_path, frame_idx)
        if pil_image is None:
            logger.warning("Failed to read frame %d, skipping", frame_idx)
            continue

        crops = _detect_single_frame(
            detector, pil_image, prompt, frame_idx, width, height,
        )

        with session._lock:
            for crop in crops:
                session.add_crop(crop)
                total_crops += 1

    # Release GPU memory
    detector.clear_cache()

    # Step 3: finalise
    progress.step = "Saving session..."
    progress.current = progress.total - 1

    with session._lock:
        session.advance_to(Phase.DETECTION)

    save_session(session)

    elapsed = time.time() - t0
    progress.step = "Detection complete."
    progress.current = progress.total

    summary = {
        "keyframes": len(keyframes),
        "total_crops": total_crops,
        "prompt": prompt,
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info(
        "Detection pipeline finished: %d crops on %d keyframes in %.1fs",
        total_crops, len(keyframes), elapsed,
    )
    return summary


# ===========================================================================
# Stage 1: fast detection on uniform frames (no embeddings)
# ===========================================================================

def run_detection_stage1(
    session: InterviewSession,
    prompt: str,
    progress: Any,
) -> Dict[str, Any]:
    """Fast initial detection: uniform frame sampling + batched inference.

    Selects ``INITIAL_KEYFRAME_COUNT`` uniformly-spaced frames, pre-decodes
    them in a single PyAV pass, runs batched Sam3 detection, and stores
    crops on the session.  No embedding computation — the user can start
    labeling within ~30-60 seconds.

    Args:
        session:  InterviewSession with video_path already set.
        prompt:   Text prompt for detection.
        progress: JobProgress object.

    Returns:
        Summary dict with detection statistics.
    """
    t0 = time.time()

    if prompt not in session.prompts:
        session.prompts.append(prompt)

    frames_count = session.frames_count
    k = min(INITIAL_KEYFRAME_COUNT, frames_count) if frames_count > 0 else INITIAL_KEYFRAME_COUNT
    keyframe_indices = uniform_indices(frames_count, k)

    # Step 1: decode target frames in one sequential pass
    progress.step = f"Decoding {len(keyframe_indices)} frames..."
    progress.total = 3
    progress.current = 0

    frame_images = _decode_frames_sequential(session.video_path, keyframe_indices)
    if not frame_images:
        raise RuntimeError(
            f"Failed to decode any of the {len(keyframe_indices)} target keyframes from {session.video_path}"
        )
    progress.current = 1

    # Step 2: batched detection
    progress.step = f"Running batched detection on {len(frame_images)} frames..."
    progress.current = 1

    detector = Sam3TextBasedDetector()
    crops = _detect_batch(
        detector, frame_images, prompt,
        session.width, session.height,
        batch_size=DEFAULT_DETECT_BATCH,
    )
    detector.clear_cache()
    progress.current = 2

    # Step 3: store crops, advance phase
    progress.step = "Saving results..."
    total_crops = 0
    with session._lock:
        session.sampled_frames = sorted(frame_images.keys())
        for crop in crops:
            session.add_crop(crop)
            total_crops += 1
        session.advance_to(Phase.DETECTION)
        save_session(session)

    elapsed = time.time() - t0
    progress.step = "Detection complete."
    progress.current = progress.total

    summary = {
        "keyframes": len(keyframe_indices),
        "total_crops": total_crops,
        "prompt": prompt,
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info(
        "Stage 1 detection: %d crops on %d uniform keyframes in %.1fs",
        total_crops, len(keyframe_indices), elapsed,
    )
    return summary


# ===========================================================================
# Background embedding + change detection
# ===========================================================================

def run_embedding_background(
    session: InterviewSession,
    progress: Any,
) -> Dict[str, Any]:
    """Compute GPU-batched frame embeddings and change-detected keyframes.

    Runs concurrently with the user's labeling session.  When complete,
    stores the change-detected keyframe indices on the session for use
    by subsequent active-learning rounds and recall strategies.

    Args:
        session:  InterviewSession with video_path already set.
        progress: JobProgress object.

    Returns:
        Summary dict with embedding stats.
    """
    t0 = time.time()

    video_path = session.video_path
    if not video_path:
        raise RuntimeError("Session has no video_path set.")

    def _progress_cb(current: int, total: int):
        progress.step = f"Embedding frames {current:,} / {total:,}..."
        progress.current = current
        progress.total = total

    progress.step = "Computing frame embeddings..."
    embeds = _do_embed_all_frames(
        video_path, DEFAULT_EMBEDDING_BATCH, progress_callback=_progress_cb,
    )

    # Change detection
    progress.step = "Running change detection..."
    frames_count = embeds.shape[0]
    diff = compute_change_scores(embeds)
    smooth = smooth_change_scores(diff, kernel_size=5)
    change_keyframes = select_keyframes(
        frames_count, DEFAULT_KEYFRAME_FRAC, smooth,
        min_spacing=DEFAULT_MIN_SPACING,
    )

    # Store on session
    with session._lock:
        session.embedding_complete = True
        session.change_keyframes = change_keyframes
        session.touch()
        save_session(session)

    elapsed = time.time() - t0
    progress.step = "Embedding complete."
    progress.current = progress.total

    summary = {
        "frames_embedded": int(frames_count),
        "change_keyframes": len(change_keyframes),
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info(
        "Background embedding: %d frames, %d change keyframes in %.1fs",
        frames_count, len(change_keyframes), elapsed,
    )
    return summary


# ===========================================================================
# Recall strategy: additional detection passes
# ===========================================================================

def _run_multi_prompt_strategy(
    session: InterviewSession,
    extra_prompts: List[str],
    progress: Any,
) -> Dict[str, Any]:
    """Run detector with additional text prompts on already-sampled frames.

    Deduplicates new detections against existing crops on each frame using
    IoU to avoid redundant proposals.

    Args:
        session:       InterviewSession (must already be in DETECTION phase).
        extra_prompts: Additional text prompts to try.
        progress:      JobProgress object.

    Returns:
        Summary dict.
    """
    keyframes = session.sampled_frames
    if not keyframes:
        raise RuntimeError("No sampled frames available. Run detection first.")

    width = session.width
    height = session.height
    detector = Sam3TextBasedDetector()

    total_new = 0
    total_deduped = 0

    progress.total = len(keyframes) * len(extra_prompts)
    progress.current = 0
    step_counter = 0

    for prompt in extra_prompts:
        if prompt not in session.prompts:
            session.prompts.append(prompt)

        for frame_idx in keyframes:
            step_counter += 1
            progress.step = (
                f"Multi-prompt '{prompt}' on frame {frame_idx} "
                f"({step_counter}/{progress.total})..."
            )
            progress.current = step_counter

            pil_image = _read_frame_pyav(session.video_path, frame_idx)
            if pil_image is None:
                continue

            new_crops = _detect_single_frame(
                detector, pil_image, prompt, frame_idx, width, height,
            )

            if not new_crops:
                continue

            # Deduplicate against existing crops on this frame
            existing_on_frame = session.get_crops_by_frame(frame_idx)
            if existing_on_frame:
                existing_boxes = np.array(
                    [c.xyxy for c in existing_on_frame], dtype=np.float32
                )
                new_boxes = np.array(
                    [c.xyxy for c in new_crops], dtype=np.float32
                )
                new_scores = np.array(
                    [c.score for c in new_crops], dtype=np.float32
                )
                keep_idx = _deduplicate_against_existing(
                    new_boxes, new_scores, existing_boxes,
                    iou_threshold=DEFAULT_DEDUP_IOU_THRESHOLD,
                )
                deduped = len(new_crops) - len(keep_idx)
                total_deduped += deduped
                new_crops = [new_crops[i] for i in keep_idx]

            # Tag source as multi-prompt
            for crop in new_crops:
                crop.source = CropSource.MULTI_PROMPT

            with session._lock:
                for crop in new_crops:
                    session.add_crop(crop)
                    total_new += 1

    detector.clear_cache()
    save_session(session)

    summary = {
        "strategy": "multi_prompt",
        "extra_prompts": extra_prompts,
        "new_crops": total_new,
        "deduplicated": total_deduped,
    }
    logger.info(
        "Multi-prompt recall: %d new crops (%d deduplicated)",
        total_new, total_deduped,
    )
    return summary


def _run_feature_search_strategy(
    session: InterviewSession,
    progress: Any,
) -> Dict[str, Any]:
    """Dense-grid DINOv3 feature search recall strategy.

    Delegates to :func:`dinov3_classifier.run_feature_search` which:
    1. Defines a sliding-window grid across sampled frames at multiple scales.
    2. Extracts DINOv3 features for each grid cell.
    3. Compares against the feature centroid of accepted crops.
    4. Returns high-similarity candidates that don't overlap existing crops.

    Args:
        session:  InterviewSession.
        progress: JobProgress object.

    Returns:
        Summary dict with strategy name and new crop count.
    """
    from .dinov3_classifier import run_feature_search

    new_crops = run_feature_search(session, progress)

    return {
        "strategy": "feature_search",
        "status": "completed",
        "new_crops": len(new_crops),
    }


def run_recall_strategy(
    session: InterviewSession,
    strategy: str,
    extra_prompts: List[str],
    progress: Any,
) -> Dict[str, Any]:
    """Dispatch to the requested recall-gap strategy.

    Called from ``routes.detect_recall_strategy`` via background executor.

    Supported strategies:
        - ``"multi_prompt"``: Run detector with additional text prompts on
          already-sampled frames; deduplicate against existing crops.
        - ``"feature_search"``: Dense-grid DINOv3 feature search.

    Args:
        session:       InterviewSession.
        strategy:      One of "multi_prompt" or "feature_search".
        extra_prompts: Additional text prompts (used by multi_prompt).
        progress:      JobProgress object.

    Returns:
        Summary dict from the selected strategy.

    Raises:
        ValueError: If the strategy name is not recognised.
    """
    t0 = time.time()

    if strategy == "multi_prompt":
        if not extra_prompts:
            raise ValueError(
                "multi_prompt strategy requires at least one prompt in 'prompts'."
            )
        result = _run_multi_prompt_strategy(session, extra_prompts, progress)
    elif strategy == "feature_search":
        result = _run_feature_search_strategy(session, progress)
    else:
        raise ValueError(
            f"Unknown recall strategy: {strategy!r}. "
            f"Supported: 'multi_prompt', 'feature_search'."
        )

    elapsed = time.time() - t0
    result["elapsed_seconds"] = round(elapsed, 1)
    progress.step = f"Recall strategy '{strategy}' complete."

    logger.info(
        "Recall strategy '%s' finished in %.1fs: %s",
        strategy, elapsed, result,
    )
    return result
