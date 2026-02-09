from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from label_studio_sdk.client import LabelStudio
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path

from mergevideoregions import extract_merge_key_from_result, merge_results_by_merge_id
import seeding_common as base


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class ReIDCLIError(Exception):
    """Custom error type for the re-identification CLI."""


@dataclass
class TrackSequence:
    """Container for a single video rectangle sequence (track)."""

    region_id: str
    merge_id: Optional[int]
    result: Dict[str, Any]
    sequence: List[Dict[str, Any]]
    frames_count: Optional[int]
    duration: Optional[float]

    @property
    def frame_range(self) -> Tuple[int, int]:
        frames = [int(f.get("frame", 0)) for f in self.sequence if isinstance(f, dict)]
        if not frames:
            return 0, -1
        return min(frames), max(frames)


@dataclass
class TrackFeatures:
    """Aggregated appearance + geometry + shape features for a track."""

    hist: np.ndarray
    geom: np.ndarray
    shape: np.ndarray
    samples: int
    sam3_vec: Optional[np.ndarray] = None


@dataclass
class ReIDPreset:
    name: str
    max_samples_per_track: int
    hist_bins: Tuple[int, int, int]
    color_weight: float
    geom_weight: float
    shape_weight: float
    min_sim_for_match: float
    decisive_margin: float
    decisive_min_top1: float


PRESETS: Dict[str, ReIDPreset] = {
    "uav": ReIDPreset(
        name="uav",
        max_samples_per_track=15,
        hist_bins=(8, 8, 8),
        color_weight=0.5,
        geom_weight=0.25,
        shape_weight=0.25,
        min_sim_for_match=0.35,
        decisive_margin=0.15,
        decisive_min_top1=0.55,
    ),
    "ugv": ReIDPreset(
        name="ugv",
        max_samples_per_track=10,
        hist_bins=(8, 8, 8),
        color_weight=0.4,
        geom_weight=0.3,
        shape_weight=0.3,
        min_sim_for_match=0.4,
        decisive_margin=0.2,
        decisive_min_top1=0.6,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute automatic re-identification suggestions for video tracks "
            "in a single Label Studio annotation and upload them as a prediction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (inside Docker):\n\n"
            "  docker compose exec segment_anything_3_video bash -lc '" "\\n"
            "    export LABEL_STUDIO_HOST=https://app.heartex.com &&" "\\n"
            "    export LABEL_STUDIO_URL=https://app.heartex.com &&" "\\n"
            "    export LABEL_STUDIO_API_KEY=\"$LABEL_STUDIO_API_KEY\" &&" "\\n"
            "    python /app/complete_reid.py \\\n"
            "      --ls-url https://app.heartex.com \\\n"
            "      --ls-api-key \"$LABEL_STUDIO_API_KEY\" \\\n"
            "      --project 123 \\\n"
            "      --task 456 \\\n"
            "      --annotation 789 \\\n"
            "      --profile uav\n" "\\n"
            "  '"
        ),
    )

    parser.add_argument(
        "--ls-url",
        required=True,
        help="Label Studio URL (e.g., https://app.heartex.com)",
    )
    parser.add_argument(
        "--ls-api-key",
        required=True,
        help="Label Studio API key",
    )
    parser.add_argument(
        "--project",
        type=int,
        required=True,
        help="Project ID (used for validation/logging)",
    )
    parser.add_argument(
        "--task",
        type=int,
        required=True,
        help="Task ID associated with the annotation",
    )
    parser.add_argument(
        "--annotation",
        type=int,
        required=True,
        help="Annotation ID to use as the source of tracks",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PRESETS.keys()),
        default=os.getenv("REID_PROFILE", "uav"),
        help=(
            "Re-ID preset profile (affects sampling and thresholds). "
            "Defaults to env REID_PROFILE or 'uav'."
        ),
    )
    parser.add_argument(
        "--feature-backend",
        choices=["classic", "sam3"],
        default=os.getenv("REID_FEATURE_BACKEND", "classic"),
        help=(
            "Feature backend to use: 'classic' (color+geometry+HOG) or "
            "'sam3' (SAM3-based embeddings). Defaults to env "
            "REID_FEATURE_BACKEND or 'classic'."
        ),
    )
    parser.add_argument(
        "--sam3-padding-fraction",
        type=float,
        default=None,
        help=(
            "When using --feature-backend sam3, controls how much to pad "
            "each bounding box on all sides before forming the SAM3 patch. "
            "If omitted, a profile-specific default is used (e.g. slightly "
            "larger for UAV than UGV)."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def _build_ls_client(ls_url: str, ls_api_key: str):
    if not ls_api_key or ls_api_key.strip() == "" or ls_api_key == "your_api_key":
        raise ReIDCLIError(
            "LABEL_STUDIO_API_KEY is required. "
            "Provide it via --ls-api-key or the LABEL_STUDIO_API_KEY env var."
        )

    # Keep env in sync so get_local_path can reuse it
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
        raise ReIDCLIError(f"Task {task_id} not found")

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
        raise ReIDCLIError(f"Annotation {annotation_id} not found")

    result = getattr(ann, "result", None)
    if not result:
        raise ReIDCLIError(f"Annotation {annotation_id} has no regions")

    logger.info(
        "Annotation fetched: id=%s with %d regions", getattr(ann, "id", annotation_id), len(result)
    )
    return ann


def _detect_video_key(task_data: Dict[str, Any]) -> Tuple[str, str]:
    """Heuristically detect the video field key in task data.

    Prefer common keys like 'video', otherwise fall back to the first
    string value that looks like a video file/URL.
    """
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

    raise ReIDCLIError(
        "Could not detect video field in task data. "
        "Ensure your task has a field like 'video' with a video URL/path."
    )


def _get_video_path(task: Dict[str, Any]) -> Tuple[str, str]:
    data = task.get("data") or {}
    key, video_url = _detect_video_key(data)
    logger.info("Using video field '%s' with URL %s", key, video_url)

    # Resolve relative URL if needed using LABEL_STUDIO_HOST/URL
    if not video_url.startswith("http") and video_url.startswith("/"):
        host = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL")
        if host:
            from urllib.parse import urljoin

            video_url = urljoin(host.rstrip("/"), video_url)
            logger.info("Resolved relative video URL to %s", video_url)

    logger.info("Downloading/caching video via get_local_path...")
    local_path = get_local_path(video_url, task_id=task["id"])
    if not os.path.exists(local_path):
        raise ReIDCLIError(f"Video file not found after download: {local_path}")

    size_mb = os.path.getsize(local_path) / 1024**2
    logger.info("Video cached at: %s (%.2f MB)", local_path, size_mb)
    return local_path, key


def _extract_tracks_from_results(results: Iterable[Dict[str, Any]]) -> List[TrackSequence]:
    tracks: List[TrackSequence] = []

    for res in results:
        if not isinstance(res, dict):
            continue
        value = res.get("value") or {}
        seq = value.get("sequence")
        if not isinstance(seq, list) or not seq:
            continue

        region_id = res.get("id")
        if not isinstance(region_id, str):
            logger.debug("Skipping result without string 'id': %r", res)
            continue

        # Interpret meta["text"] according to re-ID semantics:
        # - "id:<number>"          -> confirmed reference (merge_id = number)
        # - "id:" with no number   -> candidate (merge_id = None)
        # - "id:20 35 47" (multi)  -> previous prediction; scrub to "id:" and
        #                              treat as candidate (merge_id = None).
        merge_id: Optional[int] = None
        meta = res.get("meta")
        text_field = None
        if isinstance(meta, dict):
            text_field = meta.get("text")

        if isinstance(text_field, str):
            m = re.match(r"^\s*id\s*:(.*)$", text_field, flags=re.IGNORECASE)
            if m:
                rest = m.group(1).strip()
                if not rest:
                    # "id:" or "id:   " -> candidate placeholder
                    meta["text"] = "id:"
                else:
                    nums = re.findall(r"[0-9]+", rest)
                    if len(nums) == 1:
                        # Single confirmed ID from manual annotation
                        try:
                            merge_id = int(nums[0])
                        except ValueError:
                            merge_id = None
                    else:
                        # Multiple IDs from prior predictions; scrub and
                        # treat as a candidate for this run.
                        meta["text"] = "id:"

        # Fallback to the standard extraction logic for any other cases
        if merge_id is None:
            merge_id = extract_merge_key_from_result(res)
        frames_count = value.get("framesCount")
        duration = value.get("duration")

        track = TrackSequence(
            region_id=region_id,
            merge_id=merge_id,
            result=res,
            sequence=sorted(seq, key=lambda f: int(f.get("frame", 0))),
            frames_count=frames_count if isinstance(frames_count, int) else None,
            duration=float(duration) if isinstance(duration, (int, float)) else None,
        )
        tracks.append(track)

    logger.info("Extracted %d track(s) with sequence data", len(tracks))
    return tracks


def _select_sample_frames(track: TrackSequence, max_samples: int) -> List[Dict[str, Any]]:
    frames = [f for f in track.sequence if isinstance(f, dict) and f.get("enabled", True)]
    if not frames:
        return []
    if len(frames) <= max_samples:
        return frames

    indices = np.linspace(0, len(frames) - 1, num=max_samples, dtype=int)
    return [frames[i] for i in indices]


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array (H, W, 3) uint8 to HSV (H, W, 3) float32.

    H: [0, 180), S: [0, 256), V: [0, 256) to match OpenCV convention.
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc

    diff = maxc - minc
    s = np.where(maxc != 0, diff / maxc, 0)

    # Compute hue
    h = np.zeros_like(maxc)
    mask = diff != 0

    # Red is max
    mask_r = mask & (maxc == r)
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360

    # Green is max
    mask_g = mask & (maxc == g)
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360

    # Blue is max
    mask_b = mask & (maxc == b)
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360

    # Scale to OpenCV convention: H [0, 180), S [0, 256), V [0, 256)
    h = (h / 2.0).astype(np.float32)  # [0, 360) -> [0, 180)
    s = (s * 255).astype(np.float32)
    v = (v * 255).astype(np.float32)

    return np.stack([h, s, v], axis=-1)


def _compute_hist(crop_rgb: np.ndarray, bins: Tuple[int, int, int]) -> np.ndarray:
    """Compute HSV color histogram from RGB crop using numpy.

    Args:
        crop_rgb: RGB image array (H, W, 3) uint8
        bins: Number of bins for (H, S, V) channels

    Returns:
        Flattened, normalized histogram
    """
    hsv = _rgb_to_hsv(crop_rgb)

    # Define ranges matching OpenCV: H [0, 180), S [0, 256), V [0, 256)
    ranges = [(0, 180), (0, 256), (0, 256)]

    # Compute 3D histogram using numpy
    h_bins, s_bins, v_bins = bins
    h_vals = hsv[..., 0].flatten()
    s_vals = hsv[..., 1].flatten()
    v_vals = hsv[..., 2].flatten()

    hist, _ = np.histogramdd(
        np.stack([h_vals, s_vals, v_vals], axis=1),
        bins=[h_bins, s_bins, v_bins],
        range=ranges,
    )

    hist = hist.flatten().astype("float32")
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist


def _shape_descriptor(crop_rgb: np.ndarray, num_bins: int = 8) -> np.ndarray:
    """Compute a simple gradient-orientation histogram as a shape descriptor.

    This is loosely HOG-like: gradients are aggregated over the whole crop
    and normalized, making it relatively robust to pose and scale changes.

    Uses scipy.ndimage for gradient computation instead of cv2.
    """
    from scipy import ndimage

    # Convert to grayscale using luminosity formula
    gray = (
        0.299 * crop_rgb[..., 0].astype(np.float32)
        + 0.587 * crop_rgb[..., 1].astype(np.float32)
        + 0.114 * crop_rgb[..., 2].astype(np.float32)
    )

    # Compute gradients using Sobel filters
    gx = ndimage.sobel(gray, axis=1, mode='constant')  # horizontal gradient
    gy = ndimage.sobel(gray, axis=0, mode='constant')  # vertical gradient

    # Compute magnitude and angle
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)  # Returns angles in [-pi, pi]
    ang = (ang + np.pi) % (2 * np.pi)  # Shift to [0, 2*pi)

    # Use only non-zero gradients
    valid = mag > 0
    if not np.any(valid):
        return np.zeros(num_bins, dtype="float32")

    angles = ang[valid].flatten()
    weights = mag[valid].flatten().astype("float32")

    bin_edges = np.linspace(0.0, 2.0 * np.pi, num_bins + 1, dtype="float32")
    bin_indices = np.searchsorted(bin_edges, angles, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    hist = np.bincount(bin_indices, weights=weights, minlength=num_bins).astype(
        "float32"
    )
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist


def _geom_vector(item: Dict[str, Any]) -> np.ndarray:
    x = float(item.get("x", 0.0))
    y = float(item.get("y", 0.0))
    w = float(item.get("width", 0.0))
    h = float(item.get("height", 0.0))

    cx = (x + w * 0.5) / 100.0
    cy = (y + h * 0.5) / 100.0
    area = (w * h) / (100.0 * 100.0)
    aspect = w / (h + 1e-6)

    # Scale area and aspect to comparable ranges
    return np.array([cx, cy, area, aspect / 5.0], dtype="float32")


def _compute_track_features(
    video_path: str,
    tracks: List[TrackSequence],
    preset: ReIDPreset,
) -> Dict[str, TrackFeatures]:
    """Compute per-track appearance + geometry features.

    Frames are read lazily via random access using PyAV; only a bounded
    number of samples per track are processed.
    """
    # Build sampling map: frame_idx (0-based) -> list of (track_id, frame_item)
    frame_requests: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}
    for track in tracks:
        samples = _select_sample_frames(track, preset.max_samples_per_track)
        for item in samples:
            frame = int(item.get("frame", 0))
            if frame <= 0:
                continue
            frame_idx = frame - 1
            frame_requests.setdefault(frame_idx, []).append((track.region_id, item))

    if not frame_requests:
        logger.warning("No frames selected for feature extraction")
        return {}

    total_frames = len(frame_requests)
    logger.info(
        "Preparing to extract features from %d unique frame(s)", total_frames
    )

    # Prepare aggregators
    feat_hist: Dict[str, np.ndarray] = {}
    feat_geom: Dict[str, np.ndarray] = {}
    feat_shape: Dict[str, np.ndarray] = {}
    feat_count: Dict[str, int] = {}

    for idx, frame_idx in enumerate(sorted(frame_requests.keys()), 1):
        # Read frame via PyAV
        pil_frame = base._read_frame_pyav(video_path, frame_idx)
        if pil_frame is None:
            logger.warning("Failed to read frame %d from video", frame_idx)
            continue

        frame_rgb = np.array(pil_frame.convert('RGB'))
        h, w = frame_rgb.shape[:2]

        requests = frame_requests[frame_idx]
        for region_id, item in requests:
            x = float(item.get("x", 0.0)) / 100.0
            y = float(item.get("y", 0.0)) / 100.0
            bw = float(item.get("width", 0.0)) / 100.0
            bh = float(item.get("height", 0.0)) / 100.0

            x0 = max(0, min(w - 1, int(round(x * w))))
            y0 = max(0, min(h - 1, int(round(y * h))))
            x1 = max(0, min(w, int(round((x + bw) * w))))
            y1 = max(0, min(h, int(round((y + bh) * h))))
            if x1 <= x0 or y1 <= y0:
                continue

            crop = frame_rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            hist = _compute_hist(crop, preset.hist_bins)
            geom = _geom_vector(item)
            shape = _shape_descriptor(crop)

            if region_id not in feat_hist:
                feat_hist[region_id] = np.zeros_like(hist)
                feat_geom[region_id] = np.zeros_like(geom)
                feat_shape[region_id] = np.zeros_like(shape)
                feat_count[region_id] = 0

            feat_hist[region_id] += hist
            feat_geom[region_id] += geom
            feat_shape[region_id] += shape
            feat_count[region_id] += 1

        # Periodic progress feedback for long runs
        if idx % 50 == 0 or idx == total_frames:
            pct = 100.0 * float(idx) / float(total_frames)
            logger.info(
                "Feature extraction progress: %d/%d frames (%.1f%%)",
                idx,
                total_frames,
                pct,
            )

    features: Dict[str, TrackFeatures] = {}
    for region_id, count in feat_count.items():
        if count <= 0:
            continue
        hvec = feat_hist[region_id] / float(count)
        gvec = feat_geom[region_id] / float(count)
        svec = feat_shape[region_id] / float(count)
        features[region_id] = TrackFeatures(
            hist=hvec,
            geom=gvec,
            shape=svec,
            samples=count,
        )

    logger.info("Computed features for %d track(s)", len(features))
    return features


def _compute_track_features_sam3(
    video_path: str,
    tracks: List[TrackSequence],
    preset: ReIDPreset,
    padding_fraction: float,
) -> Dict[str, TrackFeatures]:
    """Compute per-track SAM3 embedding features.

    Each sampled bounding box is treated as a pseudo-image patch. For every
    frame, patches are embedded in a batch using SAM3 image encoder, then
    average-pooled spatially to a single vector per patch. Per-track vectors
    are averaged over all samples and L2-normalized.

    Migrated from SAM2 to SAM3 via HuggingFace transformers.
    """
    import torch

    # Build sampling map: frame_idx (0-based) -> list of (track_id, frame_item)
    frame_requests: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}
    for track in tracks:
        samples = _select_sample_frames(track, preset.max_samples_per_track)
        for item in samples:
            frame = int(item.get("frame", 0))
            if frame <= 0:
                continue
            frame_idx = frame - 1
            frame_requests.setdefault(frame_idx, []).append((track.region_id, item))

    if not frame_requests:
        logger.warning("No frames selected for SAM3 feature extraction")
        return {}

    total_frames = len(frame_requests)
    logger.info(
        "Preparing to extract SAM3 features from %d unique frame(s)", total_frames
    )

    # Get SAM3 image model for embeddings
    sam3_model, sam3_processor = base._get_sam3_image_model()

    # Aggregators for per-track embeddings
    emb_sum: Dict[str, np.ndarray] = {}
    emb_count: Dict[str, int] = {}

    for idx, frame_idx in enumerate(sorted(frame_requests.keys()), 1):
        # Read frame via PyAV
        pil_frame = base._read_frame_pyav(video_path, frame_idx)
        if pil_frame is None:
            logger.warning("Failed to read frame %d from video", frame_idx)
            continue

        frame_rgb = np.array(pil_frame.convert('RGB'))
        h, w = frame_rgb.shape[:2]

        requests = frame_requests[frame_idx]
        patches: List[Image.Image] = []
        owners: List[str] = []

        for region_id, item in requests:
            x = float(item.get("x", 0.0)) / 100.0
            y = float(item.get("y", 0.0)) / 100.0
            bw = float(item.get("width", 0.0)) / 100.0
            bh = float(item.get("height", 0.0)) / 100.0

            x0 = x * w
            y0 = y * h
            x1 = (x + bw) * w
            y1 = (y + bh) * h

            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            box_w = x1 - x0
            box_h = y1 - y0
            if box_w <= 1 or box_h <= 1:
                continue

            # Apply symmetric padding where possible, without going
            # outside the image bounds.
            pad_w = box_w * padding_fraction * 0.5
            pad_h = box_h * padding_fraction * 0.5
            x0_exp = max(0.0, cx - (box_w * 0.5 + pad_w))
            y0_exp = max(0.0, cy - (box_h * 0.5 + pad_h))
            x1_exp = min(float(w), cx + (box_w * 0.5 + pad_w))
            y1_exp = min(float(h), cy + (box_h * 0.5 + pad_h))

            ix0 = int(round(x0_exp))
            iy0 = int(round(y0_exp))
            ix1 = int(round(x1_exp))
            iy1 = int(round(y1_exp))
            if ix1 <= ix0 or iy1 <= iy0:
                continue

            patch_arr = frame_rgb[iy0:iy1, ix0:ix1]
            if patch_arr.size == 0:
                continue

            patch_pil = Image.fromarray(patch_arr)
            patches.append(patch_pil)
            owners.append(region_id)

        if not patches:
            continue

        # Batch all patches in this frame for embedding extraction
        # Process patches through SAM3 and get image embeddings
        try:
            inputs = sam3_processor(
                images=patches,
                return_tensors="pt",
            ).to(base.DEVICE)

            with torch.inference_mode():
                vision_output = sam3_model.get_vision_features(
                    pixel_values=inputs.pixel_values
                )

            # fpn_hidden_states[0]: highest-res FPN feature (B, C, H, W)
            feat = vision_output.fpn_hidden_states[0]
            if hasattr(feat, "detach"):
                emb_batch = feat.mean(dim=(2, 3)).detach().cpu().float().numpy()
            else:
                emb_batch = feat.mean(axis=(2, 3))

            for region_id, vec in zip(owners, emb_batch):
                v = vec.astype("float32")
                if region_id not in emb_sum:
                    emb_sum[region_id] = np.zeros_like(v)
                    emb_count[region_id] = 0
                emb_sum[region_id] += v
                emb_count[region_id] += 1

        except Exception as exc:
            logger.warning(
                "SAM3 embedding extraction failed on frame %d: %s",
                frame_idx,
                exc,
            )
            continue

        # Periodic progress feedback for long runs
        if idx % 50 == 0 or idx == total_frames:
            pct = 100.0 * float(idx) / float(total_frames)
            logger.info(
                "SAM3 feature extraction progress: %d/%d frames (%.1f%%)",
                idx,
                total_frames,
                pct,
            )

    features: Dict[str, TrackFeatures] = {}
    for region_id, count in emb_count.items():
        if count <= 0:
            continue
        mean_vec = emb_sum[region_id] / float(count)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0.0:
            mean_vec /= norm
        features[region_id] = TrackFeatures(
            hist=np.zeros(1, dtype="float32"),
            geom=np.zeros(1, dtype="float32"),
            shape=np.zeros(1, dtype="float32"),
            samples=count,
            sam3_vec=mean_vec.astype("float32"),
        )

    logger.info("Computed SAM3 features for %d track(s)", len(features))
    return features


def _hist_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(np.minimum(h1, h2).sum())


def _geom_similarity(g1: np.ndarray, g2: np.ndarray) -> float:
    dist = float(np.linalg.norm(g1 - g2))
    return 1.0 / (1.0 + dist)


def _compute_similarity(
    ref_feat: TrackFeatures,
    cand_feat: TrackFeatures,
    preset: ReIDPreset,
) -> float:
    sim_color = _hist_intersection(ref_feat.hist, cand_feat.hist)
    sim_geom = _geom_similarity(ref_feat.geom, cand_feat.geom)
    sim_shape = _hist_intersection(ref_feat.shape, cand_feat.shape)

    w_color = preset.color_weight
    w_geom = preset.geom_weight
    w_shape = preset.shape_weight
    w_sum = w_color + w_geom + w_shape
    if w_sum <= 0:
        # Fallback: equal weighting if misconfigured
        w_color = w_geom = w_shape = 1.0 / 3.0
        w_sum = 1.0

    w_color /= w_sum
    w_geom /= w_sum
    w_shape /= w_sum

    sim = w_color * sim_color + w_geom * sim_geom + w_shape * sim_shape
    return max(0.0, min(1.0, sim))


def _compute_similarity_sam3(
    ref_track: TrackSequence,
    ref_feat: TrackFeatures,
    cand_track: TrackSequence,
    cand_feat: TrackFeatures,
    center_frames: Dict[str, float],
) -> float:
    """Compute similarity for SAM3 backend using cosine similarity only.

    Cosine similarity between SAM3 embeddings is mapped from [-1, 1] to
    [0, 1]. No temporal term is applied; all time-awareness comes from the
    distribution of available reference/candidate tracks, not an explicit
    decay factor.
    """

    if ref_feat.sam3_vec is None or cand_feat.sam3_vec is None:
        return 0.0

    v1 = ref_feat.sam3_vec.astype("float32")
    v2 = cand_feat.sam3_vec.astype("float32")
    if v1.shape != v2.shape or v1.size == 0:
        return 0.0

    # Guard against zero-norm vectors (uninformative features)
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    cos = float(np.dot(v1, v2))
    # Numerical safety
    cos = max(-1.0, min(1.0, cos))
    sim_embed = 0.5 * (cos + 1.0)  # map to [0, 1]
    return max(0.0, min(1.0, sim_embed))


def _build_reid_matches(
    tracks: List[TrackSequence],
    features: Dict[str, TrackFeatures],
    preset: ReIDPreset,
    feature_backend: str,
) -> Dict[str, Dict[str, Any]]:
    """For each candidate track, compute top-3 reference matches and confidences.

    Returns mapping: region_id -> {
        "refs": List[Tuple[TrackSequence, float]],  # sorted by similarity desc
        "decisive": bool,
        "confidence": float,
    }
    """
    refs = [t for t in tracks if t.merge_id is not None]
    cands = [t for t in tracks if t.merge_id is None]

    logger.info("Reference tracks: %d, candidate tracks: %d", len(refs), len(cands))

    # Pre-compute simple time centers for all tracks
    center_frames: Dict[str, float] = {}
    for t in tracks:
        start, end = t.frame_range
        center_frames[t.region_id] = 0.5 * float(start + end)

    # Only consider tracks with computed features
    def has_feat(t: TrackSequence) -> bool:
        f = features.get(t.region_id)
        if f is None or f.samples <= 0:
            return False
        if feature_backend == "sam3":
            return f.sam3_vec is not None
        return True

    refs = [t for t in refs if has_feat(t)]
    cands = [t for t in cands if has_feat(t)]

    if not refs:
        logger.warning("No reference tracks with valid features; skipping re-ID")
        return {}

    matches: Dict[str, Dict[str, Any]] = {}

    for cand in cands:
        cand_feat = features[cand.region_id]

        # First compute similarities to all reference tracks with features.
        scored_all: List[Tuple[TrackSequence, float]] = []
        for ref in refs:
            ref_feat = features.get(ref.region_id)
            if ref_feat is None:
                continue
            if feature_backend == "sam3":
                sim = _compute_similarity_sam3(
                    ref_track=ref,
                    ref_feat=ref_feat,
                    cand_track=cand,
                    cand_feat=cand_feat,
                    center_frames=center_frames,
                )
            else:
                sim = _compute_similarity(ref_feat, cand_feat, preset)
            scored_all.append((ref, sim))

        if not scored_all:
            # No usable references for this candidate
            continue

        # Keep all matches above the configured threshold when possible.
        scored = [(ref, sim) for (ref, sim) in scored_all if sim >= preset.min_sim_for_match]
        if not scored:
            # Fallback: even if all similarities are low, keep the top few
            # so that every candidate still has at least one suggested match.
            scored_all.sort(key=lambda rs: rs[1], reverse=True)
            scored = scored_all[:3]
        else:
            scored.sort(key=lambda rs: rs[1], reverse=True)

        # Deduplicate by merge_id (or region_id as a fallback) so that a
        # single identity appears at most once in the candidate's top list.
        dedup_scored: List[Tuple[TrackSequence, float]] = []
        seen_keys = set()
        for ref, sim in scored:
            key = ref.merge_id if ref.merge_id is not None else ref.region_id
            if key in seen_keys:
                continue
            seen_keys.add(key)
            dedup_scored.append((ref, sim))

        if not dedup_scored:
            continue

        top_refs = dedup_scored[:3]
        sims = [s for (_, s) in top_refs]
        s1 = sims[0]
        s2 = sims[1] if len(sims) > 1 else 0.0

        decisive = bool(
            s1 >= preset.decisive_min_top1 and (s1 - s2) >= preset.decisive_margin
        )

        # Region-level confidence is always the lowest similarity among the
        # available top-k matches (k <= 3), as requested.
        confidence = float(min(sims))

        matches[cand.region_id] = {
            "refs": top_refs,
            "decisive": decisive,
            "confidence": confidence,
        }

    logger.info("Built re-ID suggestions for %d candidate track(s)", len(matches))
    return matches


def _update_results_with_matches(
    tracks: List[TrackSequence],
    matches: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Produce a new result list with updated meta.text and per-region scores."""
    new_results: List[Dict[str, Any]] = []

    for track in tracks:
        res = track.result
        # Work on a shallow copy of the region dict to avoid mutating original
        updated = dict(res)
        meta = dict(updated.get("meta") or {})

        match = matches.get(track.region_id)
        if match is not None:
            # Candidate track: attach suggested IDs and confidence.
            top_refs: List[Tuple[TrackSequence, float]] = match["refs"]
            decisive: bool = match["decisive"]
            confidence: float = match["confidence"]

            # Collect numeric IDs from matched references, ordered by similarity.
            merge_ids: List[int] = []
            for ref, _sim in top_refs:
                if ref.merge_id is None:
                    continue
                merge_ids.append(ref.merge_id)

            if merge_ids:
                if decisive or len(merge_ids) == 1:
                    # Decisive case or only one ID: keep a single, clean marker.
                    meta["text"] = f"id:{merge_ids[0]}"
                else:
                    # Non-decisive: keep all suggestions on a single line in
                    # decreasing order of similarity, e.g. "id:20 35 47".
                    meta["text"] = "id:" + " ".join(str(mid) for mid in merge_ids)

                updated["meta"] = meta
                updated["score"] = float(confidence)
        else:
            # Reference track (manual ID from user): keep its original meta.text
            # but mark it as fully confident.
            if track.merge_id is not None:
                updated["score"] = 1.0

        new_results.append(updated)

    return new_results


def _build_prediction_payload(
    results: List[Dict[str, Any]],
    matches: Dict[str, Dict[str, Any]],
    annotation_id: int,
    profile: str,
) -> Dict[str, Any]:
    if matches:
        confidences = [v["confidence"] for v in matches.values()]
        overall_score = float(min(confidences))
    else:
        overall_score = 0.0

    model_version = f"complete-reid-{profile}-ann-{annotation_id}"
    prediction = {
        "result": results,
        "score": overall_score,
        "model_version": model_version,
    }
    return prediction


def _upload_prediction(ls, task_id: int, prediction: Dict[str, Any]):
    logger.info(
        "Uploading re-ID prediction for task %s (model_version=%s, regions=%d)...",
        task_id,
        prediction.get("model_version"),
        len(prediction.get("result", [])),
    )

    result = ls.predictions.create(
        task=task_id,
        score=prediction.get("score", 0.0),
        model_version=prediction.get("model_version", "complete-reid"),
        result=prediction.get("result", []),
    )

    pred_id = getattr(result, "id", None)
    if pred_id is not None:
        logger.info("Upload complete, prediction id=%s", pred_id)
    else:
        logger.info("Upload request completed (no prediction id in response)")

    return result


def main() -> None:
    args = _parse_args()

    # Set global log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.profile not in PRESETS:
        raise ReIDCLIError(f"Unknown profile '{args.profile}'. Choose from: {sorted(PRESETS.keys())}")
    preset = PRESETS[args.profile]

    # Derive SAM3 padding fraction: CLI flag wins, otherwise use a
    # profile-specific default when the SAM3 backend is selected.
    sam3_padding_fraction = args.sam3_padding_fraction
    if sam3_padding_fraction is None:
        if args.profile == "uav":
            sam3_padding_fraction = 0.25
        elif args.profile == "ugv":
            sam3_padding_fraction = 0.2
        else:
            sam3_padding_fraction = 0.2

    logger.info("=" * 80)
    logger.info("COMPLETE RE-ID CLI STARTED")
    logger.info("=" * 80)
    logger.info("Parameters:")
    logger.info("   Label Studio URL: %s", args.ls_url)
    logger.info("   Project ID: %s", args.project)
    logger.info("   Task ID: %s", args.task)
    logger.info("   Annotation ID: %s", args.annotation)
    logger.info("   Profile: %s", args.profile)
    logger.info("   Feature backend: %s", args.feature_backend)
    if args.feature_backend == "sam3":
        logger.info("   SAM3 padding fraction: %.3f", sam3_padding_fraction)
    logger.info("=" * 80)

    exit_code = 0

    try:
        ls = _build_ls_client(args.ls_url, args.ls_api_key)

        task = _fetch_task(ls, args.project, args.task)
        ann = _fetch_annotation(ls, args.annotation)

        video_path, _video_key = _get_video_path(task)

        # Merge reference sequences that share the same numeric id:<#> into
        # single tracks, then extract TrackSequence objects from the merged
        # results. This prevents duplicates like multiple separate regions
        # all labeled "id:20".
        raw_results = getattr(ann, "result", []) or []
        merged_results = merge_results_by_merge_id(raw_results)

        # Extract tracks from merged annotation results
        tracks = _extract_tracks_from_results(merged_results)
        if not tracks:
            logger.warning("No video rectangle sequences found in annotation; nothing to do")
            return

        # Compute features according to the selected backend
        if args.feature_backend == "sam3":
            features = _compute_track_features_sam3(
                video_path,
                tracks,
                preset,
                sam3_padding_fraction,
            )
        else:
            features = _compute_track_features(video_path, tracks, preset)
        if not features:
            logger.warning("Failed to compute features for any track; aborting re-ID")
            return

        # Build matches for candidate tracks
        matches = _build_reid_matches(tracks, features, preset, args.feature_backend)
        if not matches:
            logger.warning("No candidate tracks received confident matches; uploading copy-only prediction")

        # Build updated results and prediction payload
        updated_results = _update_results_with_matches(tracks, matches)
        prediction = _build_prediction_payload(updated_results, matches, args.annotation, args.profile)
        _upload_prediction(ls, args.task, prediction)

        logger.info("=" * 80)
        logger.info("COMPLETE RE-ID CLI EXECUTION SUCCESSFUL")
        logger.info("=" * 80)

    except ReIDCLIError as e:
        logger.error("Re-ID CLI error: %s", e)
        exit_code = 1
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        exit_code = 130
    except Exception as e:  # pragma: no cover - unexpected errors
        logger.error("Unexpected error: %s", e, exc_info=True)
        exit_code = 1
    finally:
        if exit_code != 0:
            logger.info("=" * 80)
            logger.info("COMPLETE RE-ID CLI EXECUTION FAILED (exit code: %s)", exit_code)
            logger.info("=" * 80)

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
