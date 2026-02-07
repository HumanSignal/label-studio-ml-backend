"""
Refine bounding boxes of video rectangle sequences using SAM3 text+box prompts.

For each keyframe box, expands it by a search scale factor, runs SAM3 segmentation
with combined text prompt (from track label) and box prompt, and extracts a tight
bounding box from the resulting mask.

This approach handles both cases where the original box is too large OR too small:
- The expanded box ensures the target is likely contained
- The text prompt tells SAM3 WHAT to segment (e.g., "person")

Migrated from SAM2 to SAM3 via HuggingFace Transformers.
Video decoding uses PyAV (no cv2 dependency).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

import seeding_common as base
from complete_reid import (
    ReIDCLIError,
    TrackSequence,
    _build_ls_client,
    _fetch_annotation,
    _fetch_task,
    _get_video_path,
    _extract_tracks_from_results,
    _upload_prediction,
)


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine bounding boxes of video rectangle sequences in a single "
            "Label Studio annotation using SAM3 text+box prompts and upload "
            "the result as a prediction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (inside Docker):\n\n"
            "  docker compose exec segment_anything_3_video bash -lc '" "\\n"
            "    export LABEL_STUDIO_HOST=https://app.heartex.com &&" "\\n"
            "    export LABEL_STUDIO_URL=https://app.heartex.com &&" "\\n"
            "    export LABEL_STUDIO_API_KEY=\"$LABEL_STUDIO_API_KEY\" &&" "\\n"
            "    python /app/adjust_bboxes_sam3.py \\\n"
            "      --ls-url https://app.heartex.com \\\n"
            "      --ls-api-key \"$LABEL_STUDIO_API_KEY\" \\\n"
            "      --project 123 \\\n"
            "      --task 456 \\\n"
            "      --annotation 789 \\\n"
            "      --search-scale 1.3 \\\n"
            "      --default-label person\n" "\\n"
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
        help="Annotation ID whose bounding boxes will be refined",
    )
    parser.add_argument(
        "--search-scale",
        type=float,
        default=1.3,
        help=(
            "Scale factor for the SAM3 search region around each box. A value "
            "of 1.3 means the search box is ~30%% larger in width/height, "
            "centered on the original box, and clamped to the image bounds. "
            "This helps when boxes have drifted smaller than the target. "
            "Default: 1.3"
        ),
    )
    parser.add_argument(
        "--default-label",
        type=str,
        default="person",
        help=(
            "Default text prompt to use when a track has no label. "
            "This tells SAM3 what object to segment. Default: 'person'"
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def _get_track_label(track: TrackSequence, default_label: str) -> str:
    """Extract the label from a track, falling back to default if not found."""
    result = track.result
    if not isinstance(result, dict):
        return default_label

    value = result.get("value", {})
    labels = value.get("labels", [])

    if isinstance(labels, list) and labels:
        # Use first label
        label = labels[0]
        if isinstance(label, str) and label.strip():
            return label.strip()
    elif isinstance(labels, str) and labels.strip():
        return labels.strip()

    return default_label


def _build_frame_requests(
    tracks: List[TrackSequence],
    default_label: str,
) -> Dict[int, List[Tuple[TrackSequence, Dict[str, Any], str]]]:
    """Build a map of frame_idx -> list of (track, item, label) tuples to process."""
    frame_requests: Dict[int, List[Tuple[TrackSequence, Dict[str, Any], str]]] = {}
    for track in tracks:
        label = _get_track_label(track, default_label)
        for item in track.sequence:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("enabled", True)):
                continue
            frame = int(item.get("frame", 0))
            if frame <= 0:
                continue
            frame_idx = frame - 1  # 1-based to 0-based
            frame_requests.setdefault(frame_idx, []).append((track, item, label))
    return frame_requests


def _refine_bboxes_with_sam3(
    video_path: str,
    tracks: List[TrackSequence],
    search_scale: float,
    default_label: str,
) -> None:
    """Refine bounding boxes using SAM3 text+box prompt segmentation.

    Uses combined text and box prompts for robust segmentation:
    - Text prompt (e.g., "person") tells SAM3 WHAT to segment
    - Expanded box prompt tells SAM3 WHERE to look

    This handles both too-large and too-small boxes.
    """
    search_scale = float(search_scale)
    if search_scale <= 0:
        raise ReIDCLIError(f"search_scale must be > 0, got {search_scale}")

    frame_requests = _build_frame_requests(tracks, default_label)
    if not frame_requests:
        logger.warning("No frames found in any sequence; nothing to refine")
        return

    total_frames = len(frame_requests)
    logger.info(
        "Preparing to refine bounding boxes on %d unique frame(s) using SAM3 text+box prompts",
        total_frames,
    )

    # Get SAM3 image model for text+box prompted segmentation
    sam3_model, sam3_processor = base._get_sam3_image_model()

    # Get video dimensions
    width, height, frames_count, fps = base._get_video_info_pyav(video_path)

    frame_indices = sorted(frame_requests.keys())
    if tqdm is None:
        frame_iter = frame_indices
    else:
        frame_iter = tqdm(frame_indices, total=total_frames, desc="Refining frames", unit="frame")

    for idx, frame_idx in enumerate(frame_iter, 1):
        # Read frame via PyAV
        pil_frame = base._read_frame_pyav(video_path, frame_idx)
        if pil_frame is None:
            logger.warning("Failed to read frame %d from video", frame_idx)
            continue

        w, h = pil_frame.size

        for track, item, label in frame_requests[frame_idx]:
            x_pct = float(item.get("x", 0.0))
            y_pct = float(item.get("y", 0.0))
            w_pct = float(item.get("width", 0.0))
            h_pct = float(item.get("height", 0.0))

            x0 = (x_pct / 100.0) * w
            y0 = (y_pct / 100.0) * h
            bw = (w_pct / 100.0) * w
            bh = (h_pct / 100.0) * h
            if bw <= 1 or bh <= 1:
                continue

            x1 = x0 + bw
            y1 = y0 + bh

            # Center of the original box
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)

            # Expanded search region with fixed scale around the center,
            # clamped to image bounds on each side.
            half_w = 0.5 * search_scale * bw
            half_h = 0.5 * search_scale * bh

            x0_search = max(0.0, cx - half_w)
            y0_search = max(0.0, cy - half_h)
            x1_search = min(float(w), cx + half_w)
            y1_search = min(float(h), cy + half_h)

            ix0 = int(round(x0_search))
            iy0 = int(round(y0_search))
            ix1 = int(round(x1_search))
            iy1 = int(round(y1_search))
            if ix1 <= ix0 or iy1 <= iy0:
                continue

            box = [ix0, iy0, ix1, iy1]

            try:
                # Run SAM3 with combined text + box prompt
                # text: tells SAM3 WHAT to segment (e.g., "person")
                # input_boxes: tells SAM3 WHERE to look (expanded region)
                # input_boxes_labels: 1 = positive box (segment within this region)
                inputs = sam3_processor(
                    images=pil_frame,
                    text=label,
                    input_boxes=[[box]],
                    input_boxes_labels=[[1]],  # 1 = positive box
                    return_tensors="pt",
                ).to(base.DEVICE)

                with torch.inference_mode():
                    outputs = sam3_model(**inputs)

                # Post-process using instance segmentation (for text+box prompts)
                results = sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]

                masks = results.get("masks", [])
                scores = results.get("scores", [])
                boxes = results.get("boxes", [])

                if len(masks) == 0:
                    logger.debug(
                        "No masks returned for frame %d region %s with label '%s'",
                        frame_idx,
                        track.region_id,
                        label,
                    )
                    continue

                # Select best mask by confidence score
                if len(scores) > 0:
                    best_idx = int(np.argmax([s.item() if hasattr(s, 'item') else s for s in scores]))
                else:
                    best_idx = 0

                # Use the returned box if available, otherwise extract from mask
                if len(boxes) > best_idx:
                    best_box = boxes[best_idx]
                    if hasattr(best_box, 'tolist'):
                        best_box = best_box.tolist()
                    x0_tight, y0_tight, x1_tight, y1_tight = best_box
                else:
                    # Fall back to extracting bbox from mask
                    mask = masks[best_idx]
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()

                    ys, xs = np.where(mask > 0)
                    if xs.size == 0 or ys.size == 0:
                        continue

                    x0_tight = int(xs.min())
                    x1_tight = int(xs.max()) + 1
                    y0_tight = int(ys.min())
                    y1_tight = int(ys.max()) + 1

                # Clamp to image bounds
                x0_tight = max(0, min(w - 1, int(x0_tight)))
                y0_tight = max(0, min(h - 1, int(y0_tight)))
                x1_tight = max(0, min(w, int(x1_tight)))
                y1_tight = max(0, min(h, int(y1_tight)))

                if x1_tight <= x0_tight or y1_tight <= y0_tight:
                    continue

                # Convert back to LS percent coordinates
                new_x_pct = (x0_tight / float(w)) * 100.0
                new_y_pct = (y0_tight / float(h)) * 100.0
                new_w_pct = ((x1_tight - x0_tight) / float(w)) * 100.0
                new_h_pct = ((y1_tight - y0_tight) / float(h)) * 100.0

                item["x"] = new_x_pct
                item["y"] = new_y_pct
                item["width"] = new_w_pct
                item["height"] = new_h_pct

            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "SAM3 predict failed on frame %d region %s with label '%s': %s",
                    frame_idx,
                    track.region_id,
                    label,
                    exc,
                )
                continue

        # Periodic progress feedback for long runs
        if tqdm is None:
            if idx % 50 == 0 or idx == total_frames:
                pct = 100.0 * float(idx) / float(total_frames)
                logger.info(
                    "SAM3 bbox refinement progress: %d/%d frames (%.1f%%)",
                    idx,
                    total_frames,
                    pct,
                )


def _build_prediction_payload(
    results: List[Dict[str, Any]],
    annotation_id: int,
    search_scale: float,
) -> Dict[str, Any]:
    model_version = f"adjust-bboxes-sam3-scale-{search_scale:.2f}-ann-{annotation_id}"
    prediction = {
        "result": results,
        "score": 1.0,
        "model_version": model_version,
    }
    return prediction


def main() -> None:
    args = _parse_args()

    # Set global log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 80)
    logger.info("ADJUST BBOXES SAM3 CLI STARTED")
    logger.info("=" * 80)
    logger.info("Parameters:")
    logger.info("   Label Studio URL: %s", args.ls_url)
    logger.info("   Project ID: %s", args.project)
    logger.info("   Task ID: %s", args.task)
    logger.info("   Annotation ID: %s", args.annotation)
    logger.info("   Search scale: %.3f", args.search_scale)
    logger.info("   Default label: %s", args.default_label)
    logger.info("=" * 80)

    exit_code = 0

    try:
        ls = _build_ls_client(args.ls_url, args.ls_api_key)

        task = _fetch_task(ls, args.project, args.task)
        ann = _fetch_annotation(ls, args.annotation)

        video_path, _video_key = _get_video_path(task)

        raw_results = getattr(ann, "result", []) or []
        tracks = _extract_tracks_from_results(raw_results)
        if not tracks:
            logger.warning("No video rectangle sequences found in annotation; nothing to do")
            return

        _refine_bboxes_with_sam3(video_path, tracks, args.search_scale, args.default_label)

        prediction = _build_prediction_payload(raw_results, args.annotation, args.search_scale)
        _upload_prediction(ls, args.task, prediction)

        logger.info("=" * 80)
        logger.info("ADJUST BBOXES SAM3 CLI EXECUTION SUCCESSFUL")
        logger.info("=" * 80)

    except ReIDCLIError as e:
        logger.error("Adjust bboxes CLI error: %s", e)
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
            logger.info(
                "ADJUST BBOXES SAM3 CLI EXECUTION FAILED (exit code: %s)",
                exit_code,
            )
            logger.info("=" * 80)

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
