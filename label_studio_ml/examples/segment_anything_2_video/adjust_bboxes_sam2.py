from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from complete_reid import (  # type: ignore[import]
    ReIDCLIError,
    TrackSequence,
    _build_ls_client,
    _fetch_annotation,
    _fetch_task,
    _get_sam2_predictor,
    _get_video_path,
    _extract_tracks_from_results,
    _upload_prediction,
)


logger = logging.getLogger(__name__)


class _SAM2SetImageLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.INFO:
            return True
        if record.name != "root":
            return True
        if record.funcName != "set_image":
            return True

        msg = record.getMessage()
        if msg.startswith("For numpy array image, we assume"):
            return False
        if msg.startswith("Computing image embeddings for the provided image"):
            return False
        if msg.startswith("Image embeddings computed"):
            return False
        return True

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

_root_logger = logging.getLogger()
for _handler in _root_logger.handlers:
    _handler.addFilter(_SAM2SetImageLogFilter())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine bounding boxes of video rectangle sequences in a single "
            "Label Studio annotation using SAM2 masks and upload the result "
            "as a prediction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (inside Docker):\n\n"
            "  docker compose exec segment_anything_2_video bash -lc '" "\\n"
            "    export LABEL_STUDIO_HOST=https://app.heartex.com &&" "\\n"
            "    export LABEL_STUDIO_URL=https://app.heartex.com &&" "\\n"
            "    export LABEL_STUDIO_API_KEY=\"$LABEL_STUDIO_API_KEY\" &&" "\\n"
            "    python /app/adjust_bboxes_sam2.py \\\n"
            "      --ls-url https://app.heartex.com \\\n"
            "      --ls-api-key \"$LABEL_STUDIO_API_KEY\" \\\n"
            "      --project 123 \\\n"
            "      --task 456 \\\n"
            "      --annotation 789 \\\n"
            "      --search-scale 1.2\n" "\\n"
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
        default=1.2,
        help=(
            "Scale factor for the SAM2 search region around each box. A value "
            "of 1.2 means the search box is ~20%% larger in width/height, "
            "centered on the original box, and clamped to the image bounds."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def _build_frame_requests(tracks: List[TrackSequence]) -> Dict[int, List[Tuple[TrackSequence, Dict[str, Any]]]]:
    frame_requests: Dict[int, List[Tuple[TrackSequence, Dict[str, Any]]]] = {}
    for track in tracks:
        for item in track.sequence:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("enabled", True)):
                continue
            frame = int(item.get("frame", 0))
            if frame <= 0:
                continue
            frame_idx = frame - 1
            frame_requests.setdefault(frame_idx, []).append((track, item))
    return frame_requests


def _refine_bboxes_with_sam2(
    video_path: str,
    tracks: List[TrackSequence],
    search_scale: float,
) -> None:
    search_scale = float(search_scale)
    if search_scale <= 0:
        raise ReIDCLIError(f"search_scale must be > 0, got {search_scale}")

    frame_requests = _build_frame_requests(tracks)
    if not frame_requests:
        logger.warning("No frames found in any sequence; nothing to refine")
        return

    total_frames = len(frame_requests)
    logger.info(
        "Preparing to refine bounding boxes on %d unique frame(s) using SAM2",
        total_frames,
    )

    predictor = _get_sam2_predictor()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ReIDCLIError(f"Could not open video file: {video_path}")

    try:
        frame_indices = sorted(frame_requests.keys())
        if tqdm is None:
            frame_iter = frame_indices
        else:
            frame_iter = tqdm(frame_indices, total=total_frames, desc="Refining frames", unit="frame")

        for idx, frame_idx in enumerate(frame_iter, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame_bgr = cap.read()
            if not success or frame_bgr is None:
                logger.warning("Failed to read frame %d from video", frame_idx)
                continue

            h, w, _ = frame_bgr.shape
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Compute image embeddings once per frame, then reuse them for
            # multiple box prompts via predictor.predict(...).
            predictor.set_image(frame_rgb)

            for track, item in frame_requests[frame_idx]:
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

                box = np.array([ix0, iy0, ix1, iy1], dtype=np.float32)

                try:
                    masks, scores, _ = predictor.predict(
                        box=box,
                        multimask_output=True,
                        return_logits=False,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "SAM2 predict failed on frame %d region %s: %s",
                        frame_idx,
                        track.region_id,
                        exc,
                    )
                    continue

                if masks.size == 0 or scores.size == 0:
                    continue

                best_idx = int(np.argmax(scores))
                mask = masks[best_idx]

                ys, xs = np.where(mask > 0)
                if xs.size == 0 or ys.size == 0:
                    # No positive pixels; leave box unchanged
                    continue

                x0_tight = int(xs.min())
                x1_tight = int(xs.max()) + 1
                y0_tight = int(ys.min())
                y1_tight = int(ys.max()) + 1

                x0_tight = max(0, min(w - 1, x0_tight))
                y0_tight = max(0, min(h - 1, y0_tight))
                x1_tight = max(0, min(w, x1_tight))
                y1_tight = max(0, min(h, y1_tight))
                if x1_tight <= x0_tight or y1_tight <= y0_tight:
                    continue

                new_x_pct = (x0_tight / float(w)) * 100.0
                new_y_pct = (y0_tight / float(h)) * 100.0
                new_w_pct = ((x1_tight - x0_tight) / float(w)) * 100.0
                new_h_pct = ((y1_tight - y0_tight) / float(h)) * 100.0

                item["x"] = new_x_pct
                item["y"] = new_y_pct
                item["width"] = new_w_pct
                item["height"] = new_h_pct

            # Periodic progress feedback for long runs
            if tqdm is None:
                if idx % 50 == 0 or idx == total_frames:
                    pct = 100.0 * float(idx) / float(total_frames)
                    logger.info(
                        "SAM2 bbox refinement progress: %d/%d frames (%.1f%%)",
                        idx,
                        total_frames,
                        pct,
                    )
    finally:
        cap.release()


def _build_prediction_payload(
    results: List[Dict[str, Any]],
    annotation_id: int,
    search_scale: float,
) -> Dict[str, Any]:
    model_version = f"adjust-bboxes-sam2-scale-{search_scale:.2f}-ann-{annotation_id}"
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
    logger.info("🎯 ADJUST BBOXES SAM2 CLI STARTED")
    logger.info("=" * 80)
    logger.info("📋 Parameters:")
    logger.info("   • Label Studio URL: %s", args.ls_url)
    logger.info("   • Project ID: %s", args.project)
    logger.info("   • Task ID: %s", args.task)
    logger.info("   • Annotation ID: %s", args.annotation)
    logger.info("   • Search scale: %.3f", args.search_scale)
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

        _refine_bboxes_with_sam2(video_path, tracks, args.search_scale)

        prediction = _build_prediction_payload(raw_results, args.annotation, args.search_scale)
        _upload_prediction(ls, args.task, prediction)

        logger.info("=" * 80)
        logger.info("✅ ADJUST BBOXES SAM2 CLI EXECUTION SUCCESSFUL")
        logger.info("=" * 80)

    except ReIDCLIError as e:
        logger.error("❌ Adjust bboxes CLI error: %s", e)
        exit_code = 1
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        exit_code = 130
    except Exception as e:  # pragma: no cover - unexpected errors
        logger.error("❌ Unexpected error: %s", e, exc_info=True)
        exit_code = 1
    finally:
        if exit_code != 0:
            logger.info("=" * 80)
            logger.info(
                "❌ ADJUST BBOXES SAM2 CLI EXECUTION FAILED (exit code: %s)",
                exit_code,
            )
            logger.info("=" * 80)

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
