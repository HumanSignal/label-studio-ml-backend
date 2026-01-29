"""
CLI: Track objects across a video using SAM2 bidirectional propagation with manual merge.

This workflow keeps forward+backward results for each seed in the same region and
leaves cross-seed merging to manual ID assignment plus mergevideoregions.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import uuid
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

import initial_seeding_video_boxes as base_boxes
import seeding_common as base
from seeding_common import InitialSeedingError

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _parse_track_ids(raw: Optional[str]) -> Optional[Set[str]]:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    return set(parts)


def _mask_logits_to_box_and_score(mask_logits: torch.Tensor, threshold: float = 0.0) -> Tuple[Optional[np.ndarray], Optional[float]]:
    mask = (mask_logits > threshold)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        return None, None
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None, None
    box = np.array([xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item()], dtype=np.float32)

    probs = torch.sigmoid(mask_logits)
    if probs.ndim > 2:
        probs = probs.squeeze()
    if probs.ndim != 2:
        return box, None
    score = float(probs[mask].mean().item()) if mask.any() else None
    return box, score


def _generate_forward_tracklet(
    predictor,
    seed: Dict[str, Any],
    start_local: int,
    end_local: int,
    segment_frames_dir: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    safe_id = base_boxes._sanitize_filename(seed["temp_id"])
    window_dir = f"/tmp/sam2_fwd_{safe_id}_{start_local}_{end_local}_{uuid.uuid4().hex}"
    win_len = base_boxes._make_window_dir(segment_frames_dir, window_dir, start_local, end_local)

    seed_local = seed["frame_idx"]
    seed_in_win = seed_local - start_local

    try:
        state = predictor.init_state(video_path=window_dir, offload_video_to_cpu=False)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=seed_in_win,
            obj_id=1,
            box=seed["box_xyxy"],
        )

        fwd_boxes: Dict[int, np.ndarray] = {}
        fwd_scores: Dict[int, float] = {}
        for frame_local, obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state=state,
            start_frame_idx=seed_in_win,
            max_frame_num_to_track=win_len - seed_in_win,
        ):
            if 1 not in obj_ids:
                continue
            pos = obj_ids.index(1)
            box, score = _mask_logits_to_box_and_score(mask_logits[pos])
            if box is None:
                continue
            seg_frame = start_local + int(frame_local)
            fwd_boxes[seg_frame] = box
            if score is not None:
                fwd_scores[seg_frame] = score

        return fwd_boxes, fwd_scores
    finally:
        if os.path.exists(window_dir):
            shutil.rmtree(window_dir)


def _generate_backward_tracklet(
    predictor,
    seed: Dict[str, Any],
    start_local: int,
    end_local: int,
    segment_frames_dir: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    safe_id = base_boxes._sanitize_filename(seed["temp_id"])
    window_dir = f"/tmp/sam2_bwd_{safe_id}_{start_local}_{end_local}_{uuid.uuid4().hex}"
    win_len = base_boxes._make_window_dir(segment_frames_dir, window_dir, start_local, end_local)

    window_dir_rev = window_dir + "_rev"
    base_boxes._make_reversed_dir(window_dir, window_dir_rev, win_len)

    seed_local = seed["frame_idx"]
    seed_in_win = seed_local - start_local
    seed_in_win_rev = (win_len - 1 - seed_in_win)

    try:
        state_rev = predictor.init_state(video_path=window_dir_rev, offload_video_to_cpu=False)
        predictor.add_new_points_or_box(
            inference_state=state_rev,
            frame_idx=seed_in_win_rev,
            obj_id=1,
            box=seed["box_xyxy"],
        )

        bwd_boxes: Dict[int, np.ndarray] = {}
        bwd_scores: Dict[int, float] = {}
        frames_to_track = win_len - seed_in_win_rev
        for frame_local, obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state=state_rev,
            start_frame_idx=seed_in_win_rev,
            max_frame_num_to_track=frames_to_track,
        ):
            if 1 not in obj_ids:
                continue
            pos = obj_ids.index(1)
            box, score = _mask_logits_to_box_and_score(mask_logits[pos])
            if box is None:
                continue
            rev_win_frame = int(frame_local)
            win_frame = (win_len - 1 - rev_win_frame)
            seg_frame = start_local + win_frame
            bwd_boxes[seg_frame] = box
            if score is not None:
                bwd_scores[seg_frame] = score

        return bwd_boxes, bwd_scores
    finally:
        if os.path.exists(window_dir):
            shutil.rmtree(window_dir)
        if os.path.exists(window_dir_rev):
            shutil.rmtree(window_dir_rev)


def _aggregate_scores(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    return float(np.median(np.asarray(scores, dtype=np.float32)))


def _iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, (xa2 - xa1)) * max(0.0, (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1)) * max(0.0, (yb2 - yb1))
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def _resolve_frame_boxes(
    candidates: List[Tuple[np.ndarray, float]],
    iou_threshold: float = 0.3,
    mode: str = "iou-weighted",
) -> Optional[np.ndarray]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]

    if mode == "winner":
        best = max(candidates, key=lambda item: item[1])
        return best[0]

    sorted_candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
    base_box, base_score = sorted_candidates[0]
    base_score = float(base_score or 0.0)
    for box, score in sorted_candidates[1:]:
        score_val = float(score or 0.0)
        if mode == "weighted":
            weight_a = max(base_score, 1e-6)
            weight_b = max(score_val, 1e-6)
            base_box = (base_box * weight_a + box * weight_b) / (weight_a + weight_b)
            base_score = max(base_score, score_val)
            continue
        if _iou_xyxy(base_box, box) < iou_threshold:
            continue
        weight_a = max(base_score, 1e-6)
        weight_b = max(score_val, 1e-6)
        base_box = (base_box * weight_a + box * weight_b) / (weight_a + weight_b)
        base_score = max(base_score, score_val)
    return base_box


def _run_sam2_tracking(
    args: argparse.Namespace,
    task: Dict[str, Any],
    annotation: Any,
    video_path: str,
    frames_count: int,
    width: int,
    height: int,
    fps: float,
    prompt_label: Optional[str],
    track_id_filter: Optional[Set[str]],
) -> Optional[Dict[str, Any]]:
    global_start = max(0, args.global_start)
    global_end = args.global_end if args.global_end is not None else frames_count - 1
    global_end = min(frames_count - 1, global_end)

    if global_start > global_end:
        raise InitialSeedingError(f"Invalid frame range: start={global_start} > end={global_end}")

    segment_len = global_end - global_start + 1
    logger.info("Processing segment frames [%d, %d] (length=%d)", global_start, global_end, segment_len)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise InitialSeedingError(f"Could not open video: {video_path}")

    segment_frames_dir = f"/tmp/sam2_segment_frames_{uuid.uuid4().hex}"

    try:
        written = base_boxes._export_segment_to_frames_dir(cap, segment_frames_dir, global_start, global_end)
        cap.release()
        if written != segment_len:
            logger.warning("Expected %d frames but got %d, adjusting end frame", segment_len, written)
            global_end = global_start + written - 1
            segment_len = written

        manual_boxes: List[Dict[str, Any]] = []
        results = annotation.get("result", []) if isinstance(annotation, dict) else getattr(annotation, "result", [])

        available_ids: Set[str] = set()
        for region in results:
            if not isinstance(region, dict) or region.get("type") != "videorectangle":
                continue
            region_id = str(region.get("id", "unknown_region"))
            available_ids.add(region_id)

        if track_id_filter is not None:
            missing = sorted(track_id_filter - available_ids)
            if missing:
                raise InitialSeedingError(f"Track id(s) not found in annotation: {', '.join(missing)}")

        for region in results:
            if not isinstance(region, dict) or region.get("type") != "videorectangle":
                continue

            region_id = str(region.get("id", "unknown_region"))
            if track_id_filter is not None and region_id not in track_id_filter:
                continue

            value = region.get("value", {}) or {}
            sequence = value.get("sequence", []) or []

            for k, keyframe in enumerate(sequence):
                if not isinstance(keyframe, dict) or not keyframe.get("enabled", True):
                    continue

                global_frame = int(keyframe.get("frame", 1)) - 1
                if global_frame < global_start or global_frame > global_end:
                    continue

                local_frame = global_frame - global_start
                box_xyxy = base_boxes._percent_xywh_to_xyxy_px(
                    keyframe["x"], keyframe["y"], keyframe["width"], keyframe["height"], width, height
                )

                region_labels = value.get("labels", [])
                if not region_labels and prompt_label:
                    region_labels = [prompt_label]
                if not region_labels:
                    region_labels = ["object"]

                manual_boxes.append(
                    {
                        "global_frame": global_frame,
                        "frame_idx": local_frame,
                        "box_xyxy": box_xyxy,
                        "temp_id": f"{region_id}_kf{k}",
                        "region_id": region_id,
                        "labels": region_labels,
                    }
                )

        manual_boxes.sort(key=lambda b: b["frame_idx"])
        if len(manual_boxes) == 0:
            logger.warning("No keyframes found in specified segment for requested track ids.")
            return None

        logger.info("Found %d keyframe annotations in segment", len(manual_boxes))

        predictor = base_boxes._build_sam2_predictor()

        tracklets: Dict[str, Dict[str, Dict[int, Any]]] = {}
        max_frames_to_track = args.max_frames_to_track

        with tqdm(total=len(manual_boxes), desc="Generating tracklets", unit="seed") as pbar:
            for k, seed in enumerate(manual_boxes):
                prev_kf = manual_boxes[k - 1]["frame_idx"] if k > 0 else None
                next_kf = manual_boxes[k + 1]["frame_idx"] if k + 1 < len(manual_boxes) else None

                fwd_end = base_boxes._forward_end(seed["frame_idx"], next_kf, max_frames_to_track, segment_len)
                bwd_start = base_boxes._backward_start(seed["frame_idx"], prev_kf, max_frames_to_track)

                if seed["frame_idx"] == fwd_end:
                    fwd_boxes, fwd_scores = {}, {}
                else:
                    fwd_boxes, fwd_scores = _generate_forward_tracklet(
                        predictor, seed, bwd_start, fwd_end, segment_frames_dir
                    )

                if seed["frame_idx"] == bwd_start:
                    bwd_boxes, bwd_scores = {}, {}
                else:
                    bwd_boxes, bwd_scores = _generate_backward_tracklet(
                        predictor, seed, bwd_start, fwd_end, segment_frames_dir
                    )

                tracklets[seed["temp_id"]] = {
                    "fwd": fwd_boxes,
                    "bwd": bwd_boxes,
                    "fwd_scores": fwd_scores,
                    "bwd_scores": bwd_scores,
                }
                pbar.update(1)

        logger.info("Generated %d tracklets", len(tracklets))

        region_frames: Dict[str, Dict[int, List[Tuple[np.ndarray, float]]]] = {}
        region_scores: Dict[str, List[float]] = {}
        region_labels: Dict[str, List[str]] = {}

        for seed in manual_boxes:
            region_id = seed["region_id"]
            region_labels.setdefault(region_id, seed["labels"])

            t_data = tracklets.get(seed["temp_id"], {})
            for f_idx, box in t_data.get("fwd", {}).items():
                score = t_data.get("fwd_scores", {}).get(f_idx, 0.0)
                region_frames.setdefault(region_id, {}).setdefault(f_idx, []).append((box, score))
            for f_idx, box in t_data.get("bwd", {}).items():
                score = t_data.get("bwd_scores", {}).get(f_idx, 0.0)
                region_frames.setdefault(region_id, {}).setdefault(f_idx, []).append((box, score))

            for score in t_data.get("fwd_scores", {}).values():
                region_scores.setdefault(region_id, []).append(score)
            for score in t_data.get("bwd_scores", {}).values():
                region_scores.setdefault(region_id, []).append(score)

        results_out: List[Dict[str, Any]] = []
        for region_id, frames_dict in region_frames.items():
            sequence: List[Dict[str, Any]] = []
            for f_idx in sorted(frames_dict.keys()):
                candidates = frames_dict[f_idx]
                if not candidates:
                    continue
                resolved = _resolve_frame_boxes(
                    candidates,
                    iou_threshold=args.overlap_iou_threshold,
                    mode=args.overlap_mode,
                )
                if resolved is None:
                    continue

                global_frame = global_start + f_idx
                time_offset = global_frame / fps
                x_percent = float((resolved[0] / width) * 100.0)
                y_percent = float((resolved[1] / height) * 100.0)
                width_percent = float(((resolved[2] - resolved[0]) / width) * 100.0)
                height_percent = float(((resolved[3] - resolved[1]) / height) * 100.0)

                sequence.append(
                    {
                        "frame": int(global_frame + 1),
                        "x": x_percent,
                        "y": y_percent,
                        "width": width_percent,
                        "height": height_percent,
                        "enabled": True,
                        "rotation": 0,
                        "time": float(time_offset),
                    }
                )

            if not sequence:
                continue

            score = _aggregate_scores(region_scores.get(region_id, []))
            result = {
                "id": region_id,
                "type": "videorectangle",
                "from_name": "box",
                "to_name": "video",
                "score": score,
                "value": {
                    "sequence": sequence,
                    "framesCount": frames_count,
                    "duration": float(frames_count / fps),
                    "labels": region_labels.get(region_id, ["object"]),
                },
                "meta": {"text": "id:"},
            }
            base._ensure_meta_text_placeholder(result)
            results_out.append(result)

        prediction = {
            "result": results_out,
            "score": 1.0,
            "model_version": "sam2-video-boxes-manual-merge",
        }
        logger.info("Generated tracking results with %d tracks", len(results_out))
        return prediction

    finally:
        if os.path.exists(segment_frames_dir):
            shutil.rmtree(segment_frames_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label Studio I/O wrapper for SAM2 video tracking with manual merge",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--ls-url", required=True)
    parser.add_argument("--ls-api-key", required=True)
    parser.add_argument("--project", type=int, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--annotation", type=int, required=True)
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional single-token label override. If set, all videorectangle labels are set to this value.",
    )
    parser.add_argument("--track-id", default=None, help="Comma-separated list of track region ids to use as anchors")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dry-run", action="store_true", help="Print prediction JSON instead of upload")
    parser.add_argument("--global-start", type=int, default=0, help="Starting frame index (0-based inclusive)")
    parser.add_argument(
        "--global-end",
        type=int,
        default=None,
        help="Ending frame index (0-based inclusive, defaults to last frame)",
    )
    parser.add_argument(
        "--max-frames-to-track",
        type=int,
        default=300,
        help="Maximum frames to track in each direction from a keyframe",
    )
    parser.add_argument(
        "--overlap-mode",
        default="iou-weighted",
        choices=["iou-weighted", "weighted", "winner"],
        help="How to resolve same-region overlaps (default: iou-weighted)",
    )
    parser.add_argument(
        "--overlap-iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for iou-weighted overlap resolution",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    exit_code = 0
    try:
        base_boxes._disable_sam2_progress_bars()

        ls = base._build_ls_client(args.ls_url, args.ls_api_key)
        project_labels = base_boxes._fetch_project_labels(ls, args.project)
        if project_labels:
            logger.info("Project label config contains %d label(s)", len(project_labels))

        prompt_label = base_boxes._prompt_to_single_label(args.prompt)
        if prompt_label is not None and project_labels:
            canonical_label = base_boxes._canonicalize_label_to_project_config(
                label=prompt_label,
                project_labels=project_labels,
            )
            if canonical_label != prompt_label:
                logger.info(
                    "Using canonical label '%s' from project config for prompt '%s'",
                    canonical_label,
                    prompt_label,
                )
            prompt_label = canonical_label

        task = base._fetch_task(ls, args.project, args.task)
        annotation = base._fetch_annotation(ls, args.annotation)

        video_path, _ = base._get_video_path(task)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise InitialSeedingError(f"Could not open video file: {video_path}")

        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        if frames_count <= 0:
            raise InitialSeedingError("Video has no frames")

        logger.info(
            "Loaded task=%d annotation=%d | video=%s | frames=%d | size=%dx%d | fps=%.3f",
            int(args.task),
            int(args.annotation),
            str(video_path),
            int(frames_count),
            int(width),
            int(height),
            float(fps),
        )

        track_id_filter = _parse_track_ids(args.track_id)
        prediction = _run_sam2_tracking(
            args,
            task,
            annotation,
            video_path,
            frames_count,
            width,
            height,
            fps,
            prompt_label,
            track_id_filter,
        )

        if prediction is None:
            logger.warning("No matching keyframes found; exiting without changes.")
            sys.exit(0)

        if prompt_label is not None:
            results = prediction.get("result")
            if isinstance(results, list):
                for region in results:
                    if not isinstance(region, dict):
                        continue
                    if region.get("type") != "videorectangle":
                        continue
                    value = region.get("value")
                    if not isinstance(value, dict):
                        continue
                    value["labels"] = [prompt_label]

        base_boxes._validate_prediction_region_labels(
            prediction=prediction,
            expected_single_label=prompt_label,
            allowed_labels=set(project_labels) if project_labels else None,
        )

        if isinstance(annotation, dict):
            current_result = annotation.get("result", [])
        else:
            current_result = getattr(annotation, "result", [])
        if current_result is None:
            current_result = []

        seed_region_ids: Set[str]
        if track_id_filter is not None:
            seed_region_ids = set(track_id_filter)
        else:
            seed_region_ids = {
                str(region.get("id"))
                for region in current_result
                if isinstance(region, dict) and region.get("type") == "videorectangle"
            }

        filtered_result = [
            region
            for region in current_result
            if not (isinstance(region, dict) and str(region.get("id")) in seed_region_ids)
        ]

        new_tracks = prediction.get("result", [])
        merged_result = filtered_result + new_tracks

        logger.info(
            "Merging %d new tracks with %d existing regions (removed %d seed regions)",
            len(new_tracks),
            len(filtered_result),
            len(seed_region_ids),
        )

        if args.dry_run:
            print(json.dumps({"result": merged_result}, indent=2))
        else:
            payload = {
                "result": merged_result,
                "score": prediction.get("score", 1.0),
                "model_version": prediction.get("model_version", "sam2-video-boxes-manual-merge"),
            }
            base._upload_prediction(ls, task.get("id"), payload)

    except InitialSeedingError as e:
        logger.error("Error: %s", e)
        exit_code = 1
    except Exception as e:  # pragma: no cover
        logger.error("Unexpected error: %s", e, exc_info=True)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
