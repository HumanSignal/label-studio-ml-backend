"""
CLI: Track objects across a video using SAM3 bidirectional propagation with manual merge.

This workflow keeps forward+backward results for each seed in the same region and
leaves cross-seed merging to manual ID assignment plus mergevideoregions.py.

Migrated from SAM2 to SAM3 via HuggingFace Transformers.
Video decoding uses PyAV (no disk-based JPEG extraction).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import av
import numpy as np
import torch
from PIL import Image
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


# ---------------------------------------------------------------------------
# PyAV video decode (same pattern as initial_seeding_video_boxes.py)
# ---------------------------------------------------------------------------

def _decode_segment_pyav(
    video_path: str,
    start_frame: int,
    end_frame: int,
    stride: int = 1,
    manual_frames: Optional[Set[int]] = None,
) -> List[Tuple[int, Image.Image]]:
    """Decode frames [start_frame, end_frame] to [(global_idx, PIL.Image)] via PyAV.

    If stride > 1, only decodes every Nth frame but always includes manual_frames.
    Returns list of (global_frame_idx, pil_image) tuples.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]

    need_pts_sync = False
    if start_frame > 0 and stream.average_rate and stream.time_base:
        avg_fps = float(stream.average_rate)
        target_ts = int(start_frame / avg_fps / stream.time_base)
        container.seek(target_ts, stream=stream)
        need_pts_sync = True

    results: List[Tuple[int, Image.Image]] = []
    frame_idx = 0
    manual_frames = manual_frames or set()

    for packet in container.demux(stream):
        for frame in packet.decode():
            # After seeking, sync frame_idx from PTS of first decoded frame
            if need_pts_sync and frame.pts is not None:
                avg_fps = float(stream.average_rate)
                tb = float(stream.time_base)
                frame_idx = int(round(frame.pts * tb * avg_fps))
                need_pts_sync = False

            if frame_idx < start_frame:
                frame_idx += 1
                continue
            if frame_idx > end_frame:
                container.close()
                return results

            # Apply stride (but always include manual frames)
            offset = frame_idx - start_frame
            if stride > 1 and (offset % stride) != 0 and frame_idx not in manual_frames:
                frame_idx += 1
                continue

            results.append((frame_idx, frame.to_image()))
            frame_idx += 1

    container.close()
    return results


# ---------------------------------------------------------------------------
# SAM3 tracking helpers
# ---------------------------------------------------------------------------

def _mask_to_xyxy(mask: torch.Tensor, object_score_logits: Optional[torch.Tensor] = None) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Extract xyxy bbox and score from a binary mask tensor.

    Args:
        mask: Binary mask tensor (1, H, W) or (H, W).
        object_score_logits: Optional logits tensor from Sam3TrackerVideoSegmentationOutput.
            When provided, score = sigmoid(logits) instead of binarized mask mean.

    Returns:
        (box_xyxy, score) or (None, None) if mask is empty.
    """
    mask_squeezed = mask.squeeze()
    if mask_squeezed.ndim != 2:
        return None, None
    if torch.is_tensor(mask_squeezed):
        mask_np = mask_squeezed.cpu().numpy().astype(np.uint8)
    else:
        mask_np = np.asarray(mask_squeezed, dtype=np.uint8)
    ys, xs = np.where(mask_np > 0)
    if xs.size == 0 or ys.size == 0:
        return None, None
    box = np.array([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                   dtype=np.float32)

    # Compute confidence: prefer object_score_logits (sigmoid) over binarized mask mean
    if object_score_logits is not None:
        score = float(torch.sigmoid(object_score_logits).item())
    else:
        score = float(np.mean(mask_np[mask_np > 0])) if mask_np.any() else None
    return box, score


def _generate_forward_tracklet_sam3(
    *,
    frames_list: List[Image.Image],
    frame_idx_map: Dict[int, int],  # session_idx -> global_idx
    seed_session_idx: int,
    seed_box_xyxy: np.ndarray,
    score_threshold: float = 0.1,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    """Generate forward tracklet from seed using Sam3TrackerVideoModel.

    Args:
        frames_list: Decoded PIL frames for the window.
        frame_idx_map: session_idx -> global_idx mapping.
        seed_session_idx: Session-local index of the seed frame.
        seed_box_xyxy: Original annotation box in pixel xyxy coords.
        score_threshold: Minimum object_score_logits sigmoid score to keep a frame.
            After 3 consecutive below-threshold frames, tracking terminates early.

    Returns: ({global_frame_idx: box_xyxy}, {global_frame_idx: score})
    """
    if not frames_list:
        return {}, {}

    # Always insert the original seed box at the seed frame
    seed_global = frame_idx_map.get(seed_session_idx)
    fwd_boxes: Dict[int, np.ndarray] = {}
    fwd_scores: Dict[int, float] = {}
    if seed_global is not None:
        fwd_boxes[seed_global] = seed_box_xyxy.copy()
        fwd_scores[seed_global] = 1.0

    # If seed is the last frame in the window, nothing to propagate forward
    if seed_session_idx >= len(frames_list) - 1:
        return fwd_boxes, fwd_scores

    sam3_model, sam3_processor = base._get_sam3_tracker_model()

    # Init session with all frames
    session = sam3_processor.init_video_session(
        video=frames_list,
        inference_device=base.DEVICE,
        dtype=base.DTYPE,
    )

    # Add seed box at session frame
    width_img = frames_list[0].size[0]
    height_img = frames_list[0].size[1]
    inputs = sam3_processor(images=frames_list[seed_session_idx], device=base.DEVICE, return_tensors="pt")
    sam3_processor.add_inputs_to_inference_session(
        session,
        frame_idx=seed_session_idx,
        obj_ids=[0],
        input_boxes=[[seed_box_xyxy.tolist()]],
        original_size=inputs.original_sizes[0],
    )

    # Run model forward pass on seed frame to register it in the session
    # (required before propagate_in_video_iterator can determine starting frame)
    with torch.inference_mode():
        sam3_model(inference_session=session, frame=inputs.pixel_values[0])

    consecutive_below = 0

    # Propagate forward
    with torch.inference_mode():
        for output in sam3_model.propagate_in_video_iterator(session, reverse=False):
            session_idx = output.frame_idx
            if session_idx <= seed_session_idx:
                continue  # Seed frame already handled; only process frames after seed

            global_idx = frame_idx_map.get(session_idx)
            if global_idx is None:
                continue

            # Extract object_score_logits for score computation
            obj_logits = getattr(output, "object_score_logits", None)

            masks = sam3_processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[height_img, width_img]],
                binarize=True,
            )[0]

            if output.object_ids is not None and len(output.object_ids) > 0:
                for i, obj_id in enumerate(output.object_ids):
                    if int(obj_id) != 0:
                        continue
                    box, score = _mask_to_xyxy(masks[i], object_score_logits=obj_logits)

                    # Early termination: skip frames below score threshold
                    if score is not None and score < score_threshold:
                        consecutive_below += 1
                        if consecutive_below >= 3:
                            break
                        continue

                    consecutive_below = 0
                    if box is not None:
                        fwd_boxes[global_idx] = box
                        if score is not None:
                            fwd_scores[global_idx] = score
            else:
                consecutive_below += 1

            if consecutive_below >= 3:
                break

    return fwd_boxes, fwd_scores


def _generate_backward_tracklet_sam3(
    *,
    frames_list: List[Image.Image],
    frame_idx_map: Dict[int, int],  # session_idx -> global_idx
    seed_session_idx: int,
    seed_box_xyxy: np.ndarray,
    score_threshold: float = 0.1,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    """Generate backward tracklet from seed using Sam3TrackerVideoModel.

    The seed frame is NOT included (forward owns it). Only frames before seed
    are tracked.

    Args:
        frames_list: Decoded PIL frames for the window.
        frame_idx_map: session_idx -> global_idx mapping.
        seed_session_idx: Session-local index of the seed frame.
        seed_box_xyxy: Original annotation box in pixel xyxy coords.
        score_threshold: Minimum object_score_logits sigmoid score to keep a frame.
            After 3 consecutive below-threshold frames, tracking terminates early.

    Returns: ({global_frame_idx: box_xyxy}, {global_frame_idx: score})
    """
    if not frames_list:
        return {}, {}

    # If seed is the first frame, nothing to propagate backward
    if seed_session_idx <= 0:
        return {}, {}

    sam3_model, sam3_processor = base._get_sam3_tracker_model()

    # Init session with all frames
    session = sam3_processor.init_video_session(
        video=frames_list,
        inference_device=base.DEVICE,
        dtype=base.DTYPE,
    )

    # Add seed box at session frame
    width_img = frames_list[0].size[0]
    height_img = frames_list[0].size[1]
    inputs = sam3_processor(images=frames_list[seed_session_idx], device=base.DEVICE, return_tensors="pt")
    sam3_processor.add_inputs_to_inference_session(
        session,
        frame_idx=seed_session_idx,
        obj_ids=[0],
        input_boxes=[[seed_box_xyxy.tolist()]],
        original_size=inputs.original_sizes[0],
    )

    # Run model forward pass on seed frame to register it in the session
    # (required before propagate_in_video_iterator can determine starting frame)
    with torch.inference_mode():
        sam3_model(inference_session=session, frame=inputs.pixel_values[0])

    bwd_boxes: Dict[int, np.ndarray] = {}
    bwd_scores: Dict[int, float] = {}
    consecutive_below = 0

    # Propagate backward (reverse=True)
    with torch.inference_mode():
        for output in sam3_model.propagate_in_video_iterator(session, reverse=True):
            session_idx = output.frame_idx
            # Exclude seed frame (>= not >): forward owns the seed frame
            if session_idx >= seed_session_idx:
                continue

            global_idx = frame_idx_map.get(session_idx)
            if global_idx is None:
                continue

            # Extract object_score_logits for score computation
            obj_logits = getattr(output, "object_score_logits", None)

            masks = sam3_processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[height_img, width_img]],
                binarize=True,
            )[0]

            if output.object_ids is not None and len(output.object_ids) > 0:
                for i, obj_id in enumerate(output.object_ids):
                    if int(obj_id) != 0:
                        continue
                    box, score = _mask_to_xyxy(masks[i], object_score_logits=obj_logits)

                    # Early termination: skip frames below score threshold
                    if score is not None and score < score_threshold:
                        consecutive_below += 1
                        if consecutive_below >= 3:
                            break
                        continue

                    consecutive_below = 0
                    if box is not None:
                        bwd_boxes[global_idx] = box
                        if score is not None:
                            bwd_scores[global_idx] = score
            else:
                consecutive_below += 1

            if consecutive_below >= 3:
                break

    return bwd_boxes, bwd_scores


# ---------------------------------------------------------------------------
# Detection oracle (Layer 2: cross-check tracker output against Sam3VideoModel)
# ---------------------------------------------------------------------------

def _oracle_validate_tracklet(
    *,
    tracker_boxes: Dict[int, np.ndarray],
    oracle_detections: Dict[int, List[np.ndarray]],
    iou_threshold: float = 0.3,
) -> Dict[int, np.ndarray]:
    """Remove frames where tracker box doesn't overlap any oracle detection.

    This is a post-hoc cross-check that catches ID switches: the tracker may
    latch onto a different object, but the text-based oracle detects what
    should actually be there.

    Args:
        tracker_boxes: {global_frame: xyxy_box} from tracker
        oracle_detections: {global_frame: [xyxy_box, ...]} from Sam3VideoModel
        iou_threshold: minimum IoU to consider a match

    Returns:
        Validated subset of tracker_boxes (frames that pass the check).
    """
    validated: Dict[int, np.ndarray] = {}
    for frame_idx, tracker_box in tracker_boxes.items():
        det_boxes = oracle_detections.get(frame_idx)
        if det_boxes is None:
            # Oracle doesn't cover this frame -- pass through
            validated[frame_idx] = tracker_box
            continue
        if len(det_boxes) == 0:
            # Oracle ran but found nothing -- still pass (object may be occluded)
            validated[frame_idx] = tracker_box
            continue
        max_iou = max(_iou_xyxy(tracker_box, det_box) for det_box in det_boxes)
        if max_iou >= iou_threshold:
            validated[frame_idx] = tracker_box
        else:
            logger.debug(
                "Oracle rejected frame %d: max IoU=%.3f < threshold=%.3f",
                frame_idx, max_iou, iou_threshold,
            )
    return validated


# ---------------------------------------------------------------------------
# Score and overlap helpers (preserved from SAM2)
# ---------------------------------------------------------------------------

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
    """Resolve overlapping boxes at the same frame.

    Modes:
      - 'winner': highest score wins
      - 'weighted': weighted average of all candidates
      - 'iou-weighted': weighted average only if IoU > threshold
    """
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


# ---------------------------------------------------------------------------
# Main tracking implementation
# ---------------------------------------------------------------------------

def _run_sam3_tracking(
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
    """Main SAM3 tracking implementation for manual merge workflow.

    Uses per-keyframe streaming decode: for each unique keyframe, only the
    [kf - max_ftk, kf + max_ftk] window is decoded, tracked, and released.
    This keeps memory constant regardless of total video length.
    """

    global_start = 0 if args.global_start is None else max(0, args.global_start)
    global_end = frames_count - 1 if args.global_end is None else min(frames_count - 1, args.global_end)

    if global_start > global_end:
        raise InitialSeedingError(f"Invalid frame range: start={global_start} > end={global_end}")

    segment_len = global_end - global_start + 1
    frame_stride = max(1, int(args.frame_stride))
    max_ftk = args.max_frames_to_track
    logger.info("Processing segment frames [%d, %d] (length=%d)", global_start, global_end, segment_len)
    logger.info("Frame stride: %d, max_frames_to_track: %d", frame_stride, max_ftk)

    # -----------------------------------------------------------------------
    # Phase 1: Parse ALL manual_boxes upfront — no frame decode needed.
    # Coordinate conversion only needs width/height, not decoded frames.
    # -----------------------------------------------------------------------
    results = annotation.get("result", []) if isinstance(annotation, dict) else getattr(annotation, "result", [])

    manual_boxes: List[Dict[str, Any]] = []
    all_manual_frames: Set[int] = set()
    available_ids: Set[str] = set()
    region_type_counts: Counter[str] = Counter()
    total_keyframes = 0
    total_disabled = 0

    for region in results:
        if not isinstance(region, dict) or region.get("type") != "videorectangle":
            if isinstance(region, dict):
                region_type_counts[str(region.get("type"))] += 1
            continue
        region_id = str(region.get("id", "unknown_region"))
        available_ids.add(region_id)
        region_type_counts["videorectangle"] += 1
        value = region.get("value", {}) or {}
        sequence = value.get("sequence", []) or []
        total_keyframes += len(sequence)
        total_disabled += sum(1 for keyframe in sequence if isinstance(keyframe, dict) and not keyframe.get("enabled", True))

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
            if not isinstance(keyframe, dict):
                continue

            frame_value = keyframe.get("frame")
            if frame_value is None:
                time_value = keyframe.get("time")
                if time_value is None:
                    continue
                global_frame = int(round(float(time_value) * fps))
            else:
                global_frame = int(frame_value) - 1
            if global_frame < global_start or global_frame > global_end:
                continue

            box_xyxy = base_boxes._percent_xywh_to_xyxy_px(
                keyframe["x"], keyframe["y"], keyframe["width"], keyframe["height"], width, height
            )

            region_labels = value.get("labels", [])
            if not region_labels and prompt_label:
                region_labels = [prompt_label]
            if not region_labels:
                region_labels = ["object"]

            all_manual_frames.add(global_frame)
            manual_boxes.append({
                "global_frame": global_frame,
                "box_xyxy": box_xyxy,
                "temp_id": f"{region_id}_kf{k}",
                "region_id": region_id,
                "labels": region_labels,
            })

    manual_boxes.sort(key=lambda b: b["global_frame"])
    if len(manual_boxes) == 0:
        logger.warning(
            "No keyframes found in specified segment for requested track ids. "
            "Regions=%d, types=%s, videorectangle=%d, keyframes=%d, disabled=%d",
            len(results) if isinstance(results, list) else 0,
            dict(region_type_counts),
            region_type_counts.get("videorectangle", 0),
            total_keyframes,
            total_disabled,
        )
        return None

    logger.info("Found %d keyframe annotations in segment", len(manual_boxes))

    # Group seeds by their keyframe (global frame index)
    seeds_by_kf: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for box in manual_boxes:
        seeds_by_kf[box["global_frame"]].append(box)
    unique_keyframes = sorted(seeds_by_kf.keys())
    logger.info("Unique keyframes: %d, will decode per-keyframe windows of ≤%d frames each",
                len(unique_keyframes), 2 * max_ftk + 1)

    # -----------------------------------------------------------------------
    # Phase 2: Per-keyframe streaming decode, refine, and track.
    # Memory stays constant: only one window is alive at a time.
    # -----------------------------------------------------------------------
    tracklets: Dict[str, Dict[str, Dict[int, Any]]] = {}

    with tqdm(
        total=len(manual_boxes),
        desc="Generating tracklets",
        unit="seed",
        disable=args.no_progress,
    ) as pbar:
        for kf_global in unique_keyframes:
            seeds = seeds_by_kf[kf_global]

            # Compute decode window for this keyframe
            win_start = max(global_start, kf_global - max_ftk)
            win_end = min(global_end, kf_global + max_ftk)
            # Manual frames within this window (for stride inclusion)
            win_manual = {f for f in all_manual_frames if win_start <= f <= win_end}

            logger.info(
                "Keyframe %d: decoding window [%d, %d] (%d frames, %d seeds)",
                kf_global, win_start, win_end, win_end - win_start + 1, len(seeds),
            )

            decoded = _decode_segment_pyav(
                video_path, win_start, win_end,
                stride=frame_stride, manual_frames=win_manual,
            )

            if not decoded:
                logger.warning("No frames decoded for keyframe %d window [%d, %d], skipping",
                              kf_global, win_start, win_end)
                pbar.update(len(seeds))
                continue

            # Build window-local mappings
            win_g2s: Dict[int, int] = {}
            win_s2g: Dict[int, int] = {}
            win_frames: List[Image.Image] = []
            for si, (gi, pil_img) in enumerate(decoded):
                win_g2s[gi] = si
                win_s2g[si] = gi
                win_frames.append(pil_img)

            kf_session = win_g2s.get(kf_global)
            if kf_session is None:
                logger.warning("Keyframe %d not found in decoded window, skipping", kf_global)
                pbar.update(len(seeds))
                del decoded, win_frames, win_g2s, win_s2g
                continue

            # Refine seed boxes using text+box prompts (if enabled)
            if args.refine_seeds:
                for seed in seeds:
                    original_box = seed["box_xyxy"]
                    text_label = seed["labels"][0] if seed["labels"] else "object"
                    refined_box, score = base.refine_box_with_text_prompt(
                        image=win_frames[kf_session],
                        box_xyxy=original_box,
                        text_label=text_label,
                        search_scale=args.refine_search_scale,
                    )
                    if score > 0:
                        seed["box_xyxy"] = refined_box
                        logger.debug(
                            "Refined box %s: [%.1f,%.1f,%.1f,%.1f] -> [%.1f,%.1f,%.1f,%.1f]",
                            seed["temp_id"],
                            original_box[0], original_box[1], original_box[2], original_box[3],
                            refined_box[0], refined_box[1], refined_box[2], refined_box[3],
                        )

            # Track each seed within this window
            win_idx_map = {i: win_s2g[i] for i in range(len(win_frames))}
            win_len = len(win_frames)

            # --- Oracle detection (Layer 2, once per keyframe window) ---
            oracle_detections: Optional[Dict[int, List[np.ndarray]]] = None
            if getattr(args, "enable_oracle", False):
                oracle_stride = getattr(args, "oracle_stride", 30)
                try:
                    from seeding_common import _get_sam3_video_model
                    video_model, video_processor = _get_sam3_video_model()

                    # Run Sam3VideoModel on window frames (subsample by oracle_stride)
                    oracle_frame_indices = list(range(0, len(win_frames), max(1, oracle_stride)))
                    # Always include seed frame in oracle
                    if kf_session not in oracle_frame_indices:
                        oracle_frame_indices.append(kf_session)
                    oracle_frame_indices.sort()
                    oracle_subset = [win_frames[i] for i in oracle_frame_indices]

                    prompt_text = seeds[0]["labels"][0] if seeds[0]["labels"] else "object"
                    oracle_session = video_processor.init_video_session(
                        video=oracle_subset,
                        inference_device=base.DEVICE,
                        processing_device="cpu",
                        video_storage_device="cpu",
                        dtype=base.DTYPE,
                    )
                    oracle_session = video_processor.add_text_prompt(
                        oracle_session, text=prompt_text
                    )

                    oracle_detections = {}
                    with torch.inference_mode():
                        for output in video_model.propagate_in_video_iterator(oracle_session):
                            processed = video_processor.postprocess_outputs(oracle_session, output)
                            # Map oracle subset index -> global frame index
                            oracle_local = output.frame_idx
                            if oracle_local < len(oracle_frame_indices):
                                session_idx = oracle_frame_indices[oracle_local]
                                global_idx = win_s2g.get(session_idx)
                                if global_idx is not None and processed.get("boxes") is not None:
                                    boxes = processed["boxes"]
                                    oracle_detections[global_idx] = [
                                        boxes[i].cpu().numpy() if hasattr(boxes[i], "cpu") else np.asarray(boxes[i], dtype=np.float32)
                                        for i in range(len(boxes))
                                    ]

                    logger.info(
                        "Oracle detection: %d frames with detections (stride=%d)",
                        len(oracle_detections), oracle_stride,
                    )
                except Exception as exc:
                    logger.warning("Oracle detection failed, proceeding without: %s", exc)
                    oracle_detections = None

            score_threshold = getattr(args, "score_threshold", 0.1)

            for seed in seeds:
                fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                    frames_list=win_frames,
                    frame_idx_map=win_idx_map,
                    seed_session_idx=kf_session,
                    seed_box_xyxy=seed["box_xyxy"],
                    score_threshold=score_threshold,
                )

                bwd_boxes, bwd_scores = _generate_backward_tracklet_sam3(
                    frames_list=win_frames,
                    frame_idx_map=win_idx_map,
                    seed_session_idx=kf_session,
                    seed_box_xyxy=seed["box_xyxy"],
                    score_threshold=score_threshold,
                )

                # Apply oracle validation if available
                if oracle_detections is not None:
                    fwd_boxes = _oracle_validate_tracklet(
                        tracker_boxes=fwd_boxes,
                        oracle_detections=oracle_detections,
                        iou_threshold=getattr(args, "overlap_iou_threshold", 0.3),
                    )
                    bwd_boxes = _oracle_validate_tracklet(
                        tracker_boxes=bwd_boxes,
                        oracle_detections=oracle_detections,
                        iou_threshold=getattr(args, "overlap_iou_threshold", 0.3),
                    )
                    # Prune scores to match validated boxes
                    fwd_scores = {k: v for k, v in fwd_scores.items() if k in fwd_boxes}
                    bwd_scores = {k: v for k, v in bwd_scores.items() if k in bwd_boxes}

                tracklets[seed["temp_id"]] = {
                    "fwd": fwd_boxes,
                    "bwd": bwd_boxes,
                    "fwd_scores": fwd_scores,
                    "bwd_scores": bwd_scores,
                }
                pbar.update(1)

            # Release window memory
            del decoded, win_frames, win_g2s, win_s2g
            oracle_detections = None

    logger.info("Generated %d tracklets", len(tracklets))

    # Aggregate per-region (no cross-seed merging — that's manual)
    region_frames: Dict[str, Dict[int, List[Tuple[np.ndarray, float]]]] = {}
    region_scores: Dict[str, List[float]] = {}
    region_labels: Dict[str, List[str]] = {}

    for seed in manual_boxes:
        region_id = seed["region_id"]
        region_labels.setdefault(region_id, seed["labels"])

        t_data = tracklets.get(seed["temp_id"], {})
        for g_idx, box in t_data.get("fwd", {}).items():
            score = t_data.get("fwd_scores", {}).get(g_idx, 0.0)
            region_frames.setdefault(region_id, {}).setdefault(g_idx, []).append((box, score))
        for g_idx, box in t_data.get("bwd", {}).items():
            score = t_data.get("bwd_scores", {}).get(g_idx, 0.0)
            region_frames.setdefault(region_id, {}).setdefault(g_idx, []).append((box, score))

        for score in t_data.get("fwd_scores", {}).values():
            region_scores.setdefault(region_id, []).append(score)
        for score in t_data.get("bwd_scores", {}).values():
            region_scores.setdefault(region_id, []).append(score)

    # Build results per region
    results_out: List[Dict[str, Any]] = []
    for region_id, frames_dict in region_frames.items():
        sequence: List[Dict[str, Any]] = []
        for g_idx in sorted(frames_dict.keys()):
            candidates = frames_dict[g_idx]
            if not candidates:
                continue
            resolved = _resolve_frame_boxes(
                candidates,
                iou_threshold=args.overlap_iou_threshold,
                mode=args.overlap_mode,
            )
            if resolved is None:
                continue

            time_offset = g_idx / fps
            x_percent = float((resolved[0] / width) * 100.0)
            y_percent = float((resolved[1] / height) * 100.0)
            width_percent = float(((resolved[2] - resolved[0]) / width) * 100.0)
            height_percent = float(((resolved[3] - resolved[1]) / height) * 100.0)

            sequence.append({
                "frame": int(g_idx + 1),  # 1-based
                "x": x_percent,
                "y": y_percent,
                "width": width_percent,
                "height": height_percent,
                "enabled": True,
                "rotation": 0,
                "time": float(time_offset),
            })

        # Mark frames before gaps as enabled=False to prevent LS from
        # interpolating across periods where the object is absent.
        # Use a 5-second threshold to avoid over-fragmentation from
        # minor tracker hiccups or stride irregularities.
        gap_threshold = max(1, int(round(fps * 5)))
        if len(sequence) >= 2:
            for i in range(len(sequence) - 1):
                gap = sequence[i + 1]["frame"] - sequence[i]["frame"]
                if gap > gap_threshold:
                    sequence[i]["enabled"] = False
        # Last frame always disabled (track ends)
        if sequence:
            sequence[-1]["enabled"] = False

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
        "model_version": "sam3-video-boxes-manual-merge",
    }
    logger.info("Generated tracking results with %d tracks", len(results_out))
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label Studio I/O wrapper for SAM3 video tracking with manual merge",
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
    parser.add_argument(
        "--dump-payload",
        default=None,
        help="Optional path to write the submission payload JSON before upload",
    )
    parser.add_argument(
        "--global-start",
        type=int,
        default=None,
        help="Starting frame index (0-based inclusive, defaults to 0 when omitted)",
    )
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
        "--frame-stride",
        type=int,
        default=1,
        help="Sample one frame every N frames for tracking (default: 1 = no downsampling)",
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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars for frame export and tracking",
    )
    parser.add_argument(
        "--no-refine-seeds", action="store_false", dest="refine_seeds", default=True,
        help="Disable seed box refinement (refinement is enabled by default)"
    )
    parser.add_argument(
        "--refine-search-scale", type=float, default=1.3,
        help="Search scale for seed box refinement (default: 1.3 = 30%% expansion)"
    )
    parser.add_argument(
        "--enable-oracle",
        action="store_true",
        default=False,
        help="Enable Sam3VideoModel detection oracle to cross-check tracker output (Layer 2)",
    )
    parser.add_argument(
        "--oracle-stride",
        type=int,
        default=30,
        help="Run oracle detection every N frames within each window (default: 30)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="object_score_logits threshold for early termination (sigmoid; default: 0.1)",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    exit_code = 0
    try:
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
        width, height, frames_count, fps = base._get_video_info_pyav(video_path)

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
        prediction = _run_sam3_tracking(
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

        has_global_start = args.global_start is not None
        has_global_end = args.global_end is not None
        range_provided = has_global_start and has_global_end

        new_tracks = prediction.get("result", [])
        existing_meta_by_id: Dict[str, Dict[str, Any]] = {}
        if track_id_filter is not None:
            for region in current_result:
                if not isinstance(region, dict) or region.get("type") != "videorectangle":
                    continue
                region_id = str(region.get("id", "unknown_region"))
                if region_id not in track_id_filter:
                    continue
                meta = region.get("meta")
                if isinstance(meta, dict):
                    existing_meta_by_id[region_id] = dict(meta)

            for region in new_tracks:
                if not isinstance(region, dict):
                    continue
                region_id = str(region.get("id", "unknown_region"))
                existing_meta = existing_meta_by_id.get(region_id)
                if not existing_meta:
                    continue
                meta = region.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                meta_text = existing_meta.get("text")
                if meta_text is not None:
                    meta["text"] = meta_text
                region["meta"] = meta

        merged_result: List[Dict[str, Any]] = []

        if track_id_filter is None:
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
            merged_result = filtered_result + new_tracks
            logger.info(
                "Merging %d new tracks with %d existing regions (removed %d seed regions)",
                len(new_tracks),
                len(filtered_result),
                len(seed_region_ids),
            )
        elif not range_provided:
            filtered_result = [
                region
                for region in current_result
                if not (
                    isinstance(region, dict)
                    and region.get("type") == "videorectangle"
                    and str(region.get("id")) in track_id_filter
                )
            ]
            merged_result = filtered_result + new_tracks
            logger.info(
                "Merging %d new tracks with %d existing regions (replaced %d target regions)",
                len(new_tracks),
                len(filtered_result),
                len(track_id_filter),
            )
        else:
            merge_start = max(0, args.global_start)
            merge_end = min(frames_count - 1, args.global_end)
            new_tracks_by_id = {
                str(region.get("id")): region
                for region in new_tracks
                if isinstance(region, dict)
            }

            for region in current_result:
                if not isinstance(region, dict) or region.get("type") != "videorectangle":
                    merged_result.append(region)
                    continue

                region_id = str(region.get("id", "unknown_region"))
                if region_id not in track_id_filter:
                    merged_result.append(region)
                    continue

                value = region.get("value", {}) or {}
                sequence = value.get("sequence", []) or []
                kept_sequence: List[Dict[str, Any]] = []
                for item in sequence:
                    frame_1b = int(item.get("frame", 1))
                    frame_0b = frame_1b - 1
                    if frame_0b < merge_start or frame_0b > merge_end:
                        kept_sequence.append(item)

                new_region = new_tracks_by_id.get(region_id)
                new_sequence: List[Dict[str, Any]] = []
                if isinstance(new_region, dict):
                    new_value = new_region.get("value", {}) or {}
                    new_sequence = list(new_value.get("sequence", []) or [])

                merged_sequence = kept_sequence + new_sequence
                if not merged_sequence:
                    continue

                merged_sequence.sort(key=lambda item: int(item.get("frame", 1)))
                updated_region = dict(region)
                updated_value = dict(value)
                updated_value["sequence"] = merged_sequence
                if not updated_value.get("labels") and isinstance(new_region, dict):
                    new_labels = new_region.get("value", {}).get("labels")
                    if new_labels:
                        updated_value["labels"] = new_labels
                updated_region["value"] = updated_value
                if isinstance(new_region, dict) and new_region.get("score") is not None:
                    updated_region["score"] = new_region.get("score")

                merged_result.append(updated_region)

            logger.info(
                "Merged %d target regions within range [%d, %d]",
                len(track_id_filter),
                merge_start,
                merge_end,
            )

        if args.dry_run:
            print(json.dumps({"result": merged_result}, indent=2))
        else:
            if track_id_filter is not None and range_provided:
                payload = {"result": merged_result}
            else:
                payload = {
                    "result": merged_result,
                    "score": prediction.get("score", 1.0),
                    "model_version": prediction.get("model_version", "sam3-video-boxes-manual-merge"),
                }

            if args.dump_payload:
                dump_path = os.path.expanduser(args.dump_payload)
                dump_dir = os.path.dirname(dump_path)
                if dump_dir:
                    os.makedirs(dump_dir, exist_ok=True)
                with open(dump_path, "w", encoding="utf-8") as dump_file:
                    json.dump(payload, dump_file, indent=2)
                logger.info("Wrote submission payload to %s", dump_path)

            if track_id_filter is not None and range_provided:
                base_boxes._patch_annotation(args.ls_url, args.ls_api_key, args.annotation, merged_result)
            else:
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
