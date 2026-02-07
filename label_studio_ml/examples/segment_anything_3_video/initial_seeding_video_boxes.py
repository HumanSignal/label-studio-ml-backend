"""
CLI: Track people across a long video using SAM3 video predictor with chunked overlap,
bidirectional propagation, duplicate suppression, and Hungarian stitching across segments.

Migrated from SAM2 to SAM3 via HuggingFace Transformers.
Video decoding uses PyAV (no disk-based JPEG extraction).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import av
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import seeding_common as base
from seeding_common import InitialSeedingError

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Label handling helpers (preserved from SAM2 version)
# ---------------------------------------------------------------------------

def _prompt_to_single_label(prompt: Optional[str]) -> Optional[str]:
    if prompt is None:
        return None
    label = prompt.strip()
    if not label:
        return None
    label = label.rstrip(".").strip()
    if not label:
        return None
    if any(ch.isspace() for ch in label):
        return None
    return label


def _extract_label_values_from_label_config(label_config_xml: str) -> List[str]:
    try:
        root = ET.fromstring(label_config_xml)
    except Exception:
        return []

    labels: List[str] = []
    seen: set[str] = set()
    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        if tag != "Label":
            continue
        value = elem.attrib.get("value")
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        labels.append(cleaned)
    return labels


def _canonicalize_label_to_project_config(*, label: str, project_labels: List[str]) -> str:
    if not project_labels:
        return label
    if label in project_labels:
        return label
    lower_map = {cand.lower(): cand for cand in project_labels if cand}
    return lower_map.get(label.lower(), label)


def _fetch_project_labels(ls, project_id: int) -> List[str]:
    try:
        project = ls.projects.get(id=project_id)
    except Exception as exc:
        logger.warning("Could not fetch project %s to read label config: %s", project_id, exc)
        return []

    label_config = getattr(project, "label_config", None)
    if not isinstance(label_config, str) or not label_config.strip():
        return []

    return _extract_label_values_from_label_config(label_config)


def _validate_prediction_region_labels(
    *, prediction: Dict[str, Any], expected_single_label: Optional[str] = None, allowed_labels: Optional[set[str]] = None
) -> None:
    results = prediction.get("result")
    if not isinstance(results, list):
        raise InitialSeedingError("Tracking result is missing a valid 'result' list")

    missing: List[str] = []
    label_counts: Counter[str] = Counter()
    mismatched: List[str] = []
    unknown: List[str] = []

    for idx, region in enumerate(results):
        if not isinstance(region, dict):
            missing.append(f"result[{idx}] (not a dict)")
            continue

        if region.get("type") != "videorectangle":
            continue

        region_id = str(region.get("id", f"result[{idx}]"))
        value = region.get("value")
        if not isinstance(value, dict):
            missing.append(f"{region_id}: missing value")
            continue

        labels = value.get("labels")
        if not isinstance(labels, list) or not labels:
            missing.append(f"{region_id}: missing/empty value.labels")
            continue

        clean_labels: List[str] = []
        for lbl in labels:
            if not isinstance(lbl, str):
                continue
            stripped = lbl.strip()
            if stripped:
                clean_labels.append(stripped)

        if not clean_labels:
            missing.append(f"{region_id}: labels are blank")
            continue

        first_label = clean_labels[0]
        label_counts[first_label] += 1
        if expected_single_label is not None and first_label != expected_single_label:
            mismatched.append(f"{region_id}: '{first_label}'")
        if allowed_labels is not None and first_label not in allowed_labels:
            unknown.append(f"{region_id}: '{first_label}'")

    logger.info("Tracking result label summary (first label per region): %s", dict(label_counts))

    if mismatched:
        logger.warning(
            "Some regions did not use the expected label '%s': %s",
            expected_single_label,
            ", ".join(mismatched[:10]),
        )

    if missing:
        raise InitialSeedingError(
            "Tracking result contains regions with missing/blank labels: " + ", ".join(missing[:10])
        )

    if unknown:
        raise InitialSeedingError(
            "Tracking result contains regions with labels not present in the project label config: "
            + ", ".join(unknown[:10])
        )


# ---------------------------------------------------------------------------
# Coordinate conversion (canonical pattern)
# ---------------------------------------------------------------------------

def _percent_xywh_to_xyxy_px(x, y, w, h, W, H):
    """Convert Label Studio percent XYWH to pixel XYXY."""
    x1 = (x / 100.0) * W
    y1 = (y / 100.0) * H
    x2 = x1 + (w / 100.0) * W
    y2 = y1 + (h / 100.0) * H
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ---------------------------------------------------------------------------
# PyAV video decode (replaces cv2.VideoCapture + disk JPEG extraction)
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


def _forward_end(t, next_kf, max_frames_to_track, segment_len):
    """Calculate forward tracking boundary."""
    # Ignore next_kf to allow tracking through interleaved objects
    return min(segment_len - 1, t + max_frames_to_track)


def _backward_start(t, prev_kf, max_frames_to_track):
    """Calculate backward tracking boundary."""
    # Ignore prev_kf to allow tracking through interleaved objects
    return max(0, t - max_frames_to_track)


def _sanitize_filename(name):
    """Sanitize string for use in filenames."""
    return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)


def _generate_forward_tracklet_sam3(
    *,
    frames_list: List[Image.Image],
    frame_idx_map: Dict[int, int],  # session_idx -> global_idx
    seed_session_idx: int,
    seed_box_xyxy: np.ndarray,
    score_threshold: float = 0.1,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    """Generate forward tracklet from seed using Sam3TrackerVideoModel.

    Uses chunked batch mode: all frames in memory, propagate forward from seed.
    Returns: ({global_frame_idx: box_xyxy}, {global_frame_idx: score})
    """
    if not frames_list:
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

    fwd_boxes: Dict[int, np.ndarray] = {}
    fwd_scores: Dict[int, float] = {}
    consecutive_below = 0

    # Propagate forward
    with torch.inference_mode():
        for output in sam3_model.propagate_in_video_iterator(session, reverse=False):
            session_idx = output.frame_idx
            if session_idx < seed_session_idx:
                continue  # Only forward

            global_idx = frame_idx_map.get(session_idx)
            if global_idx is None:
                continue

            # Seed frame: use original annotation box with score 1.0
            if session_idx == seed_session_idx:
                fwd_boxes[global_idx] = seed_box_xyxy.copy()
                fwd_scores[global_idx] = 1.0
                consecutive_below = 0
                continue

            masks = sam3_processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[height_img, width_img]],
                binarize=True,
            )[0]

            if output.object_ids is not None and len(output.object_ids) > 0:
                for i, obj_id in enumerate(output.object_ids):
                    if int(obj_id) != 0:
                        continue
                    obj_logits = output.object_score_logits if hasattr(output, 'object_score_logits') else None
                    box, score = _mask_to_xyxy(masks[i], object_score_logits=obj_logits)

                    # Early termination on low scores
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

    Uses chunked batch mode: all frames in memory, propagate backward from seed.
    Returns: ({global_frame_idx: box_xyxy}, {global_frame_idx: score})
    """
    if not frames_list:
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
            # Exclude seed frame (forward owns it) and frames after seed
            if session_idx >= seed_session_idx:
                continue

            global_idx = frame_idx_map.get(session_idx)
            if global_idx is None:
                continue

            masks = sam3_processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[height_img, width_img]],
                binarize=True,
            )[0]

            if output.object_ids is not None and len(output.object_ids) > 0:
                for i, obj_id in enumerate(output.object_ids):
                    if int(obj_id) != 0:
                        continue
                    obj_logits = output.object_score_logits if hasattr(output, 'object_score_logits') else None
                    box, score = _mask_to_xyxy(masks[i], object_score_logits=obj_logits)

                    # Early termination on low scores
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


def _oracle_validate_tracklet(
    *,
    tracker_boxes: Dict[int, np.ndarray],
    oracle_detections: Dict[int, List[np.ndarray]],
    iou_threshold: float = 0.3,
) -> Dict[int, np.ndarray]:
    """Validate tracker boxes against oracle detections, removing ID-switched frames.

    For each frame: if the tracker box has no overlap (max IoU < threshold)
    with any oracle detection, the frame is removed.

    Args:
        tracker_boxes: {global_frame_idx: box_xyxy} from tracker.
        oracle_detections: {global_frame_idx: [box_xyxy, ...]} from Sam3VideoModel.
        iou_threshold: Minimum IoU for a tracker box to be considered valid.

    Returns:
        Filtered {global_frame_idx: box_xyxy} with ID-switched frames removed.
    """
    validated: Dict[int, np.ndarray] = {}
    for frame_idx, tracker_box in tracker_boxes.items():
        detections = oracle_detections.get(frame_idx)
        if detections is None:
            # No oracle data for this frame; keep the tracker box
            validated[frame_idx] = tracker_box
            continue

        if len(detections) == 0:
            # Oracle ran but found nothing; remove the tracker box
            continue

        # Compute IoU of tracker_box against each oracle detection
        max_iou = 0.0
        tb = tracker_box.astype(np.float64)
        for det_box in detections:
            db = det_box.astype(np.float64)
            inter_x1 = max(tb[0], db[0])
            inter_y1 = max(tb[1], db[1])
            inter_x2 = min(tb[2], db[2])
            inter_y2 = min(tb[3], db[3])
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            if inter_area <= 0:
                continue
            area_tb = max(0.0, tb[2] - tb[0]) * max(0.0, tb[3] - tb[1])
            area_db = max(0.0, db[2] - db[0]) * max(0.0, db[3] - db[1])
            union = area_tb + area_db - inter_area
            if union > 0:
                iou = inter_area / union
                if iou > max_iou:
                    max_iou = iou

        if max_iou >= iou_threshold:
            validated[frame_idx] = tracker_box

    return validated


# ---------------------------------------------------------------------------
# IoU and Hungarian matching (preserved from SAM2)
# ---------------------------------------------------------------------------

def _box_iou_diag_xyxy_torch(a, b, eps=1e-6):
    """Compute IoU between paired boxes."""
    inter_x1 = torch.maximum(a[:, 0], b[:, 0])
    inter_y1 = torch.maximum(a[:, 1], b[:, 1])
    inter_x2 = torch.minimum(a[:, 2], b[:, 2])
    inter_y2 = torch.minimum(a[:, 3], b[:, 3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter = inter_w * inter_h

    area_a = torch.clamp(a[:, 2] - a[:, 0], min=0.0) * torch.clamp(a[:, 3] - a[:, 1], min=0.0)
    area_b = torch.clamp(b[:, 2] - b[:, 0], min=0.0) * torch.clamp(b[:, 3] - b[:, 1], min=0.0)

    return inter / (area_a + area_b - inter + eps)


def _mean_iou_over_overlapped_frames(temp_id_prev, temp_id_next, tracklets, device):
    """Compute mean IoU over overlapped frames between two tracklets."""
    fwd = tracklets[temp_id_prev]["fwd"]
    bwd = tracklets[temp_id_next]["bwd"]

    common_frames = sorted(set(fwd.keys()) & set(bwd.keys()))
    if len(common_frames) == 0:
        return 0.0

    # Stack numpy arrays first to avoid slow list-to-tensor conversion warning
    a_np = np.stack([fwd[t] for t in common_frames])
    b_np = np.stack([bwd[t] for t in common_frames])

    a = torch.from_numpy(a_np).to(device)
    b = torch.from_numpy(b_np).to(device)

    ious = _box_iou_diag_xyxy_torch(a, b)
    mean_iou = float(ious.mean().item())

    # Debug logging for the first few comparisons or if IoU is unexpectedly low/high
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Match check: %s vs %s | Common frames: %d | Mean IoU: %.4f",
            temp_id_prev, temp_id_next, len(common_frames), mean_iou
        )
        if len(common_frames) > 0 and mean_iou < 0.1:
            # Log first frame boxes to see divergence
            logger.debug("  Frame %d: Box A (fwd)=%s, Box B (bwd)=%s",
                         common_frames[0], a_np[0].tolist(), b_np[0].tolist())
            # Log last frame boxes to see divergence
            logger.debug("  Frame %d: Box A (fwd)=%s, Box B (bwd)=%s",
                         common_frames[-1], a_np[-1].tolist(), b_np[-1].tolist())

    return mean_iou


def _hungarian_min_cost(cost):
    """Hungarian algorithm implementation (Kuhn-Munkres)."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape

    # Algorithm requires n <= m (rows <= cols). If n > m, transpose.
    if n > m:
        matches = _hungarian_min_cost(cost.T)
        return [(c, r) for r, c in matches]

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int32)
    way = np.zeros(m + 1, dtype=np.int32)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # Fixed: Convert 1-indexed to 0-indexed assignments
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]


class _DSU:
    """Disjoint Set Union for track merging."""
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# LS API helpers (preserved)
# ---------------------------------------------------------------------------

def _patch_annotation(ls_url: str, ls_api_key: str, annotation_id: int, result: List[Dict[str, Any]]) -> None:
    """Patch existing annotation with new results."""
    import requests

    headers = {
        "Authorization": f"Token {ls_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{ls_url.rstrip('/')}/api/annotations/{annotation_id}/"
    # Ensure result is a list
    payload = {"result": list(result)}

    try:
        response = requests.patch(url, headers=headers, json=payload, timeout=180)
    except Exception as e:
        raise InitialSeedingError(f"Failed to patch annotation: {e}")

    if response.status_code in {200, 201, 202, 204}:
        logger.info("Annotation %d updated successfully", annotation_id)
        return

    if response.status_code == 504:
        logger.warning("Received 504 Gateway Timeout during annotation patch. Treating as success.")
        return

    raise InitialSeedingError(f"Failed to patch annotation {annotation_id} (status={response.status_code}): {response.text[:200]}")


# ---------------------------------------------------------------------------
# Main tracking implementation
# ---------------------------------------------------------------------------

def _run_sam3_tracking(ls, args, task, annotation, video_path,
                      frames_count, width, height, fps, prompt_label):
    """Main SAM3 tracking implementation (replaces SAM2).

    Uses per-keyframe streaming decode: for each unique keyframe, only the
    [kf - max_ftk, kf + max_ftk] window is decoded, tracked, and released.
    This keeps memory constant regardless of total video length.
    """

    # STEP 1: Validate and clamp global frame range
    global_start = max(0, args.global_start)
    global_end = args.global_end if args.global_end is not None else frames_count - 1
    global_end = min(frames_count - 1, global_end)

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

    all_manual_frames: Set[int] = set()
    manual_boxes = []

    for region in results:
        if not isinstance(region, dict) or region.get("type") != "videorectangle":
            continue

        region_id = str(region.get("id", "unknown_region"))
        value = region.get("value", {}) or {}
        sequence = value.get("sequence", []) or []

        for k, keyframe in enumerate(sequence):
            if not isinstance(keyframe, dict) or not keyframe.get("enabled", True):
                continue

            # Convert 1-based LS frame to 0-based global
            global_frame = int(keyframe.get("frame", 1)) - 1

            # Filter to segment range
            if global_frame < global_start or global_frame > global_end:
                continue

            box_xyxy = _percent_xywh_to_xyxy_px(
                keyframe["x"], keyframe["y"], keyframe["width"], keyframe["height"],
                width, height
            )

            # Capture labels from source region, fallback to prompt or default
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
                "labels": region_labels,
            })

    manual_boxes.sort(key=lambda b: b["global_frame"])

    if len(manual_boxes) == 0:
        raise InitialSeedingError("No keyframes found in specified segment")

    logger.info("Found %d keyframe annotations in segment", len(manual_boxes))

    # Group seeds by their keyframe (global frame index)
    seeds_by_kf: Dict[int, List] = defaultdict(list)
    for box in manual_boxes:
        seeds_by_kf[box["global_frame"]].append(box)
    unique_keyframes = sorted(seeds_by_kf.keys())
    logger.info("Unique keyframes: %d, will decode per-keyframe windows of ≤%d frames each",
                len(unique_keyframes), 2 * max_ftk + 1)

    device = torch.device(base.DEVICE)

    # -----------------------------------------------------------------------
    # Phase 2: Per-keyframe streaming decode, refine, and track.
    # Memory stays constant: only one window is alive at a time.
    # -----------------------------------------------------------------------
    tracklets = {}
    max_frames_to_track = max_ftk

    with tqdm(total=len(manual_boxes), desc="Generating tracklets", unit="seed") as pbar:
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

            # Run oracle detection on window frames if enabled
            oracle_detections: Optional[Dict[int, List[np.ndarray]]] = None
            if getattr(args, 'enable_oracle', False):
                try:
                    from seeding_common import _get_sam3_video_model
                    video_model, video_processor = _get_sam3_video_model()
                    oracle_session = video_processor.init_video_session(
                        video=win_frames, inference_device=base.DEVICE,
                        processing_device="cpu", video_storage_device="cpu", dtype=base.DTYPE
                    )
                    oracle_prompt = prompt_label or "person"
                    oracle_session = video_processor.add_text_prompt(oracle_session, text=oracle_prompt)
                    oracle_detections = {}
                    with torch.inference_mode():
                        for o_output in video_model.propagate_in_video_iterator(oracle_session):
                            processed = video_processor.postprocess_outputs(oracle_session, o_output)
                            o_global_idx = win_s2g.get(o_output.frame_idx)
                            if o_global_idx is not None and processed.get("boxes") is not None:
                                boxes_tensor = processed["boxes"]
                                oracle_detections[o_global_idx] = [
                                    boxes_tensor[bi].cpu().numpy() for bi in range(len(boxes_tensor))
                                ]
                except Exception as exc:
                    logger.warning("Oracle detection failed: %s", exc)
                    oracle_detections = None

            score_threshold = getattr(args, 'score_threshold', 0.1)

            for seed in seeds:
                if kf_session >= win_len - 1:
                    fwd_boxes, fwd_scores = {}, {}
                else:
                    fwd_boxes, fwd_scores = _generate_forward_tracklet_sam3(
                        frames_list=win_frames,
                        frame_idx_map=win_idx_map,
                        seed_session_idx=kf_session,
                        seed_box_xyxy=seed["box_xyxy"],
                        score_threshold=score_threshold,
                    )

                if kf_session <= 0:
                    bwd_boxes, bwd_scores = {}, {}
                else:
                    bwd_boxes, bwd_scores = _generate_backward_tracklet_sam3(
                        frames_list=win_frames,
                        frame_idx_map=win_idx_map,
                        seed_session_idx=kf_session,
                        seed_box_xyxy=seed["box_xyxy"],
                        score_threshold=score_threshold,
                    )

                # Oracle validation (Layer 2): remove ID-switched frames
                if oracle_detections is not None:
                    fwd_boxes = _oracle_validate_tracklet(
                        tracker_boxes=fwd_boxes,
                        oracle_detections=oracle_detections,
                        iou_threshold=0.3,
                    )
                    bwd_boxes = _oracle_validate_tracklet(
                        tracker_boxes=bwd_boxes,
                        oracle_detections=oracle_detections,
                        iou_threshold=0.3,
                    )

                tracklets[seed["temp_id"]] = {
                    "fwd": fwd_boxes,
                    "bwd": bwd_boxes,
                    "fwd_scores": fwd_scores,
                    "bwd_scores": bwd_scores,
                }
                pbar.update(1)

            # Release window memory
            del decoded, win_frames, win_g2s, win_s2g

    logger.info("Generated %d tracklets", len(tracklets))

    # STEP 6: Match tracklets globally using DSU + IoU overlap
    dsu = _DSU()
    iou_threshold = 0.30
    merge_count = 0

    # Greedy matching across all valid pairs
    edges = []

    # manual_boxes is sorted by global_frame
    for i in range(len(manual_boxes)):
        for j in range(i + 1, len(manual_boxes)):
            b_i = manual_boxes[i]
            b_j = manual_boxes[j]

            # Skip if time gap is too large (using global frame distance)
            if b_j["global_frame"] - b_i["global_frame"] > max_frames_to_track:
                break

            miou = _mean_iou_over_overlapped_frames(
                b_i["temp_id"], b_j["temp_id"], tracklets, device
            )

            # Only consider meaningful overlaps
            if miou > iou_threshold:
                edges.append({
                    "u": i,
                    "v": j,
                    "cost": 1.0 - miou,
                    "u_id": b_i["temp_id"],
                    "v_id": b_j["temp_id"]
                })

    # Sort by best match (lowest cost/highest IoU)
    edges.sort(key=lambda x: x["cost"])

    matched_u = set()
    matched_v = set()

    for e in edges:
        if e["u"] not in matched_u and e["v"] not in matched_v:
            dsu.union(e["u_id"], e["v_id"])
            matched_u.add(e["u"])
            matched_v.add(e["v"])
            merge_count += 1

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Merged %s -> %s (cost=%.4f)",
                    e["u_id"], e["v_id"], e["cost"]
                )

    logger.info("Merged %d pairs of tracklets based on global IoU overlap", merge_count)

    # STEP 7: Consolidate object IDs
    root_to_gid = {}
    root_to_labels = {}
    gid = 1
    for b in manual_boxes:
        r = dsu.find(b["temp_id"])
        if r not in root_to_gid:
            root_to_gid[r] = gid
            root_to_labels[r] = b["labels"]
            gid += 1
        b["obj_id"] = root_to_gid[r]

    logger.info("Consolidated into %d object tracks", len(root_to_gid))

    # STEP 8: Merge tracklets into final tracks
    logger.info("Merging tracklets into final object tracks...")

    # obj_id -> global_frame -> list of boxes
    merged_tracks: Dict[int, Dict[int, List[np.ndarray]]] = {}

    for b in manual_boxes:
        obj_id = int(b["obj_id"])
        temp_id = b["temp_id"]

        if temp_id not in tracklets:
            continue

        t_data = tracklets[temp_id]

        if obj_id not in merged_tracks:
            merged_tracks[obj_id] = {}

        # Collect forward boxes (keyed by global frame)
        for g_idx, box in t_data["fwd"].items():
            if g_idx not in merged_tracks[obj_id]:
                merged_tracks[obj_id][g_idx] = []
            merged_tracks[obj_id][g_idx].append(box)

        # Collect backward boxes
        for g_idx, box in t_data["bwd"].items():
            if g_idx not in merged_tracks[obj_id]:
                merged_tracks[obj_id][g_idx] = []
            merged_tracks[obj_id][g_idx].append(box)

    # Collapse (average) overlaps and format results
    # Reverse lookup for labels: gid -> labels
    gid_to_labels = {gid: labels for r, (gid, labels) in zip(root_to_gid.keys(), zip(root_to_gid.values(), root_to_labels.values()))}

    obj_tracks = {}
    for obj_id, frames_dict in merged_tracks.items():
        obj_tracks[obj_id] = []

        sorted_frames = sorted(frames_dict.keys())
        for g_idx in sorted_frames:
            boxes = frames_dict[g_idx]
            if not boxes:
                continue

            # Average boxes if multiple (e.g. from overlapping forward/backward passes)
            if len(boxes) == 1:
                avg_box = boxes[0]
            else:
                avg_box = np.mean(np.stack(boxes), axis=0)

            time_offset = g_idx / fps

            # Convert pixel XYXY back to percent XYWH
            x_percent = float((avg_box[0] / width) * 100.0)
            y_percent = float((avg_box[1] / height) * 100.0)
            width_percent = float(((avg_box[2] - avg_box[0]) / width) * 100.0)
            height_percent = float(((avg_box[3] - avg_box[1]) / height) * 100.0)

            obj_tracks[obj_id].append({
                "frame": int(g_idx + 1),  # LS uses 1-based frames
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
    for obj_id, sequence in obj_tracks.items():
        if len(sequence) >= 2:
            for i in range(len(sequence) - 1):
                gap = sequence[i + 1]["frame"] - sequence[i]["frame"]
                if gap > gap_threshold:
                    sequence[i]["enabled"] = False
        if sequence:
            sequence[-1]["enabled"] = False

    # Build Label Studio tracking results
    results_out = []
    for obj_id, sequence in obj_tracks.items():
        # Use propagated labels or fallback
        labels = gid_to_labels.get(obj_id, ["object"])

        results_out.append({
            "id": f"auto-track-{obj_id}",
            "type": "videorectangle",
            "from_name": "box",
            "to_name": "video",
            "value": {
                "sequence": sequence,
                "framesCount": frames_count,
                "duration": float(frames_count / fps),
                "labels": labels
            }
        })

    prediction = {
        "result": results_out,
        "score": 1.0,
        "model_version": "sam3-video-boxes"
    }

    logger.info("Generated tracking results with %d tracks", len(results_out))

    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label Studio I/O wrapper for SAM3 video tracking with segment-based processing",
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
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dry-run", action="store_true", help="Print prediction JSON instead of upload")
    parser.add_argument(
        "--global-start", type=int, default=0,
        help="Starting frame index (0-based inclusive)"
    )
    parser.add_argument(
        "--global-end", type=int, default=None,
        help="Ending frame index (0-based inclusive, defaults to last frame)"
    )
    parser.add_argument(
        "--max-frames-to-track", type=int, default=300,
        help="Maximum frames to track for each tracklet"
    )
    parser.add_argument(
        "--frame-stride", type=int, default=1,
        help="Sample one frame every N frames for tracking (default: 1 = no downsampling)"
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
        "--enable-oracle", action="store_true", default=False,
        help="Enable Sam3VideoModel oracle cross-check for ID switch detection"
    )
    parser.add_argument(
        "--oracle-stride", type=int, default=30,
        help="Frame stride for oracle detection (default: 30)"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.1,
        help="object_score_logits threshold for early termination (default: 0.1)"
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    exit_code = 0
    try:
        ls = base._build_ls_client(args.ls_url, args.ls_api_key)
        project_labels = _fetch_project_labels(ls, args.project)
        if project_labels:
            logger.info("Project label config contains %d label(s)", len(project_labels))

        prompt_label = _prompt_to_single_label(args.prompt)
        if prompt_label is not None and project_labels:
            canonical_label = _canonicalize_label_to_project_config(
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

        # SAM3 tracking implementation
        prediction = _run_sam3_tracking(
            ls, args, task, annotation, video_path,
            frames_count, width, height, fps, prompt_label
        )

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

        _validate_prediction_region_labels(
            prediction=prediction,
            expected_single_label=prompt_label,
            allowed_labels=set(project_labels) if project_labels else None,
        )

        # Retrieve existing results
        if isinstance(annotation, dict):
            current_result = annotation.get("result", [])
        else:
            current_result = getattr(annotation, "result", [])

        if current_result is None:
            current_result = []

        # Resolve frame range for trimming
        start_trim = max(0, args.global_start)
        end_trim = args.global_end if args.global_end is not None else frames_count - 1
        end_trim = min(frames_count - 1, end_trim)

        logger.info("Trimming existing tracks in range [%d, %d]", start_trim, end_trim)

        trimmed_result = []
        for region in current_result:
            # Pass through non-video/non-rectangle regions unchanged
            if not isinstance(region, dict) or region.get("type") != "videorectangle":
                trimmed_result.append(region)
                continue

            value = region.get("value", {})
            sequence = value.get("sequence", [])

            new_sequence = []
            for item in sequence:
                # LS frames are 1-based
                frame_1b = int(item.get("frame", 1))
                frame_0b = frame_1b - 1

                # Keep frame if it falls OUTSIDE the processed segment
                if frame_0b < start_trim or frame_0b > end_trim:
                    new_sequence.append(item)

            if new_sequence:
                value["sequence"] = new_sequence
                # Note: We don't update framesCount/duration here, relying on original metadata
                # or subsequent cleanup tools.
                trimmed_result.append(region)
            else:
                # If a region is completely contained in the segment, it is removed.
                # This effectively replaces manual regions with auto-tracked regions in this segment.
                pass

        new_tracks = prediction.get("result", [])
        merged_result = trimmed_result + new_tracks

        logger.info("Merging %d new tracks with %d existing regions (trimmed from %d)",
                   len(new_tracks), len(trimmed_result), len(current_result))

        if args.dry_run:
            # Output the full payload that would be sent
            print(json.dumps({"result": merged_result}, indent=2))
        else:
            _patch_annotation(args.ls_url, args.ls_api_key, args.annotation, merged_result)

    except InitialSeedingError as e:
        logger.error("Error: %s", e)
        exit_code = 1
    except Exception as e:  # pragma: no cover
        logger.error("Unexpected error: %s", e, exc_info=True)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
