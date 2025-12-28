"""
CLI: Track people across a long video using SAM2 video predictor with chunked overlap,
bidirectional propagation, duplicate suppression, and Hungarian stitching across segments.

Requirements are specified in the user prompt. This script intentionally lives standalone and
reuses seeding_common helpers for LS I/O where possible without modifying existing files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests
import torch
from tqdm import tqdm

import seeding_common as base
from seeding_common import InitialSeedingError

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError as e:
    raise InitialSeedingError(f"Failed to import SAM2: {e}")

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _disable_sam2_progress_bars() -> None:
    """Disable SAM2 tqdm progress bars to avoid log clutter."""
    try:
        import sam2.sam2_video_predictor as sam2_video_predictor
        import sam2.utils.misc as sam2_misc
    except Exception as exc:
        logger.debug("Could not patch SAM2 tqdm progress bars: %s", exc)
        return

    def _quiet_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return tqdm(*args, **kwargs)

    setattr(sam2_video_predictor, "tqdm", _quiet_tqdm)
    setattr(sam2_misc, "tqdm", _quiet_tqdm)


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
        raise InitialSeedingError("Prediction is missing a valid 'result' list")

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

    logger.info("Prediction label summary (first label per region): %s", dict(label_counts))

    if mismatched:
        logger.warning(
            "Some regions did not use the expected label '%s': %s",
            expected_single_label,
            ", ".join(mismatched[:10]),
        )

    if missing:
        raise InitialSeedingError(
            "Prediction contains regions with missing/blank labels: " + ", ".join(missing[:10])
        )

    if unknown:
        raise InitialSeedingError(
            "Prediction contains regions with labels not present in the project label config: "
            + ", ".join(unknown[:10])
        )


def _build_sam2_predictor():
    """Build SAM2 video predictor from environment variables."""
    device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model_config = os.getenv("MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
    model_checkpoint = os.getenv("MODEL_CHECKPOINT", "sam2.1_hiera_large.pt")
    
    # Check multiple possible locations for checkpoint
    cand_app = os.path.join(os.getcwd(), "checkpoints", model_checkpoint)
    cand_sam2 = os.path.join("/sam2", "checkpoints", model_checkpoint)
    
    if os.path.exists(cand_app):
        checkpoint_path = cand_app
    elif os.path.exists(cand_sam2):
        checkpoint_path = cand_sam2
    else:
        # Try the path as-is
        checkpoint_path = model_checkpoint
    
    if not os.path.exists(checkpoint_path):
        raise InitialSeedingError(f"SAM2 checkpoint not found: {checkpoint_path}")
    
    return build_sam2_video_predictor(model_config, checkpoint_path, device=device)


def _percent_xywh_to_xyxy_px(x, y, w, h, W, H):
    """Convert Label Studio percent XYWH to pixel XYXY."""
    x1 = (x / 100.0) * W
    y1 = (y / 100.0) * H
    x2 = x1 + (w / 100.0) * W
    y2 = y1 + (h / 100.0) * H
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _export_segment_to_frames_dir(cap, out_dir, global_start, global_end):
    """Export video segment to frames directory for SAM2."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Seek to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(global_start))
    
    local_idx = 0
    for global_idx in range(global_start, global_end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        out_path = os.path.join(out_dir, f"{local_idx:05d}.jpg")
        cv2.imwrite(out_path, frame)
        local_idx += 1
    
    return local_idx


def _mask_logits_to_box_xyxy(mask_logits, threshold=0.0):
    """Convert SAM2 mask logits to bounding box."""
    mask = (mask_logits > threshold)
    
    # Ensure 2D mask (H, W)
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    # Verify shape is valid for torch.where unpacking
    if mask.ndim != 2:
        # If squeezing didn't result in 2D (e.g. scalar or 3D with C>1), handle gracefully
        if mask.ndim == 3:
             # Take first channel if multiple exist (unlikely for single obj tracking)
             mask = mask[0]
        elif mask.ndim < 2:
             return None
             
    if mask.ndim != 2:
        return None

    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    return np.array([xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item()], dtype=np.float32)


def _make_window_dir(seg_frames_dir, window_dir, start_local, end_local):
    """Create window directory with symlinks to segment frames."""
    os.makedirs(window_dir, exist_ok=True)
    local = 0
    for src_local in range(start_local, end_local + 1):
        src = os.path.join(seg_frames_dir, f"{src_local:05d}.jpg")
        dst = os.path.join(window_dir, f"{local:05d}.jpg")
        if not os.path.exists(dst):
            os.symlink(src, dst)
        local += 1
    return (end_local - start_local + 1)


def _make_reversed_dir(window_dir, window_dir_rev, window_len):
    """Create reversed window directory for backward tracking."""
    os.makedirs(window_dir_rev, exist_ok=True)
    for i in range(window_len):
        src = os.path.join(window_dir, f"{(window_len - 1 - i):05d}.jpg")
        dst = os.path.join(window_dir_rev, f"{i:05d}.jpg")
        if not os.path.exists(dst):
            os.symlink(src, dst)


def _forward_end(t, next_kf, max_frames_to_track, segment_len):
    """Calculate forward tracking boundary."""
    if next_kf is None:
        return min(segment_len - 1, t + max_frames_to_track)
    return min(next_kf, t + max_frames_to_track)


def _backward_start(t, prev_kf, max_frames_to_track):
    """Calculate backward tracking boundary.""" 
    if prev_kf is None:
        return max(0, t - max_frames_to_track)
    return max(prev_kf, t - max_frames_to_track)


def _sanitize_filename(name):
    """Sanitize string for use in filenames."""
    return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)


def _generate_forward_tracklet(predictor, seed, start_local, end_local, segment_frames_dir):
    """Generate forward tracklet from seed."""
    safe_id = _sanitize_filename(seed['temp_id'])
    # Use UUID to prevent collisions in parallel runs
    window_dir = f"/tmp/sam2_fwd_{safe_id}_{start_local}_{end_local}_{uuid.uuid4().hex}"
    win_len = _make_window_dir(segment_frames_dir, window_dir, start_local, end_local)
    
    seed_local = seed["frame_idx"]
    seed_in_win = seed_local - start_local
    
    try:
        state = predictor.init_state(video_path=window_dir, offload_video_to_cpu=False)
        predictor.add_new_points_or_box(
            inference_state=state, 
            frame_idx=seed_in_win, 
            obj_id=1, 
            box=seed["box_xyxy"]
        )
        
        fwd = {}
        for frame_local, obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state=state,
            start_frame_idx=seed_in_win,
            max_frame_num_to_track=win_len - seed_in_win,
        ):
            if 1 not in obj_ids:
                continue
            pos = obj_ids.index(1)
            box = _mask_logits_to_box_xyxy(mask_logits[pos])
            if box is None:
                continue
            seg_frame = start_local + int(frame_local)
            fwd[seg_frame] = box
            
        return fwd
    finally:
        if os.path.exists(window_dir):
            shutil.rmtree(window_dir)


def _generate_backward_tracklet(predictor, seed, start_local, end_local, segment_frames_dir):
    """Generate backward tracklet from seed using reversed directory."""
    safe_id = _sanitize_filename(seed['temp_id'])
    # Use UUID to prevent collisions in parallel runs
    window_dir = f"/tmp/sam2_bwd_{safe_id}_{start_local}_{end_local}_{uuid.uuid4().hex}"
    win_len = _make_window_dir(segment_frames_dir, window_dir, start_local, end_local)
    
    window_dir_rev = window_dir + "_rev"
    _make_reversed_dir(window_dir, window_dir_rev, win_len)
    
    seed_local = seed["frame_idx"]
    seed_in_win = seed_local - start_local
    seed_in_win_rev = (win_len - 1 - seed_in_win)
    
    try:
        state_rev = predictor.init_state(video_path=window_dir_rev, offload_video_to_cpu=False)
        predictor.add_new_points_or_box(
            inference_state=state_rev,
            frame_idx=seed_in_win_rev,
            obj_id=1,
            box=seed["box_xyxy"]
        )
        
        bwd = {}
        # We propagate 'forward' in the reversed video, from the seed (reversed index)
        # to the end of the reversed video (which corresponds to the start of the original window).
        # Frames to track = Total frames - Start index
        frames_to_track = win_len - seed_in_win_rev

        for frame_local, obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state=state_rev,
            start_frame_idx=seed_in_win_rev,
            max_frame_num_to_track=frames_to_track,
        ):
            if 1 not in obj_ids:
                continue
            pos = obj_ids.index(1)
            box = _mask_logits_to_box_xyxy(mask_logits[pos])
            if box is None:
                continue
            rev_win_frame = int(frame_local)
            win_frame = (win_len - 1 - rev_win_frame)
            seg_frame = start_local + win_frame
            bwd[seg_frame] = box
            
        return bwd
    finally:
        if os.path.exists(window_dir):
            shutil.rmtree(window_dir)
        if os.path.exists(window_dir_rev):
            shutil.rmtree(window_dir_rev)


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
    
    a = torch.tensor([fwd[t] for t in common_frames], device=device)
    b = torch.tensor([bwd[t] for t in common_frames], device=device)
    
    return float(_box_iou_diag_xyxy_torch(a, b).mean().item())


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


def _patch_annotation(ls_url: str, ls_api_key: str, annotation_id: int, result: List[Dict[str, Any]]) -> None:
    """Patch existing annotation with new results."""
    headers = {
        "Authorization": f"Token {ls_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{ls_url.rstrip('/')}/api/annotations/{annotation_id}/"
    # Ensure result is a list
    payload = {"result": list(result)}
    
    try:
        response = requests.patch(url, headers=headers, json=payload, timeout=180)
    except requests.exceptions.RequestException as e:
        raise InitialSeedingError(f"Failed to patch annotation: {e}")

    if response.status_code in {200, 201, 202, 204}:
        logger.info("Annotation %d updated successfully", annotation_id)
        return
        
    if response.status_code == 504:
        logger.warning("Received 504 Gateway Timeout during annotation patch. Treating as success.")
        return
        
    raise InitialSeedingError(f"Failed to patch annotation {annotation_id} (status={response.status_code}): {response.text[:200]}")


def _run_sam2_tracking(ls, args, task, annotation, video_path, 
                      frames_count, width, height, fps, prompt_label):
    """Main SAM2 tracking implementation."""
    
    # STEP 1: Validate and clamp global frame range
    global_start = max(0, args.global_start)
    global_end = args.global_end if args.global_end is not None else frames_count - 1
    global_end = min(frames_count - 1, global_end)
    
    if global_start > global_end:
        raise InitialSeedingError(f"Invalid frame range: start={global_start} > end={global_end}")
    
    segment_len = global_end - global_start + 1
    logger.info("Processing segment frames [%d, %d] (length=%d)", global_start, global_end, segment_len)
    
    # STEP 2: Export segment frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise InitialSeedingError(f"Could not open video: {video_path}")
    
    # Use unique directory for this segment execution
    segment_frames_dir = f"/tmp/sam2_segment_frames_{uuid.uuid4().hex}"
    
    try:
        written = _export_segment_to_frames_dir(cap, segment_frames_dir, global_start, global_end)
        cap.release()
        
        if written != segment_len:
            logger.warning("Expected %d frames but got %d, adjusting end frame", segment_len, written)
            global_end = global_start + written - 1
            segment_len = written
        
        # STEP 3: Parse Label Studio keyframes and filter to segment
    manual_boxes = []
    results = annotation.get("result", []) if isinstance(annotation, dict) else getattr(annotation, "result", [])
    
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
            
            local_frame = global_frame - global_start  # segment-local index
            
            box_xyxy = _percent_xywh_to_xyxy_px(
                keyframe["x"], keyframe["y"], keyframe["width"], keyframe["height"], 
                width, height
            )
            
            manual_boxes.append({
                "global_frame": global_frame,
                "frame_idx": local_frame,
                "box_xyxy": box_xyxy,
                "temp_id": f"{region_id}_kf{k}",
            })
    
    manual_boxes.sort(key=lambda b: b["frame_idx"])
    
    if len(manual_boxes) == 0:
        raise InitialSeedingError("No keyframes found in specified segment")
    
    logger.info("Found %d keyframe annotations in segment", len(manual_boxes))
    
    # Group by frame
    def _group_by_local_frame(sorted_boxes):
        groups = {}
        for b in sorted_boxes:
            groups.setdefault(b["frame_idx"], []).append(b)
        frame_list = sorted(groups.keys())
        return [groups[f] for f in frame_list], frame_list
    
    frame_groups, keyframe_local_frames = _group_by_local_frame(manual_boxes)
    
    # STEP 4: Build SAM2 predictor
    predictor = _build_sam2_predictor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # STEP 5: Generate tracklets
    tracklets = {}
    max_frames_to_track = args.max_frames_to_track
    
    for k, (t_kf, boxes_at_t) in enumerate(zip(keyframe_local_frames, frame_groups)):
        prev_kf = keyframe_local_frames[k - 1] if k > 0 else None
        next_kf = keyframe_local_frames[k + 1] if k + 1 < len(keyframe_local_frames) else None
        
        fwd_end = _forward_end(t_kf, next_kf, max_frames_to_track, segment_len)
        bwd_start = _backward_start(t_kf, prev_kf, max_frames_to_track)
        
        for seed in boxes_at_t:
            fwd = _generate_forward_tracklet(predictor, seed, bwd_start, fwd_end, segment_frames_dir)
            bwd = _generate_backward_tracklet(predictor, seed, bwd_start, fwd_end, segment_frames_dir)
            tracklets[seed["temp_id"]] = {"fwd": fwd, "bwd": bwd}
    
    logger.info("Generated %d tracklets", len(tracklets))
    
    # STEP 6: Match tracklets between consecutive keyframes
    dsu = _DSU()
    iou_threshold = 0.40
    cost_threshold = 1.0 - iou_threshold
    
    for g in range(len(frame_groups) - 1):
        cur = frame_groups[g]
        nxt = frame_groups[g + 1]
        
        cost = np.ones((len(cur), len(nxt)), dtype=np.float32)
        for i, bi in enumerate(cur):
            for j, bj in enumerate(nxt):
                miou = _mean_iou_over_overlapped_frames(
                    bi["temp_id"], bj["temp_id"], tracklets, device
                )
                cost[i, j] = 1.0 - miou
        
        matches = _hungarian_min_cost(cost)
        for i, j in matches:
            if cost[i, j] < cost_threshold:
                dsu.union(cur[i]["temp_id"], nxt[j]["temp_id"])
    
    # STEP 7: Consolidate object IDs
    root_to_gid = {}
    gid = 1
    for b in manual_boxes:
        r = dsu.find(b["temp_id"])
        if r not in root_to_gid:
            root_to_gid[r] = gid
            gid += 1
        b["obj_id"] = root_to_gid[r]
    
    logger.info("Consolidated into %d object tracks", len(root_to_gid))
    
    # STEP 8: Merge tracklets into final tracks
    logger.info("Merging tracklets into final object tracks...")
    
    # obj_id -> frame_idx -> list of boxes
    merged_tracks: Dict[int, Dict[int, List[np.ndarray]]] = {}
    
    for b in manual_boxes:
        obj_id = int(b["obj_id"])
        temp_id = b["temp_id"]
        
        if temp_id not in tracklets:
            continue
            
        t_data = tracklets[temp_id]
        
        if obj_id not in merged_tracks:
            merged_tracks[obj_id] = {}
            
        # Collect forward boxes
        for f_idx, box in t_data["fwd"].items():
            if f_idx not in merged_tracks[obj_id]:
                merged_tracks[obj_id][f_idx] = []
            merged_tracks[obj_id][f_idx].append(box)
            
        # Collect backward boxes
        for f_idx, box in t_data["bwd"].items():
            if f_idx not in merged_tracks[obj_id]:
                merged_tracks[obj_id][f_idx] = []
            merged_tracks[obj_id][f_idx].append(box)
            
    # Collapse (average) overlaps and format results
    obj_tracks = {}
    
    for obj_id, frames_dict in merged_tracks.items():
        obj_tracks[obj_id] = []
        
        sorted_frames = sorted(frames_dict.keys())
        for f_idx in sorted_frames:
            boxes = frames_dict[f_idx]
            if not boxes:
                continue
                
            # Average boxes if multiple (e.g. from overlapping forward/backward passes)
            if len(boxes) == 1:
                avg_box = boxes[0]
            else:
                avg_box = np.mean(np.stack(boxes), axis=0)
                
            global_frame = f_idx
            time_offset = global_frame / fps
            
            # Convert pixel XYXY back to percent XYWH
            x_percent = (avg_box[0] / width) * 100.0
            y_percent = (avg_box[1] / height) * 100.0
            width_percent = ((avg_box[2] - avg_box[0]) / width) * 100.0
            height_percent = ((avg_box[3] - avg_box[1]) / height) * 100.0
            
            obj_tracks[obj_id].append({
                "frame": global_frame + 1,  # LS uses 1-based frames
                "x": x_percent,
                "y": y_percent,
                "width": width_percent,
                "height": height_percent,
                "enabled": True,
                "rotation": 0,
                "time": time_offset,
            })
    
    # Build Label Studio prediction
    results = []
    for obj_id, sequence in obj_tracks.items():
        results.append({
            "id": f"auto-track-{obj_id}",
            "type": "videorectangle",
            "from_name": "box",
            "to_name": "video",
            "value": {
                "sequence": sequence,
                "framesCount": frames_count,
                "duration": frames_count / fps,
                "labels": [prompt_label] if prompt_label else ["object"]
            }
        })
    
    prediction = {
        "result": results,
        "score": 1.0,
        "model_version": "sam2-video-boxes"
    }
    
    logger.info("Generated prediction with %d tracks", len(results))
    
    return prediction
    
    finally:
        # Cleanup temp directories
        if os.path.exists(segment_frames_dir):
            shutil.rmtree(segment_frames_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label Studio I/O wrapper for SAM2 video tracking with segment-based processing",
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
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    exit_code = 0
    try:
        _disable_sam2_progress_bars()
        
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

        # SAM2 tracking implementation
        prediction = _run_sam2_tracking(
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
