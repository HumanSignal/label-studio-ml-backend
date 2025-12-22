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
import math
import os
import shutil
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

import seeding_common as base
from seeding_common import InitialSeedingError, xyxy_to_percent
from initial_seeding_video import _build_sam2_video_predictor, _disable_sam2_progress_bars

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# -------------------------- Data structures -------------------------- #
@dataclass
class PromptBox:
    frame_global: int
    frame_local: int
    xyxy: np.ndarray
    label: str
    obj_group: int
    score: float = 1.0


@dataclass
class TrackRecord:
    xyxy: np.ndarray
    conf: float
    enabled: bool
    ambiguity: float = 0.0


# -------------------------- Utilities -------------------------- #
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _xyxy_center(xyxy: np.ndarray) -> Tuple[float, float]:
    x0, y0, x1, y1 = xyxy
    return float(x0 + x1) * 0.5, float(y0 + y1) * 0.5


def _xyxy_area(xyxy: np.ndarray) -> float:
    x0, y0, x1, y1 = xyxy
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _mask_logits_to_xyxy(mask_logits: torch.Tensor) -> Optional[np.ndarray]:
    mask = (mask_logits > 0.0).detach().to("cpu")
    while mask.ndim > 2:
        mask = mask.squeeze(0)
    mask_np = mask.numpy().astype(np.uint8)
    ys, xs = np.where(mask_np > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _extract_all_frames(video_path: str, temp_total_dir: str, jpeg_quality: int) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise InitialSeedingError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if os.path.isdir(temp_total_dir):
        existing = [f for f in os.listdir(temp_total_dir) if f.lower().endswith(".jpg")]
        if len(existing) == total_frames:
            logger.info("tmp_total_frames already has %d frames; skipping extraction", total_frames)
            cap.release()
            return total_frames
        shutil.rmtree(temp_total_dir)
    _ensure_dir(temp_total_dir)

    logger.info("Extracting %d frames to %s", total_frames, temp_total_dir)
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Extract frames", unit="frame")
    while frame_idx < total_frames:
        success, frame_bgr = cap.read()
        if not success or frame_bgr is None:
            break
        out_path = os.path.join(temp_total_dir, f"{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        frame_idx += 1
        pbar.update(1)
    pbar.close()

    cap.release()
    if frame_idx != total_frames:
        raise InitialSeedingError(f"Frame extraction incomplete: wrote {frame_idx}/{total_frames}")
    return total_frames


def _prepare_segment(temp_total_dir: str, seg_dir: str, start: int, end: int) -> int:
    if end < start:
        raise InitialSeedingError(f"Invalid segment range {start}-{end}")
    _ensure_dir(seg_dir)
    for name in os.listdir(seg_dir):
        if name.lower().endswith(".jpg"):
            try:
                os.remove(os.path.join(seg_dir, name))
            except FileNotFoundError:
                pass
    num_frames = end - start + 1
    for local_idx, global_idx in enumerate(range(start, end + 1)):
        src = os.path.join(temp_total_dir, f"{global_idx:06d}.jpg")
        dst = os.path.join(seg_dir, f"{local_idx:05d}.jpg")
        if not os.path.exists(src):
            raise InitialSeedingError(f"Missing extracted frame {src}")
        _link_or_copy(src, dst)
    return num_frames


def _prepare_segment_reversed(seg_dir: str, seg_rev_dir: str, num_frames: int) -> None:
    _ensure_dir(seg_rev_dir)
    for name in os.listdir(seg_rev_dir):
        if name.lower().endswith(".jpg"):
            try:
                os.remove(os.path.join(seg_rev_dir, name))
            except FileNotFoundError:
                pass
    for local_idx in range(num_frames):
        rev_idx = num_frames - 1 - local_idx
        src = os.path.join(seg_dir, f"{local_idx:05d}.jpg")
        dst = os.path.join(seg_rev_dir, f"{rev_idx:05d}.jpg")
        if not os.path.exists(src):
            raise InitialSeedingError(f"Missing segment frame {src} for reverse build")
        _link_or_copy(src, dst)


# -------------------------- Hungarian (numpy) -------------------------- #
def _hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return row_indices, col_indices for minimal cost assignment. Pads rectangular matrix."""
    cost = np.array(cost, dtype=np.float64, copy=True)
    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)
    big = cost.max() + abs(cost.max()) + 1.0 if cost.size > 0 else 1e6
    padded = np.full((n, n), big, dtype=np.float64)
    padded[:n_rows, :n_cols] = cost

    # Hungarian algorithm (Kuhn-Munkres)
    u = np.zeros(n)
    v = np.zeros(n)
    p = np.full(n, -1, dtype=int)
    way = np.full(n, -1, dtype=int)
    for i in range(n):
        p[0] = i
        j0 = 0
        minv = np.full(n, np.inf)
        used = np.zeros(n, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n):
                if used[j]:
                    continue
                cur = padded[i0, j] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == -1:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    # Build assignment
    row_to_col = np.full(n_rows, -1, dtype=int)
    col_to_row = np.full(n_cols, -1, dtype=int)
    for j in range(1, n):
        i = p[j]
        if i < n_rows and j < n_cols:
            row_to_col[i] = j
            col_to_row[j] = i
    return row_to_col, col_to_row


# -------------------------- Parsing manual boxes -------------------------- #
def _parse_manual_boxes(annotation: Any, width: int, height: int) -> List[PromptBox]:
    results = getattr(annotation, "result", None) or []
    prompts: List[PromptBox] = []
    for region in results:
        if not isinstance(region, dict):
            continue
        if region.get("type") != "videorectangle":
            continue
        value = region.get("value") or {}
        label_list = value.get("labels") or []
        label = label_list[0].strip() if label_list and isinstance(label_list[0], str) else "object"
        seq = value.get("sequence") or []
        for item in seq:
            if not isinstance(item, dict):
                continue
            frame = int(item.get("frame", 1))
            frame_global = max(0, frame - 1)  # convert to 0-based
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            w = float(item.get("width", 1.0))
            h = float(item.get("height", 1.0))
            x0 = (x / 100.0) * width
            y0 = (y / 100.0) * height
            x1 = x0 + (w / 100.0) * width
            y1 = y0 + (h / 100.0) * height
            xyxy = np.array([x0, y0, x1, y1], dtype=np.float32)
            prompts.append(
                PromptBox(
                    frame_global=frame_global,
                    frame_local=0,  # fill later
                    xyxy=xyxy,
                    label=label,
                    obj_group=-1,
                )
            )
    prompts.sort(key=lambda p: p.frame_global)
    return prompts


def _cluster_prompts_same_frame(boxes: List[PromptBox], iou_thresh: float = 0.7) -> List[PromptBox]:
    if not boxes:
        return []
    used = [False] * len(boxes)
    merged: List[PromptBox] = []
    group_id = 1
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        group_members = [i]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if _compute_iou(b.xyxy, boxes[j].xyxy) > iou_thresh:
                used[j] = True
                group_members.append(j)
        # choose representative (first)
        rep = boxes[group_members[0]]
        rep.obj_group = group_id
        merged.append(rep)
        group_id += 1
    return merged


# -------------------------- Propagation helpers -------------------------- #
def _add_prompts(predictor, inference_state, prompts: List[PromptBox]) -> None:
    for p in prompts:
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(p.frame_local),
            obj_id=int(p.obj_group),
            box=p.xyxy.astype(np.float32),
        )


def _run_propagation(
    predictor,
    inference_state,
    start_frame: int,
    max_frames: int,
    use_cuda: bool,
) -> Iterable[Tuple[int, List[int], torch.Tensor]]:
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        yield from predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=start_frame,
            max_frame_num_to_track=max_frames,
        )


def _track_segment_bidirectional(
    *,
    predictor,
    seg_dir: str,
    seg_rev_dir: str,
    seg_start: int,
    seg_end: int,
    prompts_local: List[PromptBox],
) -> Dict[int, Dict[int, TrackRecord]]:
    num_frames = seg_end - seg_start + 1
    if num_frames <= 0:
        return {}
    if not prompts_local:
        return {}

    use_cuda = torch.cuda.is_available() and os.getenv("DEVICE", "cuda").startswith("cuda")

    # Forward
    inference_state_fwd = predictor.init_state(video_path=seg_dir, offload_video_to_cpu=True)
    predictor.reset_state(inference_state_fwd)
    _add_prompts(predictor, inference_state_fwd, prompts_local)
    min_prompt = min(p.frame_local for p in prompts_local)
    forward: Dict[int, Dict[int, TrackRecord]] = defaultdict(dict)
    for frame_local, obj_ids, mask_logits in _run_propagation(
        predictor,
        inference_state_fwd,
        start_frame=min_prompt,
        max_frames=num_frames,
        use_cuda=use_cuda,
    ):
        for i, obj_id in enumerate(obj_ids):
            bbox = _mask_logits_to_xyxy(mask_logits[i])
            conf = float(torch.sigmoid(mask_logits[i]).max().item()) if bbox is not None else 0.0
            rec = TrackRecord(
                xyxy=bbox if bbox is not None else None,
                conf=conf,
                enabled=bbox is not None,
            )
            forward[int(obj_id)][int(frame_local)] = rec

    # Backward via reversed directory
    _prepare_segment_reversed(seg_dir, seg_rev_dir, num_frames)
    prompts_rev: List[PromptBox] = []
    for p in prompts_local:
        rev_frame = num_frames - 1 - p.frame_local
        prompts_rev.append(
            PromptBox(
                frame_global=p.frame_global,
                frame_local=rev_frame,
                xyxy=p.xyxy,
                label=p.label,
                obj_group=p.obj_group,
                score=p.score,
            )
        )

    inference_state_bwd = predictor.init_state(video_path=seg_rev_dir, offload_video_to_cpu=True)
    predictor.reset_state(inference_state_bwd)
    _add_prompts(predictor, inference_state_bwd, prompts_rev)
    max_prompt = max(p.frame_local for p in prompts_rev)
    backward: Dict[int, Dict[int, TrackRecord]] = defaultdict(dict)
    for frame_rev, obj_ids, mask_logits in _run_propagation(
        predictor,
        inference_state_bwd,
        start_frame=max_prompt,
        max_frames=num_frames,
        use_cuda=use_cuda,
    ):
        orig_frame = num_frames - 1 - int(frame_rev)
        for i, obj_id in enumerate(obj_ids):
            bbox = _mask_logits_to_xyxy(mask_logits[i])
            conf = float(torch.sigmoid(mask_logits[i]).max().item()) if bbox is not None else 0.0
            backward[int(obj_id)][orig_frame] = TrackRecord(
                xyxy=bbox if bbox is not None else None,
                conf=conf,
                enabled=bbox is not None,
            )

    # Fuse
    fused: Dict[int, Dict[int, TrackRecord]] = defaultdict(dict)
    for obj_id in {p.obj_group for p in prompts_local}:
        frames = sorted(set(forward.get(obj_id, {}).keys()) | set(backward.get(obj_id, {}).keys()))
        prev_box: Optional[np.ndarray] = None
        for f in frames:
            f_rec = forward.get(obj_id, {}).get(f)
            b_rec = backward.get(obj_id, {}).get(f)
            chosen: Optional[TrackRecord] = None
            if f_rec and f_rec.enabled and b_rec and b_rec.enabled:
                if prev_box is not None:
                    iou_f = _compute_iou(prev_box, f_rec.xyxy) if f_rec.xyxy is not None else 0.0
                    iou_b = _compute_iou(prev_box, b_rec.xyxy) if b_rec.xyxy is not None else 0.0
                else:
                    iou_f = iou_b = 0.0
                if (iou_f, f_rec.conf) >= (iou_b, b_rec.conf):
                    chosen = f_rec
                else:
                    chosen = b_rec
            elif f_rec and f_rec.enabled:
                chosen = f_rec
            elif b_rec and b_rec.enabled:
                chosen = b_rec
            else:
                chosen = f_rec or b_rec or TrackRecord(xyxy=None, conf=0.0, enabled=False)
            if chosen.xyxy is not None:
                prev_box = chosen.xyxy
            fused[obj_id][f] = chosen
    return fused


# -------------------------- Merge duplicate local tracklets -------------------------- #
def _merge_local_tracklets(
    fused: Dict[int, Dict[int, TrackRecord]],
    sample_frames: int = 10,
    iou_thresh: float = 0.6,
    dist_gate: float = 0.2,
) -> Dict[int, Dict[int, TrackRecord]]:
    obj_ids = sorted(fused.keys())
    if len(obj_ids) <= 1:
        return fused

    def track_frames(obj_id: int) -> List[int]:
        return sorted(fused[obj_id].keys())

    merged_map: Dict[int, int] = {oid: oid for oid in obj_ids}
    for i, oid_a in enumerate(obj_ids):
        for oid_b in obj_ids[i + 1 :]:
            frames_inter = sorted(set(track_frames(oid_a)) & set(track_frames(oid_b)))
            if not frames_inter:
                continue
            if len(frames_inter) > sample_frames:
                idx = np.linspace(0, len(frames_inter) - 1, num=sample_frames, dtype=int)
                frames_use = [frames_inter[k] for k in idx]
            else:
                frames_use = frames_inter

            ious: List[float] = []
            dist_ok = True
            for f in frames_use:
                a = fused[oid_a][f].xyxy
                b = fused[oid_b][f].xyxy
                if a is None or b is None:
                    continue
                ious.append(_compute_iou(a, b))
                cx_a, cy_a = _xyxy_center(a)
                cx_b, cy_b = _xyxy_center(b)
                diag = math.sqrt(_xyxy_area(a)) + 1e-6
                if diag > 0:
                    if math.hypot(cx_a - cx_b, cy_a - cy_b) / diag > dist_gate:
                        dist_ok = False
                        break
            if not dist_ok or not ious:
                continue
            if float(np.mean(ious)) > iou_thresh:
                # merge b into a
                merged_map[oid_b] = oid_a

    out: Dict[int, Dict[int, TrackRecord]] = defaultdict(dict)
    for oid, frames in fused.items():
        target = merged_map.get(oid, oid)
        for f, rec in frames.items():
            existing = out[target].get(f)
            if existing is None or rec.conf > existing.conf:
                out[target][f] = rec
            elif existing.xyxy is None and rec.xyxy is not None:
                out[target][f] = rec
            else:
                # ambiguity bump
                existing.ambiguity = min(1.0, existing.ambiguity + 0.1)
    return out


# -------------------------- Stitching across segments -------------------------- #
def _compute_match_cost(
    old_track: Dict[int, TrackRecord],
    new_track: Dict[int, TrackRecord],
    frames_sample: List[int],
    width: int,
    height: int,
    w_iou: float = 1.0,
    w_dist: float = 0.5,
    w_size: float = 0.2,
) -> float:
    ious: List[float] = []
    dists: List[float] = []
    sizes: List[float] = []
    for f in frames_sample:
        a = old_track.get(f)
        b = new_track.get(f)
        if a is None or b is None or a.xyxy is None or b.xyxy is None:
            continue
        ious.append(_compute_iou(a.xyxy, b.xyxy))
        cx_a, cy_a = _xyxy_center(a.xyxy)
        cx_b, cy_b = _xyxy_center(b.xyxy)
        diag = math.sqrt(width * width + height * height) + 1e-6
        dists.append(math.hypot(cx_a - cx_b, cy_a - cy_b) / diag)
        area_a = _xyxy_area(a.xyxy) + 1e-6
        area_b = _xyxy_area(b.xyxy) + 1e-6
        sizes.append(abs(math.log(area_a / area_b)))
    if not ious:
        return 1e6
    mean_iou = float(np.mean(ious))
    mean_dist = float(np.mean(dists)) if dists else 1.0
    mean_size = float(np.mean(sizes)) if sizes else 0.0
    if mean_iou < 0.1 and mean_dist > 0.2:
        return 1e6
    return w_iou * (1.0 - mean_iou) + w_dist * mean_dist + w_size * mean_size


def _match_segments(
    global_tracks: Dict[int, Dict[int, TrackRecord]],
    local_tracks: Dict[int, Dict[int, TrackRecord]],
    overlap_frames: List[int],
    width: int,
    height: int,
) -> Tuple[Dict[int, int], Dict[int, float]]:
    old_ids = sorted([gid for gid, frames in global_tracks.items() if any(f in frames for f in overlap_frames)])
    new_ids = sorted([lid for lid, frames in local_tracks.items() if any(f in frames for f in overlap_frames)])
    if not old_ids or not new_ids:
        return {}, {lid: 1.0 for lid in new_ids}
    sample_frames = overlap_frames
    if len(sample_frames) > 12:
        idx = np.linspace(0, len(sample_frames) - 1, num=12, dtype=int)
        sample_frames = [sample_frames[i] for i in idx]
    cost = np.full((len(old_ids), len(new_ids)), 1e6, dtype=np.float64)
    for i, gid in enumerate(old_ids):
        for j, lid in enumerate(new_ids):
            cost[i, j] = _compute_match_cost(
                global_tracks[gid],
                local_tracks[lid],
                sample_frames,
                width,
                height,
            )
    row_to_col, _ = _hungarian(cost)
    matches: Dict[int, int] = {}
    ambiguity: Dict[int, float] = {}
    for i, gid in enumerate(old_ids):
        j = row_to_col[i]
        if j < 0 or j >= len(new_ids):
            continue
        lid = new_ids[j]
        matches[lid] = gid
        row = cost[i]
        sorted_row = np.sort(row)
        best = row[j]
        second = sorted_row[1] if sorted_row.shape[0] > 1 else (best + 1.0)
        margin = second - best
        ambiguity[lid] = float(np.clip(1.0 - margin, 0.0, 1.0))
    # Unmatched locals ambiguity high
    for lid in new_ids:
        if lid not in ambiguity:
            ambiguity[lid] = 1.0
    return matches, ambiguity


# -------------------------- Prediction building -------------------------- #
def _build_prediction(
    tracks: Dict[int, Dict[int, TrackRecord]],
    labels: Dict[int, str],
    width: int,
    height: int,
    frames_count: int,
    fps: float,
) -> Dict[str, Any]:
    duration = frames_count / fps if fps > 0 else 0.0
    results: List[Dict[str, Any]] = []
    for gid, frames in tracks.items():
        if not frames:
            continue
        sequence = []
        ambiguities = []
        confs = []
        for frame_idx in sorted(frames.keys()):
            rec = frames[frame_idx]
            if rec.xyxy is None:
                continue
            x_pct, y_pct, w_pct, h_pct = xyxy_to_percent(rec.xyxy, width, height)
            item = {
                "frame": int(frame_idx + 1),
                "x": x_pct,
                "y": y_pct,
                "width": w_pct,
                "height": h_pct,
                "enabled": rec.enabled,
                "rotation": 0,
                "time": frame_idx / fps if fps > 0 else 0.0,
            }
            sequence.append(item)
            ambiguities.append(rec.ambiguity)
            confs.append(rec.conf)
        if not sequence:
            continue
        score = float(np.clip(1.0 - (np.mean(ambiguities) if ambiguities else 0.0), 0.0, 1.0))
        if confs:
            score *= float(np.clip(np.mean(confs), 0.0, 1.0))
        results.append(
            {
                "id": f"auto-track-{gid}",
                "type": "videorectangle",
                "from_name": "box",
                "to_name": "video",
                "score": score,
                "origin": "manual",
                "value": {
                    "sequence": sequence,
                    "framesCount": frames_count,
                    "duration": duration,
                    "labels": [labels.get(gid, "person")],
                },
                "meta": {"text": "id:"},
            }
        )
        base._ensure_meta_text_placeholder(results[-1])
    return {"result": results, "score": 1.0, "model_version": "sam2-video-boxes"}


# -------------------------- Main pipeline -------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunked SAM2 video tracking with overlap, bidirectional fusion, Hungarian stitching",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--ls-url", required=True)
    parser.add_argument("--ls-api-key", required=True)
    parser.add_argument("--project", type=int, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--annotation", type=int, required=True)
    parser.add_argument("--segment-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--keep-temp", action="store_true", help="Keep tmp_total_frames after run")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dry-run", action="store_true", help="Print prediction JSON instead of upload")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    temp_root = os.path.abspath(".")
    temp_total_dir = os.path.join(temp_root, "tmp_total_frames")
    temp_segments_root = os.path.join(temp_root, "tmp_segments")
    _ensure_dir(temp_segments_root)

    exit_code = 0
    try:
        ls = base._build_ls_client(args.ls_url, args.ls_api_key)
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

        jpeg_quality = 90
        _extract_all_frames(video_path, temp_total_dir, jpeg_quality)

        manual_prompts = _parse_manual_boxes(annotation, width, height)
        logger.info("Parsed %d manual boxes from annotation", len(manual_prompts))

        predictor = _build_sam2_video_predictor()
        _disable_sam2_progress_bars()

        global_tracks: Dict[int, Dict[int, TrackRecord]] = {}
        global_labels: Dict[int, str] = {}
        next_gid = 0

        # seed globals for first segment prompts
        for p in manual_prompts:
            gid = next_gid
            next_gid += 1
            global_labels[gid] = p.label
            global_tracks[gid] = {p.frame_global: TrackRecord(xyxy=p.xyxy, conf=1.0, enabled=True)}

        # Pre-compute segment starts for progress bar
        core_starts: List[int] = []
        cs = 0
        while cs < frames_count:
            core_starts.append(cs)
            ce = min(cs + args.segment_size - 1, frames_count - 1)
            cs = ce + 1 - args.overlap
            if cs < 0:
                cs = ce + 1
        segment_pbar = tqdm(total=len(core_starts), desc="Segments", unit="seg")

        segment_idx = 0
        core_start = 0
        while core_start < frames_count:
            core_end = min(core_start + args.segment_size - 1, frames_count - 1)
            seg_start = max(0, core_start - args.overlap)
            seg_end = min(frames_count - 1, core_end + args.overlap)
            seg_dir = os.path.join(temp_segments_root, f"seg_{core_start:06d}")
            seg_rev_dir = os.path.join(temp_segments_root, f"seg_{core_start:06d}_rev")
            num_frames = _prepare_segment(temp_total_dir, seg_dir, seg_start, seg_end)

            # collect prompts in this segment
            prompts_in_seg = [p for p in manual_prompts if seg_start <= p.frame_global <= seg_end]
            for p in prompts_in_seg:
                p.frame_local = p.frame_global - seg_start
            prompts_by_frame: Dict[int, List[PromptBox]] = defaultdict(list)
            for p in prompts_in_seg:
                prompts_by_frame[p.frame_local].append(p)
            merged_prompts: List[PromptBox] = []
            for _, boxes in prompts_by_frame.items():
                merged_prompts.extend(_cluster_prompts_same_frame(boxes))

            logger.debug(
                "Segment %d core=[%d,%d] seg=[%d,%d] prompts=%d",
                segment_idx,
                core_start,
                core_end,
                seg_start,
                seg_end,
                len(merged_prompts),
            )

            fused_local = _track_segment_bidirectional(
                predictor=predictor,
                seg_dir=seg_dir,
                seg_rev_dir=seg_rev_dir,
                seg_start=seg_start,
                seg_end=seg_end,
                prompts_local=merged_prompts,
            )
            fused_local = _merge_local_tracklets(fused_local)

            # Map local frames to global
            local_global_tracks: Dict[int, Dict[int, TrackRecord]] = defaultdict(dict)
            for lid, frames in fused_local.items():
                for f_local, rec in frames.items():
                    g = seg_start + f_local
                    # keep only core frames later
                    local_global_tracks[lid][g] = rec

            overlap_frames = list(range(core_start, min(core_end, core_start + args.overlap - 1) + 1))
            matches, ambiguity = _match_segments(global_tracks, local_global_tracks, overlap_frames, width, height)

            # commit frames for core region
            for lid, frames in local_global_tracks.items():
                gid = matches.get(lid)
                if gid is None:
                    gid = next_gid
                    next_gid += 1
                    global_labels[gid] = merged_prompts[0].label if merged_prompts else "person"
                for g_frame, rec in frames.items():
                    if g_frame < core_start or g_frame > core_end:
                        continue  # skip overlap to avoid dup
                    rec.ambiguity = ambiguity.get(lid, 1.0)
                    global_tracks.setdefault(gid, {})[g_frame] = rec

            logger.debug(
                "Segment %d done | local tracks=%d | global tracks=%d",
                segment_idx,
                len(local_global_tracks),
                len(global_tracks),
            )

            # cleanup segment dirs
            shutil.rmtree(seg_dir, ignore_errors=True)
            shutil.rmtree(seg_rev_dir, ignore_errors=True)
            segment_pbar.update(1)

            segment_idx += 1
            if core_end >= frames_count - 1:
                break
            core_start = core_end + 1 - args.overlap  # advance with overlap
            if core_start < 0:
                core_start = core_end + 1

        segment_pbar.close()
        prediction = _build_prediction(global_tracks, global_labels, width, height, frames_count, fps)
        if args.dry_run:
            print(json.dumps(prediction, indent=2))
        else:
            base._upload_prediction(ls, args.task, prediction)

    except InitialSeedingError as e:
        logger.error("Error: %s", e)
        exit_code = 1
    except Exception as e:  # pragma: no cover
        logger.error("Unexpected error: %s", e, exc_info=True)
        exit_code = 1
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_total_dir, ignore_errors=True)
        shutil.rmtree(temp_segments_root, ignore_errors=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
