"""
CLI: Auto-detect objects via text prompts, track them across a video using SAM3,
and stitch tracklets into complete tracks using teacher confirmation or Hungarian matching.

Replaces the SAM2 + Grounding DINO pipeline with SAM3 from HuggingFace Transformers.
Video decoding uses PyAV (no disk-based JPEG extraction).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import av
import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm import tqdm
from PIL import Image

import seeding_common as base

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# PyAV video decode (replaces JPEG extraction to disk)
# ---------------------------------------------------------------------------

def _decode_segment_pyav(
    video_path: str,
    start_idx: int,
    end_idx: int,
) -> List[Image.Image]:
    """Decode frames [start_idx, end_idx] to PIL images in memory via PyAV."""
    container = av.open(video_path)
    stream = container.streams.video[0]

    if start_idx > 0 and stream.average_rate and stream.time_base:
        avg_fps = float(stream.average_rate)
        target_ts = int(start_idx / avg_fps / stream.time_base)
        container.seek(target_ts, stream=stream)

    frames: List[Image.Image] = []
    frame_idx = 0
    for packet in container.demux(stream):
        for frame in packet.decode():
            if frame_idx < start_idx:
                frame_idx += 1
                continue
            if frame_idx > end_idx:
                container.close()
                return frames
            frames.append(frame.to_image())
            frame_idx += 1

    container.close()
    return frames


# ---------------------------------------------------------------------------
# SAM3 segment tracking (replaces _track_segment with SAM2 predictor)
# ---------------------------------------------------------------------------

def _mask_to_xyxy(mask: torch.Tensor) -> Optional[np.ndarray]:
    """Extract xyxy bbox from a post-processed binary mask tensor."""
    mask_squeezed = mask.squeeze()
    if mask_squeezed.ndim != 2:
        return None
    if torch.is_tensor(mask_squeezed):
        mask_np = mask_squeezed.cpu().numpy().astype(np.uint8)
    else:
        mask_np = np.asarray(mask_squeezed, dtype=np.uint8)
    ys, xs = np.where(mask_np > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return np.array([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    dtype=np.float32)


def _track_segment_sam3(
    *,
    video_path: str,
    start_idx: int,
    end_idx: int,
    start_dets: List[base.KeyframeDetection],
    refine_seeds: bool = True,
    refine_search_scale: float = 1.3,
) -> List[Dict[str, Any]]:
    """Track objects through a segment using Sam3TrackerVideoModel (chunked batch).

    Decodes frames [start_idx, end_idx] into memory, seeds with start_dets
    at frame 0 of the session, and propagates forward.

    If refine_seeds is True, each detection box is refined using SAM3 text+box
    prompts before being used to seed the tracker. This helps when auto-detected
    boxes are imperfect.
    """
    if end_idx < start_idx or not start_dets:
        return []

    sam3_model, sam3_processor = base._get_sam3_tracker_model()

    # Decode frames for this segment
    frames_list = _decode_segment_pyav(video_path, start_idx, end_idx)
    num_frames = len(frames_list)
    if num_frames == 0:
        return []

    logger.debug("Tracking segment [%d, %d]: %d frames in memory", start_idx, end_idx, num_frames)

    # Refine seed boxes using text+box prompts (hybrid approach)
    if refine_seeds and frames_list:
        first_frame = frames_list[0]
        for det in start_dets:
            refined_box, score = base.refine_box_with_text_prompt(
                image=first_frame,
                box_xyxy=det.xyxy,
                text_label=det.label,
                search_scale=refine_search_scale,
            )
            if score > 0:
                logger.debug(
                    "Refined detection box for '%s': [%.1f,%.1f,%.1f,%.1f] -> [%.1f,%.1f,%.1f,%.1f]",
                    det.label,
                    det.xyxy[0], det.xyxy[1], det.xyxy[2], det.xyxy[3],
                    refined_box[0], refined_box[1], refined_box[2], refined_box[3],
                )
                det.xyxy = refined_box

    # Init session with all frames
    session = sam3_processor.init_video_session(
        video=frames_list, inference_device=base.DEVICE, dtype=base.DTYPE
    )

    # Add all start detections at session frame 0
    local_obj_ids: List[int] = []
    label_by_obj: Dict[int, str] = {}
    last_box_by_obj: Dict[int, np.ndarray] = {}

    for local_id, det in enumerate(start_dets):
        local_obj_ids.append(local_id)
        last_box_by_obj[local_id] = det.xyxy.copy()
        label_by_obj[local_id] = det.label

        inputs = sam3_processor(images=frames_list[0], device=base.DEVICE, return_tensors="pt")
        sam3_processor.add_inputs_to_inference_session(
            session,
            frame_idx=0,
            obj_ids=[local_id],
            input_boxes=[[det.xyxy.tolist()]],
            original_size=inputs.original_sizes[0],
        )

    # Propagate
    _, height_img, width_img = frames_list[0].size[0], frames_list[0].size[1], frames_list[0].size[0]
    # PIL Image.size is (width, height)
    width_img = frames_list[0].size[0]
    height_img = frames_list[0].size[1]

    sequences: Dict[int, List[Dict[str, Any]]] = {oid: [] for oid in local_obj_ids}

    with torch.inference_mode():
        for output in sam3_model.propagate_in_video_iterator(session):
            session_idx = output.frame_idx
            global_frame = start_idx + session_idx

            masks = sam3_processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[height_img, width_img]],
                binarize=True,
            )[0]

            if output.object_ids is not None:
                for i, obj_id in enumerate(output.object_ids):
                    obj_id_int = int(obj_id)
                    if obj_id_int not in sequences:
                        continue
                    bbox = _mask_to_xyxy(masks[i])
                    enabled = True
                    if bbox is not None:
                        last_box_by_obj[obj_id_int] = bbox
                    else:
                        enabled = False

                    sequences[obj_id_int].append({
                        "frame": global_frame,
                        "xyxy": last_box_by_obj[obj_id_int].copy(),
                        "enabled": enabled,
                    })

    # Build tracklets
    tracklets: List[Dict[str, Any]] = []
    for obj_id in local_obj_ids:
        seq = sequences.get(obj_id) or []
        if not seq:
            continue
        start_box = start_dets[obj_id].xyxy.copy()
        global_track_id = start_dets[obj_id].track_id
        if global_track_id is None:
            raise base.InitialSeedingError("Missing track_id on start detection")
        visible_at_end = bool(seq[-1].get("enabled", True))
        end_box = seq[-1]["xyxy"].copy()
        tracklets.append({
            "local_id": obj_id,
            "track_id": int(global_track_id),
            "label": label_by_obj.get(obj_id, "object"),
            "start_frame": start_idx,
            "end_frame": end_idx,
            "start_box": start_box,
            "end_box": end_box,
            "visible_at_end": visible_at_end,
            "sequence": seq,
        })

    return tracklets


# ---------------------------------------------------------------------------
# Teacher confirmation at segment boundaries (Sam3Model embeddings)
# ---------------------------------------------------------------------------

def _get_crop_embedding(
    pil_image: Image.Image,
    box_xyxy: np.ndarray,
    sam3_model,
    sam3_processor,
) -> Optional[np.ndarray]:
    """Get SAM3 image embedding for a cropped box region."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(pil_image.width, x2)
    y2 = min(pil_image.height, y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    crop = pil_image.crop((x1, y1, x2, y2))
    embed = base._extract_sam3_image_embedding(sam3_model, sam3_processor, crop)
    return embed.detach().cpu().numpy().flatten()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _match_tracklets_teacher(
    *,
    tracklets: List[Dict[str, Any]],
    end_dets: List[base.KeyframeDetection],
    boundary_frame: Image.Image,
    merge_threshold: float,
    width: int,
    height: int,
) -> Dict[int, int]:
    """Match tracklet endpoints to boundary detections using Sam3Model teacher embeddings.

    Uses cosine similarity of cropped-box embeddings. Only matches with
    similarity >= merge_threshold are accepted. Prefers over-fragmentation
    over ID switches.
    """
    if not tracklets or not end_dets:
        return {}

    require_visible = base.DEVICE  # reuse env check pattern
    eligible = [
        idx for idx, trk in enumerate(tracklets)
        if bool(trk.get("visible_at_end", True))
    ]
    if not eligible:
        return {}

    sam3_model, sam3_processor = base._get_sam3_image_model()

    # Compute embeddings for tracklet end boxes
    track_embeds: Dict[int, Optional[np.ndarray]] = {}
    for idx in eligible:
        end_box = tracklets[idx]["end_box"]
        track_embeds[idx] = _get_crop_embedding(boundary_frame, end_box, sam3_model, sam3_processor)

    # Compute embeddings for detections
    det_embeds: Dict[int, Optional[np.ndarray]] = {}
    for j, det in enumerate(end_dets):
        det_embeds[j] = _get_crop_embedding(boundary_frame, det.xyxy, sam3_model, sam3_processor)

    # Build cost matrix (1 - cosine_sim) with spatial distance as tiebreaker
    diag = float(math.sqrt(float(width * width + height * height)))
    if diag <= 0:
        diag = 1.0

    pairs: List[Tuple[float, int, int]] = []
    for idx in eligible:
        t_embed = track_embeds.get(idx)
        if t_embed is None:
            continue
        t_box = tracklets[idx]["end_box"]
        t_center = _xyxy_center(t_box)

        for j in range(len(end_dets)):
            d_embed = det_embeds.get(j)
            if d_embed is None:
                continue
            sim = _cosine_similarity(t_embed, d_embed)
            if sim < merge_threshold:
                continue

            # Spatial distance as tiebreaker
            d_center = _xyxy_center(end_dets[j].xyxy)
            dist = math.sqrt((t_center[0] - d_center[0])**2 + (t_center[1] - d_center[1])**2)
            dist_norm = dist / diag

            cost = (1.0 - sim) + 0.1 * dist_norm
            pairs.append((cost, idx, j))

    # Greedy matching (lowest cost first)
    pairs.sort(key=lambda x: x[0])
    used_t: set = set()
    used_d: set = set()
    matches: Dict[int, int] = {}
    for cost, ti, dj in pairs:
        if ti in used_t or dj in used_d:
            continue
        used_t.add(ti)
        used_d.add(dj)
        matches[ti] = dj
        logger.debug("Teacher match: tracklet %d -> det %d (cost=%.3f)", ti, dj, cost)

    unmatched_t = len(eligible) - len(matches)
    unmatched_d = len(end_dets) - len(used_d)
    if unmatched_t > 0 or unmatched_d > 0:
        logger.info("Teacher stitching: %d matches, %d unmatched tracks (new endpoints), "
                     "%d unmatched dets (new tracks)", len(matches), unmatched_t, unmatched_d)

    return matches


# ---------------------------------------------------------------------------
# Segment management helpers (preserved, no SAM dependency)
# ---------------------------------------------------------------------------

def _augment_keyframes_for_max_segment(
    *,
    keyframes: List[int],
    frames_count: int,
    max_segment_frames: int,
) -> List[int]:
    if frames_count <= 0:
        return []

    max_segment_frames = int(max_segment_frames)
    if max_segment_frames <= 1:
        cleaned_set: set[int] = set()
        for k in keyframes:
            ki = int(k)
            if 0 <= ki < frames_count:
                cleaned_set.add(ki)
        return sorted(cleaned_set)

    cleaned_set: set[int] = set()
    for k in keyframes:
        ki = int(k)
        if 0 <= ki < frames_count:
            cleaned_set.add(ki)
    cleaned = sorted(cleaned_set)

    if not cleaned:
        cleaned = [0, frames_count - 1]
    else:
        if cleaned[0] != 0:
            cleaned = [0] + cleaned
        if cleaned[-1] != frames_count - 1:
            cleaned = cleaned + [frames_count - 1]

    step = max_segment_frames - 1
    augmented: List[int] = [cleaned[0]]
    for start_kf, end_kf in zip(cleaned[:-1], cleaned[1:]):
        cur = int(start_kf)
        end_kf_int = int(end_kf)
        while end_kf_int - cur + 1 > max_segment_frames:
            cur = cur + step
            if cur >= end_kf_int:
                break
            augmented.append(cur)
        augmented.append(end_kf_int)

    return sorted(set(augmented))


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Label handling helpers (preserved)
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


def _validate_prediction_region_labels(
    *,
    prediction: Dict[str, Any],
    expected_single_label: Optional[str] = None,
    allowed_labels: Optional[set[str]] = None,
) -> None:
    results = prediction.get("result")
    if not isinstance(results, list):
        raise base.InitialSeedingError("Prediction is missing a valid 'result' list")

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
        raise base.InitialSeedingError(
            "Prediction contains regions with missing/blank labels: " + ", ".join(missing[:10])
        )

    if unknown:
        raise base.InitialSeedingError(
            "Prediction contains regions with labels not present in the project label config: "
            + ", ".join(unknown[:10])
        )


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


# ---------------------------------------------------------------------------
# Geometry helpers (preserved)
# ---------------------------------------------------------------------------

def _xyxy_area(xyxy: np.ndarray) -> float:
    x0, y0, x1, y1 = xyxy
    return float(max(0.0, x1 - x0) * max(0.0, y1 - y0))


def _xyxy_center(xyxy: np.ndarray) -> Tuple[float, float]:
    x0, y0, x1, y1 = xyxy
    return (float(x0 + x1) * 0.5, float(y0 + y1) * 0.5)


def _xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0 = max(float(ax0), float(bx0))
    iy0 = max(float(ay0), float(by0))
    ix1 = min(float(ax1), float(bx1))
    iy1 = min(float(ay1), float(by1))

    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = _xyxy_area(a)
    area_b = _xyxy_area(b)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


# ---------------------------------------------------------------------------
# Sparsification (preserved)
# ---------------------------------------------------------------------------

def _sparsify_visible_run(
    run: List[Dict[str, Any]],
    *,
    iou_thresh: float,
    max_interval: int,
) -> List[Dict[str, Any]]:
    if not run:
        return []

    kept: List[Dict[str, Any]] = []
    last_kept_box: Optional[np.ndarray] = None
    last_kept_frame: Optional[int] = None

    for item in run:
        frame = int(item["frame"])
        box = np.array(item["xyxy"], dtype=np.float32)
        if last_kept_box is None:
            kept.append({"frame": frame, "xyxy": box.copy(), "enabled": True})
            last_kept_box = box
            last_kept_frame = frame
            continue

        if max_interval > 0 and last_kept_frame is not None and frame - last_kept_frame >= max_interval:
            kept.append({"frame": frame, "xyxy": box.copy(), "enabled": True})
            last_kept_box = box
            last_kept_frame = frame
            continue

        iou = _xyxy_iou(last_kept_box, box)
        if iou < iou_thresh:
            kept.append({"frame": frame, "xyxy": box.copy(), "enabled": True})
            last_kept_box = box
            last_kept_frame = frame

    last_item = run[-1]
    last_frame = int(last_item["frame"])
    if not kept or int(kept[-1]["frame"]) != last_frame:
        kept.append({
            "frame": last_frame,
            "xyxy": np.array(last_item["xyxy"], dtype=np.float32).copy(),
            "enabled": True,
        })

    return kept


def _sparsify_track_sequence_zero_based(
    *,
    sequence: List[Dict[str, Any]],
    frames_count: int,
) -> List[Dict[str, Any]]:
    if not sequence:
        return []
    if frames_count <= 0:
        return sequence

    iou_thresh = float(os.getenv("SPARSE_IOU_THRESH", "0.2"))
    max_interval = int(os.getenv("SPARSE_MAX_INTERVAL", "0"))

    seq_sorted = sorted(sequence, key=lambda x: int(x["frame"]))
    sparse: List[Dict[str, Any]] = []
    run: List[Dict[str, Any]] = []

    def _flush_run(*, off_frame: Optional[int]) -> None:
        nonlocal run, sparse
        if not run:
            return
        sparse.extend(_sparsify_visible_run(run, iou_thresh=iou_thresh, max_interval=max_interval))
        run = []
        if off_frame is None:
            return
        if not sparse:
            return
        last_box = np.array(sparse[-1]["xyxy"], dtype=np.float32)
        sparse.append({"frame": int(off_frame), "xyxy": last_box.copy(), "enabled": False})

    for item in seq_sorted:
        enabled = bool(item.get("enabled", True))
        if enabled:
            run.append(item)
            continue

        if run:
            _flush_run(off_frame=int(item["frame"]))
        continue

    if run:
        sparse.extend(_sparsify_visible_run(run, iou_thresh=iou_thresh, max_interval=max_interval))
        last_visible = int(run[-1]["frame"])
        if not sparse:
            return []
        if last_visible < frames_count - 1:
            last_box = np.array(sparse[-1]["xyxy"], dtype=np.float32)
            sparse.append({"frame": last_visible + 1, "xyxy": last_box.copy(), "enabled": False})
        else:
            sparse[-1]["enabled"] = False

    return sparse


# ---------------------------------------------------------------------------
# Hungarian stitching (preserved from SAM2, optional mode)
# ---------------------------------------------------------------------------

def _match_tracklets_to_end_dets(
    *,
    tracklets: List[Dict[str, Any]],
    end_dets: List[base.KeyframeDetection],
    width: int,
    height: int,
) -> Dict[int, int]:
    if not tracklets or not end_dets:
        return {}

    require_visible_at_end = _get_env_bool("STITCH_REQUIRE_VISIBLE_AT_END", True)
    eligible = [
        idx
        for idx, trk in enumerate(tracklets)
        if (not require_visible_at_end) or bool(trk.get("visible_at_end", True))
    ]
    if not eligible:
        return {}

    track_boxes = np.stack([tracklets[idx]["end_box"] for idx in eligible], axis=0).astype(np.float32)
    det_boxes = np.stack([d.xyxy for d in end_dets], axis=0).astype(np.float32)

    ious = (
        box_iou(torch.from_numpy(track_boxes), torch.from_numpy(det_boxes))
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    diag = float(math.sqrt(float(width * width + height * height)))
    if diag <= 0:
        diag = 1.0

    track_centers = np.array([_xyxy_center(b) for b in track_boxes], dtype=np.float32)
    det_centers = np.array([_xyxy_center(b) for b in det_boxes], dtype=np.float32)
    dists = np.linalg.norm(track_centers[:, None, :] - det_centers[None, :, :], axis=-1)
    dist_norm = (dists / diag).astype(np.float32)

    track_areas = np.array([_xyxy_area(b) for b in track_boxes], dtype=np.float32)
    det_areas = np.array([_xyxy_area(b) for b in det_boxes], dtype=np.float32)
    eps = np.float32(1e-6)
    area_ratio = np.maximum(
        (track_areas[:, None] + eps) / (det_areas[None, :] + eps),
        (det_areas[None, :] + eps) / (track_areas[:, None] + eps),
    ).astype(np.float32)
    size_cost = np.abs(np.log((track_areas[:, None] + eps) / (det_areas[None, :] + eps))).astype(np.float32)

    w_iou = float(os.getenv("STITCH_W_IOU", "1.0"))
    w_dist = float(os.getenv("STITCH_W_DIST", "1.0"))
    w_size = float(os.getenv("STITCH_W_SIZE", "0.2"))
    cost = (w_iou * (1.0 - ious)) + (w_dist * dist_norm) + (w_size * size_cost)

    iou_min = float(os.getenv("STITCH_IOU_MIN", "0.3"))
    dist_max = float(os.getenv("STITCH_DIST_MAX", "0.15"))
    area_ratio_max = float(os.getenv("STITCH_AREA_RATIO_MAX", "2.5"))
    max_cost = float(os.getenv("STITCH_MAX_COST", "1.2"))

    candidate = (ious >= iou_min) | ((dist_norm <= dist_max) & (area_ratio <= area_ratio_max))
    pairs: List[Tuple[float, int, int]] = []
    for ti in range(cost.shape[0]):
        for dj in range(cost.shape[1]):
            if not bool(candidate[ti, dj]):
                continue
            c = float(cost[ti, dj])
            if max_cost > 0 and c > max_cost:
                continue
            pairs.append((c, ti, dj))

    pairs.sort(key=lambda x: x[0])
    used_t: set[int] = set()
    used_d: set[int] = set()
    matches_local: Dict[int, int] = {}
    for _, ti, dj in pairs:
        if ti in used_t or dj in used_d:
            continue
        used_t.add(ti)
        used_d.add(dj)
        matches_local[ti] = dj

    return {eligible[ti]: dj for ti, dj in matches_local.items()}


# ---------------------------------------------------------------------------
# Track finalization & prediction building (preserved)
# ---------------------------------------------------------------------------

def _finalize_tracks(
    *,
    global_tracks: Dict[int, Dict[int, Dict[str, Any]]],
    track_labels: Dict[int, str],
) -> List[Dict[str, Any]]:
    tracks: List[Dict[str, Any]] = []
    for tid, frame_map in global_tracks.items():
        merged_seq = [
            {
                "frame": int(f),
                "xyxy": np.array(data["xyxy"], dtype=np.float32),
                "enabled": bool(data.get("enabled", True)),
            }
            for f, data in sorted(frame_map.items())
        ]

        last_enabled_idx = -1
        for i, item in enumerate(merged_seq):
            if item.get("enabled", True):
                last_enabled_idx = i

        if last_enabled_idx < 0:
            continue

        trim_end = last_enabled_idx + 1
        if trim_end < len(merged_seq) and not merged_seq[trim_end].get("enabled", True):
            trim_end += 1
        merged_seq = merged_seq[:trim_end]

        tracks.append({
            "track_id": int(tid),
            "sequence": merged_seq,
            "label": track_labels.get(int(tid), "object"),
        })

    return tracks


def _build_prediction_zero_based(
    *,
    tracks: List[Dict[str, Any]],
    width: int,
    height: int,
    frames_count: int,
    fps: float,
    use_sparse: Optional[bool] = None,
) -> Dict[str, Any]:
    converted: List[Dict[str, Any]] = []
    if use_sparse is None:
        use_sparse = _get_env_bool("SPARSE_SEQUENCE", True)
    for tr in tracks:
        seq_0b = tr.get("sequence", [])
        if use_sparse:
            seq_0b = _sparsify_track_sequence_zero_based(sequence=seq_0b, frames_count=frames_count)
        seq_1b = []
        for item in seq_0b:
            seq_1b.append({
                "frame": int(item["frame"]) + 1,
                "xyxy": np.array(item["xyxy"], dtype=np.float32),
                "enabled": bool(item.get("enabled", True)),
            })
        converted.append({
            "track_id": int(tr.get("track_id", 0)),
            "sequence": seq_1b,
            "label": tr.get("label", "object"),
        })

    return base._build_prediction(converted, width, height, frames_count=frames_count, fps=fps)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initial seeding pipeline using SAM3: text detection + video tracking per segment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ls-url", required=True, help="Label Studio URL")
    parser.add_argument("--ls-api-key", required=True, help="Label Studio API key")
    parser.add_argument("--project", type=int, required=True, help="Project ID")
    parser.add_argument("--task", type=int, required=True, help="Task ID")
    parser.add_argument("--annotation", type=int, required=True, help="Annotation ID")
    parser.add_argument(
        "--embedding-batch",
        type=int,
        default=int(os.getenv("EMBED_BATCH", "8")),
        help="Batch size for SAM3 embedding computation",
    )
    parser.add_argument(
        "--keyframe-frac",
        type=float,
        default=0.1,
        help="Fraction of frames to keep as keyframes (default 0.1 => 10%%)",
    )
    parser.add_argument(
        "--min-spacing",
        type=int,
        default=30,
        help="Minimum spacing between high-change keyframes",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.getenv("CACHE_DIR", "./cache_dir/joblib"),
        help="Cache directory for embeddings",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Class-of-interest text prompt for detection (overrides env PROMPT_TEXT).",
    )
    parser.add_argument(
        "--stitch-mode",
        choices=["teacher", "hungarian"],
        default="teacher",
        help="Stitching mode: 'teacher' (Sam3Model embedding comparison, default) "
             "or 'hungarian' (IoU + distance cost matrix).",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.6,
        help="Cosine similarity threshold for teacher stitching (default: 0.6). "
             "Higher = stricter matching, more fragmentation. Only used with --stitch-mode=teacher.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Save prediction to JSON file instead of uploading.",
    )

    sparse_group = parser.add_mutually_exclusive_group()
    sparse_group.add_argument(
        "--sparse-sequence",
        dest="sparse_sequence",
        action="store_true",
        help="Enable sparse sequence generation.",
    )
    sparse_group.add_argument(
        "--no-sparse-sequence",
        dest="sparse_sequence",
        action="store_false",
        help="Disable sparse sequence generation.",
    )
    parser.set_defaults(sparse_sequence=None)

    parser.add_argument(
        "--no-refine-seeds", action="store_false", dest="refine_seeds", default=True,
        help="Disable seed box refinement (refinement is enabled by default)"
    )
    parser.add_argument(
        "--refine-search-scale", type=float, default=1.3,
        help="Search scale for seed box refinement (default: 1.3 = 30%% expansion)"
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    prompt_label = _prompt_to_single_label(args.prompt)

    exit_code = 0
    try:
        ls = base._build_ls_client(args.ls_url, args.ls_api_key)
        project_labels = _fetch_project_labels(ls, args.project)
        if project_labels:
            logger.info("Project label config contains %d label(s)", len(project_labels))

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
        _ = base._fetch_annotation(ls, args.annotation)

        video_path, _video_key = base._get_video_path(task)
        keyframes, width, height, frames_count, fps = base._detect_keyframes(
            video_path=video_path,
            cache_dir=args.cache_dir,
            cache_key=f"{task['id']}",
            embedding_batch=args.embedding_batch,
            keyframe_frac=args.keyframe_frac,
            min_spacing=args.min_spacing,
        )
        if frames_count <= 0:
            raise base.InitialSeedingError("Video has no frames")

        keyframes_raw = sorted({int(k) for k in keyframes} | {0, frames_count - 1})
        max_segment_frames = int(os.getenv("MAX_SEGMENT_FRAMES", "1024"))
        keyframes = _augment_keyframes_for_max_segment(
            keyframes=keyframes_raw,
            frames_count=frames_count,
            max_segment_frames=max_segment_frames,
        )
        if len(keyframes) < 2:
            raise base.InitialSeedingError("Need at least two keyframes to form segments")

        if len(keyframes) != len(keyframes_raw):
            logger.info(
                "Augmented keyframes from %d to %d to enforce max_segment_frames=%d",
                len(keyframes_raw),
                len(keyframes),
                max_segment_frames,
            )

        seg_lens = [int(b) - int(a) + 1 for a, b in zip(keyframes[:-1], keyframes[1:])]
        if seg_lens and max(seg_lens) > max_segment_frames:
            raise base.InitialSeedingError(
                f"Keyframe augmentation failed: max segment length {max(seg_lens)} > "
                f"max_segment_frames={max_segment_frames}"
            )

        logger.info(
            "Video frames=%d | keyframes=%d | segments=%d | segment_len[min=%d max=%d avg=%.2f]",
            frames_count,
            len(keyframes),
            max(0, len(keyframes) - 1),
            min(seg_lens) if seg_lens else 0,
            max(seg_lens) if seg_lens else 0,
            float(sum(seg_lens)) / float(len(seg_lens)) if seg_lens else 0.0,
        )

        # Text detection on keyframes (replaces Grounding DINO)
        detections_by_frame = base._run_text_detection_on_keyframes(video_path, keyframes, args.prompt)
        for k in keyframes:
            detections_by_frame.setdefault(k, [])

        if prompt_label is not None:
            for dets in detections_by_frame.values():
                for det in dets:
                    det.label = prompt_label

        # Initialize global tracking state
        global_tracks: Dict[int, Dict[int, Dict[str, Any]]] = {}
        track_labels: Dict[int, str] = {}
        next_track_id = 0

        first_kf = keyframes[0]
        for det in detections_by_frame.get(first_kf, []):
            det.track_id = next_track_id
            track_labels[next_track_id] = det.label
            global_tracks.setdefault(next_track_id, {})[first_kf] = {
                "xyxy": det.xyxy.copy(),
                "enabled": True,
            }
            next_track_id += 1

        segment_total = max(0, len(keyframes) - 1)
        segment_bar = tqdm(total=segment_total, desc="Segments (track+stitch)", unit="seg")

        try:
            for start_kf, end_kf in zip(keyframes[:-1], keyframes[1:]):
                start_dets = detections_by_frame.get(start_kf, [])
                end_dets = detections_by_frame.get(end_kf, [])

                for det in start_dets:
                    if det.track_id is None:
                        det.track_id = next_track_id
                        track_labels[next_track_id] = det.label
                        next_track_id += 1
                    global_tracks.setdefault(int(det.track_id), {})[start_kf] = {
                        "xyxy": det.xyxy.copy(),
                        "enabled": True,
                    }

                # Track segment using SAM3 (in-memory, no temp dirs)
                tracklets = _track_segment_sam3(
                    video_path=video_path,
                    start_idx=start_kf,
                    end_idx=end_kf,
                    start_dets=start_dets,
                    refine_seeds=args.refine_seeds,
                    refine_search_scale=args.refine_search_scale,
                )

                segment_bar.update(1)

                # Stitch: match tracklet endpoints to detections at boundary
                if args.stitch_mode == "teacher":
                    # Read boundary frame for teacher embedding comparison
                    boundary_pil = base._read_frame_pyav(video_path, end_kf)
                    if boundary_pil is not None:
                        matches = _match_tracklets_teacher(
                            tracklets=tracklets,
                            end_dets=end_dets,
                            boundary_frame=boundary_pil,
                            merge_threshold=args.merge_threshold,
                            width=width,
                            height=height,
                        )
                    else:
                        logger.warning("Could not read boundary frame %d; falling back to hungarian", end_kf)
                        matches = _match_tracklets_to_end_dets(
                            tracklets=tracklets,
                            end_dets=end_dets,
                            width=width,
                            height=height,
                        )
                else:
                    matches = _match_tracklets_to_end_dets(
                        tracklets=tracklets,
                        end_dets=end_dets,
                        width=width,
                        height=height,
                    )

                matched_tracklets = set(matches.keys())
                for ti, dj in matches.items():
                    tid = int(tracklets[ti]["track_id"])
                    end_dets[dj].track_id = tid
                    if tid not in track_labels:
                        track_labels[tid] = end_dets[dj].label
                    global_tracks.setdefault(tid, {})[end_kf] = {
                        "xyxy": end_dets[dj].xyxy.copy(),
                        "enabled": True,
                    }

                for det in end_dets:
                    if det.track_id is not None:
                        continue
                    det.track_id = next_track_id
                    track_labels[next_track_id] = det.label
                    global_tracks.setdefault(next_track_id, {})[end_kf] = {
                        "xyxy": det.xyxy.copy(),
                        "enabled": True,
                    }
                    next_track_id += 1

                for ti, trk in enumerate(tracklets):
                    if ti not in matched_tracklets:
                        for item in trk.get("sequence", []):
                            if int(item.get("frame", -1)) == end_kf:
                                item["enabled"] = False

                    tid = int(trk["track_id"])
                    frame_map = global_tracks.setdefault(tid, {})
                    for item in trk.get("sequence", []):
                        frame_idx = int(item["frame"])
                        if frame_idx == start_kf:
                            continue
                        existing = frame_map.get(frame_idx)
                        if existing is not None and existing.get("enabled", True):
                            continue
                        frame_map[frame_idx] = {
                            "xyxy": np.array(item["xyxy"], dtype=np.float32),
                            "enabled": bool(item.get("enabled", True)),
                        }

        finally:
            segment_bar.close()

        tracks = _finalize_tracks(global_tracks=global_tracks, track_labels=track_labels)
        prediction = _build_prediction_zero_based(
            tracks=tracks,
            width=width,
            height=height,
            frames_count=frames_count,
            fps=fps,
            use_sparse=args.sparse_sequence,
        )

        _validate_prediction_region_labels(
            prediction=prediction,
            expected_single_label=prompt_label,
            allowed_labels=set(project_labels) if project_labels else None,
        )

        if args.dry_run:
            output_file = f"prediction_task_{args.task}.json"
            with open(output_file, "w") as f:
                json.dump(prediction, f, indent=2)
            logger.info("DRY RUN: Prediction saved to %s", output_file)
        else:
            base._upload_prediction(ls, args.task, prediction)

    except base.InitialSeedingError as e:
        logger.error("Initial seeding error: %s", e)
        exit_code = 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        exit_code = 130
    except Exception as e:  # pragma: no cover
        logger.error("Unexpected error: %s", e, exc_info=True)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
