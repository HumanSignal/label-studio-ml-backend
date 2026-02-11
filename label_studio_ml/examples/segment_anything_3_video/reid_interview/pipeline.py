"""ReID pipeline: annotation loading, embedding, clustering, pair generation.

Imports shared utilities from interview/ and seeding_common rather than
duplicating code.
"""

from __future__ import annotations

import copy
import logging
import os
import random
import sys
import uuid
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .state import CropInfo, ReIDInterviewSession, ReIDPairInfo, TrackInfo

logger = logging.getLogger(__name__)

# Ensure sibling modules are importable
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


# ---------------------------------------------------------------------------
# Lightweight coordinate conversion (avoids importing seeding_common at
# module level, which pulls in av, torch, etc.)
# ---------------------------------------------------------------------------

def _percent_xywh_to_xyxy_px(
    x_pct: float, y_pct: float, w_pct: float, h_pct: float,
    img_w: int, img_h: int,
) -> np.ndarray:
    """Convert LS percent coords to pixel xyxy."""
    x1 = (x_pct / 100.0) * img_w
    y1 = (y_pct / 100.0) * img_h
    x2 = x1 + (w_pct / 100.0) * img_w
    y2 = y1 + (h_pct / 100.0) * img_h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ===========================================================================
# Stage 1: Parse annotation into tracks + crops
# ===========================================================================

def parse_annotation_tracks(
    annotation_result: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> Tuple[List[TrackInfo], Dict[str, CropInfo]]:
    """Parse LS annotation result into TrackInfo + CropInfo objects.

    Only processes regions with type == "videorectangle".
    Returns (tracks, crops_dict) where crops_dict is keyed by crop_id.
    """

    tracks: List[TrackInfo] = []
    crops: Dict[str, CropInfo] = {}

    for region in annotation_result:
        if region.get("type") != "videorectangle":
            continue

        region_id = region["id"]
        value = region.get("value", {})
        sequence = value.get("sequence", [])
        labels = value.get("labels", [])
        meta = region.get("meta", {})
        meta_text = ""
        if isinstance(meta, dict):
            raw = meta.get("text", "")
            if isinstance(raw, list):
                meta_text = " ".join(str(t) for t in raw)
            else:
                meta_text = str(raw) if raw else ""

        track = TrackInfo(
            region_id=region_id,
            sequence=sequence,
            labels=labels,
            meta_text=meta_text,
            frames_count=value.get("framesCount", 0),
            duration=value.get("duration", 0.0),
        )
        tracks.append(track)

        for kf in sequence:
            frame = int(kf["frame"])
            x_pct = float(kf["x"])
            y_pct = float(kf["y"])
            w_pct = float(kf["width"])
            h_pct = float(kf["height"])

            xyxy_px = _percent_xywh_to_xyxy_px(x_pct, y_pct, w_pct, h_pct, img_w, img_h)
            crop_id = f"{region_id}_f{frame}"

            crops[crop_id] = CropInfo(
                crop_id=crop_id,
                track_region_id=region_id,
                frame_idx=frame,
                x_pct=x_pct,
                y_pct=y_pct,
                w_pct=w_pct,
                h_pct=h_pct,
                xyxy_px=xyxy_px,
            )

    return tracks, crops


# ===========================================================================
# Stage 2-6: Full pipeline (runs as background job)
# ===========================================================================

def run_reid_pipeline(session: ReIDInterviewSession, progress) -> Dict[str, Any]:
    """Complete ReID pipeline: load annotation → embed → cluster → pairs.

    Args:
        session: The ReID interview session.
        progress: JobProgress object for status reporting.

    Returns:
        Summary dict with pipeline results.
    """
    from seeding_common import (
        _build_ls_client, _fetch_task, _fetch_annotation,
        _get_video_path, _get_video_info_pyav, _read_frame_pyav,
    )

    progress.step = "Connecting to Label Studio..."
    progress.total = 7
    progress.current = 0

    ls_url = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL", "")
    ls_api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
    ls = _build_ls_client(ls_url, ls_api_key)

    # Stage 1: Load annotation
    progress.step = "Fetching annotation..."
    progress.current = 1
    task = _fetch_task(ls, session.project_id, session.task_id)
    video_path, video_key = _get_video_path(task)
    width, height, frames_count, fps = _get_video_info_pyav(video_path)

    ann = _fetch_annotation(ls, session.annotation_id)
    ann_result = getattr(ann, "result", []) or []

    with session._lock:
        session.video_path = video_path
        session.video_key = video_key
        session.width = width
        session.height = height
        session.frames_count = frames_count
        session.fps = fps

    tracks, crops = parse_annotation_tracks(ann_result, width, height)
    with session._lock:
        session.tracks = tracks
        session.crops = crops
        session.crop_id_list = sorted(crops.keys())
        session.touch()

    if len(crops) < 2:
        with session._lock:
            session.phase = "reviewing"
        return {"n_tracks": len(tracks), "n_crops": len(crops), "error": "Need at least 2 crops"}

    # Stage 2: Extract crop images
    progress.step = "Extracting crop images..."
    progress.current = 2

    # Group by frame for efficient video seeking
    frame_to_crops: Dict[int, List[CropInfo]] = defaultdict(list)
    for crop in crops.values():
        frame_to_crops[crop.frame_idx].append(crop)

    crop_images = {}  # crop_id -> PIL Image
    for frame_idx in sorted(frame_to_crops.keys()):
        pil_frame = _read_frame_pyav(video_path, frame_idx)
        if pil_frame is None:
            logger.warning("Could not read frame %d", frame_idx)
            continue
        for crop in frame_to_crops[frame_idx]:
            x1, y1, x2, y2 = [int(round(v)) for v in crop.xyxy_px]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(pil_frame.width, x2)
            y2 = min(pil_frame.height, y2)
            if x2 > x1 and y2 > y1:
                crop_images[crop.crop_id] = pil_frame.crop((x1, y1, x2, y2))

    # Stage 3: Compute DINOv3 embeddings
    progress.step = "Computing DINOv3 embeddings..."
    progress.current = 3

    from interview.dinov3_classifier import extract_features

    ordered_ids = sorted(crop_images.keys())
    ordered_pils = [crop_images[cid] for cid in ordered_ids]
    if ordered_pils:
        embeddings = extract_features(ordered_pils, batch_size=16)
        with session._lock:
            for i, cid in enumerate(ordered_ids):
                session.features[cid] = embeddings[i]

    # Stage 4: Compute HSV histograms
    progress.step = "Computing color histograms..."
    progress.current = 4

    from complete_reid import _compute_hist

    with session._lock:
        for cid in ordered_ids:
            if cid in crop_images:
                crop_rgb = np.array(crop_images[cid])
                hist = _compute_hist(crop_rgb, (8, 8, 8))
                session.histograms[cid] = hist

    # Stage 5: Build similarity matrix
    progress.step = "Computing similarity matrix..."
    progress.current = 5

    from interview.reid_phase import compute_fused_similarity

    crop_ids = [cid for cid in session.crop_id_list if cid in session.features and cid in session.histograms]
    with session._lock:
        session.crop_id_list = crop_ids

    n = len(crop_ids)
    sim_matrix = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_fused_similarity(
                session.features[crop_ids[i]],
                session.features[crop_ids[j]],
                session.histograms[crop_ids[i]],
                session.histograms[crop_ids[j]],
            )
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    with session._lock:
        session.similarity_matrix = sim_matrix

    # Stage 6: Cluster
    progress.step = "Clustering identities..."
    progress.current = 6

    from interview.reid_phase import spherical_kmeans, estimate_k

    feat_matrix = np.stack([session.features[cid] for cid in crop_ids])
    k = estimate_k(feat_matrix, k_range=(2, min(15, n - 1))) if n > 2 else 1
    if k < 1:
        k = 1

    if k == 1 or n <= 2:
        assignments = np.zeros(n, dtype=int)
    else:
        assignments = spherical_kmeans(feat_matrix, k)

    clusters: Dict[int, List[str]] = defaultdict(list)
    for i, cid in enumerate(crop_ids):
        clusters[int(assignments[i])].append(cid)

    with session._lock:
        session.clusters = dict(clusters)
        session.n_clusters = len(clusters)
        # Assign per-crop identity based on cluster
        for cluster_id, cids in clusters.items():
            for cid in cids:
                session.per_crop_identity[cid] = cluster_id

    # Stage 7: Generate pairs
    progress.step = "Generating comparison pairs..."
    progress.current = 7

    pairs = generate_pairs(session)
    with session._lock:
        session.pairs = pairs
        session.phase = "reviewing"
        session.touch()

    # Free crop images from memory
    del crop_images

    return {
        "n_tracks": len(tracks),
        "n_crops": len(crops),
        "n_clusters": len(clusters),
        "n_pairs": len(pairs),
    }


# ===========================================================================
# Pair generation with difficulty escalation + calibration
# ===========================================================================

def generate_pairs(session: ReIDInterviewSession) -> List[ReIDPairInfo]:
    """Generate pairs with auto-resolution, calibration checks, and difficulty ordering."""
    crop_ids = session.crop_id_list
    n = len(crop_ids)
    if n < 2:
        return []

    sim_matrix = session.similarity_matrix
    clusters = session.clusters
    if sim_matrix is None or len(clusters) == 0:
        return []

    # Build crop_id -> index mapping
    id_to_idx = {cid: i for i, cid in enumerate(crop_ids)}

    # Build crop_id -> cluster_id mapping
    crop_to_cluster = {}
    for cid_val, cids in clusters.items():
        for c in cids:
            crop_to_cluster[c] = cid_val

    # Categorize all cross-cluster pairs by similarity
    auto_same = []         # sim > 0.85, same cluster
    auto_different = []    # sim < 0.20, different cluster
    ambiguous = []         # everything else cross-cluster

    for i in range(n):
        for j in range(i + 1, n):
            cid_a = crop_ids[i]
            cid_b = crop_ids[j]
            sim = float(sim_matrix[i, j])
            cl_a = crop_to_cluster.get(cid_a, -1)
            cl_b = crop_to_cluster.get(cid_b, -1)

            if cl_a == cl_b:
                if sim > 0.85:
                    auto_same.append((cid_a, cid_b, sim))
            else:
                if sim < 0.20:
                    auto_different.append((cid_a, cid_b, sim))
                else:
                    ambiguous.append((cid_a, cid_b, sim))

    # Auto-resolve confident pairs
    session.auto_resolved = {}
    for (a, b, s) in auto_same:
        pid = str(uuid.uuid4())[:8]
        session.auto_resolved[pid] = "same"
    for (a, b, s) in auto_different:
        pid = str(uuid.uuid4())[:8]
        session.auto_resolved[pid] = "different"

    # Sample calibration checks (1-2 per auto-resolved group)
    calibration_pairs: List[ReIDPairInfo] = []

    # From auto_same: pick up to 2
    cal_same = random.sample(auto_same, min(2, len(auto_same))) if auto_same else []
    for (a, b, s) in cal_same:
        pair = ReIDPairInfo(
            pair_id=str(uuid.uuid4())[:8],
            crop_id_a=a,
            crop_id_b=b,
            track_a=session.crops[a].track_region_id,
            track_b=session.crops[b].track_region_id,
            similarity=s,
            pool="calibration",
            difficulty=0,
            model_prediction="same",
        )
        calibration_pairs.append(pair)
        session.calibration_answers[pair.pair_id] = "same"

    # From auto_different: pick up to 2
    cal_diff = random.sample(auto_different, min(2, len(auto_different))) if auto_different else []
    for (a, b, s) in cal_diff:
        pair = ReIDPairInfo(
            pair_id=str(uuid.uuid4())[:8],
            crop_id_a=a,
            crop_id_b=b,
            track_a=session.crops[a].track_region_id,
            track_b=session.crops[b].track_region_id,
            similarity=s,
            pool="calibration",
            difficulty=0,
            model_prediction="different",
        )
        calibration_pairs.append(pair)
        session.calibration_answers[pair.pair_id] = "different"

    # Build warmup pairs (high confidence cross-cluster, easy to judge)
    warmup_pairs: List[ReIDPairInfo] = []
    # Sort ambiguous by distance from 0.5 (most obvious first)
    sorted_amb = sorted(ambiguous, key=lambda x: abs(x[2] - 0.5), reverse=True)
    for (a, b, s) in sorted_amb[:5]:
        pred = "same" if s > 0.5 else "different"
        pair = ReIDPairInfo(
            pair_id=str(uuid.uuid4())[:8],
            crop_id_a=a,
            crop_id_b=b,
            track_a=session.crops[a].track_region_id,
            track_b=session.crops[b].track_region_id,
            similarity=s,
            pool="warmup",
            difficulty=0,
            model_prediction=pred,
        )
        warmup_pairs.append(pair)

    warmup_ids = {(p.crop_id_a, p.crop_id_b) for p in warmup_pairs}

    # Build ambiguous pairs (remaining, sorted by difficulty)
    ambiguous_pairs: List[ReIDPairInfo] = []
    remaining_amb = [(a, b, s) for (a, b, s) in sorted_amb[5:]
                     if (a, b) not in warmup_ids]
    # Sort by difficulty ascending (easiest ambiguous first)
    remaining_amb.sort(key=lambda x: abs(x[2] - 0.5), reverse=True)

    for i, (a, b, s) in enumerate(remaining_amb):
        # Difficulty: 0-2 based on proximity to 0.5
        dist = abs(s - 0.5)
        if dist > 0.15:
            difficulty = 0
        elif dist > 0.05:
            difficulty = 1
        else:
            difficulty = 2

        pred = "same" if s > 0.5 else "different"
        pair = ReIDPairInfo(
            pair_id=str(uuid.uuid4())[:8],
            crop_id_a=a,
            crop_id_b=b,
            track_a=session.crops[a].track_region_id,
            track_b=session.crops[b].track_region_id,
            similarity=s,
            pool="ambiguous",
            difficulty=difficulty,
            model_prediction=pred,
        )
        ambiguous_pairs.append(pair)

    # Cap total pairs to avoid overwhelming the user
    max_ambiguous = 40
    if len(ambiguous_pairs) > max_ambiguous:
        ambiguous_pairs = ambiguous_pairs[:max_ambiguous]

    # Interleave: warmup → calibration scattered → ambiguous ascending
    return order_pairs_by_difficulty(warmup_pairs + calibration_pairs + ambiguous_pairs)


def order_pairs_by_difficulty(pairs: List[ReIDPairInfo]) -> List[ReIDPairInfo]:
    """Order pairs: warmup first, then calibration interleaved with ambiguous by difficulty."""
    warmup = [p for p in pairs if p.pool == "warmup"]
    calibration = [p for p in pairs if p.pool == "calibration"]
    ambiguous = [p for p in pairs if p.pool == "ambiguous"]

    # Sort ambiguous by difficulty ascending
    ambiguous.sort(key=lambda p: p.difficulty)

    # Interleave calibration into ambiguous at ~20% intervals
    result = list(warmup)
    if not ambiguous:
        result.extend(calibration)
        return result

    cal_interval = max(1, len(ambiguous) // (len(calibration) + 1)) if calibration else len(ambiguous) + 1
    cal_idx = 0
    for i, p in enumerate(ambiguous):
        if cal_idx < len(calibration) and i > 0 and i % cal_interval == 0:
            result.append(calibration[cal_idx])
            cal_idx += 1
        result.append(p)

    # Append remaining calibration
    while cal_idx < len(calibration):
        result.append(calibration[cal_idx])
        cal_idx += 1

    return result


# ===========================================================================
# Pair resolution + incremental cluster update
# ===========================================================================

def apply_pair_resolution(
    clusters: Dict[int, List[str]],
    resolution: Dict[str, Any],
) -> Dict[int, List[str]]:
    """Apply a single pair resolution to clusters.

    If resolution is "same", merge the clusters containing the two crops.
    Returns a new clusters dict with renumbered keys starting from 0.
    """
    res = resolution.get("resolution", "unsure")
    if res != "same":
        return copy.deepcopy(clusters)

    crop_a = resolution["crop_id_a"]
    crop_b = resolution["crop_id_b"]

    # Find which clusters they belong to
    cluster_a = None
    cluster_b = None
    for cid, members in clusters.items():
        if crop_a in members:
            cluster_a = cid
        if crop_b in members:
            cluster_b = cid

    if cluster_a is None or cluster_b is None:
        return copy.deepcopy(clusters)

    if cluster_a == cluster_b:
        return copy.deepcopy(clusters)

    # Merge cluster_b into cluster_a
    new_clusters = {}
    for cid, members in clusters.items():
        if cid == cluster_b:
            continue  # skip — will be merged
        if cid == cluster_a:
            new_clusters[cid] = list(members) + list(clusters[cluster_b])
        else:
            new_clusters[cid] = list(members)

    # Renumber from 0
    renumbered = {}
    for new_idx, (_, members) in enumerate(sorted(new_clusters.items())):
        renumbered[new_idx] = members

    return renumbered


# ===========================================================================
# Gamification helpers
# ===========================================================================

def compute_accuracy(
    resolutions: Dict[str, str],
    calibration_answers: Dict[str, str],
) -> float:
    """Compute accuracy of human resolutions on calibration pairs.

    Returns percentage (0-100).
    """
    correct = 0
    total = 0
    for pair_id, human_answer in resolutions.items():
        if pair_id in calibration_answers:
            total += 1
            if human_answer == calibration_answers[pair_id]:
                correct += 1
    if total == 0:
        return 0.0
    return (correct / total) * 100.0
