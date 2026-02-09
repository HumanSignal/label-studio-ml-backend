"""Dense seed generation and Label Studio upload for the Interview UI.

Implements Phase 3 of the interview workflow: after the user has trained an
MLP classifier (detection phase) and resolved identity clusters (ReID phase),
this module scans every Nth frame of the video, detects candidate bounding
boxes, classifies them with the trained MLP, assigns identities via nearest
ReID cluster centroid, and uploads the results to Label Studio as
videorectangle regions with ``enabled=false`` keyframes.

Functions:
    generate_seeds  -- run the full detection+classification+identity pipeline
    upload_seeds    -- push seed regions to Label Studio as a prediction
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seeding_common import (
    _build_ls_client, _build_prediction, _upload_prediction,
    _read_frame_pyav, xyxy_to_percent, DEVICE, DTYPE,
)

from .state import CropData, CropLabel, InterviewSession, Phase
from .cache_manager import save_session, load_model
from .background import JobProgress
from .dinov3_classifier import extract_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_cluster_centroids(
    session: InterviewSession,
) -> Dict[int, np.ndarray]:
    """Compute mean DINOv3 feature vector per ReID cluster.

    Iterates over ``session.reid_clusters`` (mapping of cluster_id to list of
    crop_ids), collects the DINOv3 feature vector stored on each
    :class:`CropData`, and averages them to produce a single centroid per
    identity cluster.

    Args:
        session: The current interview session with populated ``reid_clusters``
            and ``crops`` (each crop should have a ``.features`` array).

    Returns:
        Dictionary mapping cluster_id (int) to the L2-normalised centroid
        vector of shape ``(feat_dim,)`` (typically 1024 for DINOv3).
    """
    centroids: Dict[int, np.ndarray] = {}

    for cluster_id, crop_ids in session.reid_clusters.items():
        feature_vectors: List[np.ndarray] = []
        for cid in crop_ids:
            crop = session.get_crop(cid)
            if crop is not None and crop.features is not None:
                feature_vectors.append(crop.features.astype(np.float32))

        if not feature_vectors:
            logger.warning(
                "Cluster %d has no crops with features; skipping centroid",
                cluster_id,
            )
            continue

        centroid = np.mean(feature_vectors, axis=0)
        # L2-normalise so cosine similarity reduces to a dot product
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[cluster_id] = centroid

    logger.info(
        "Computed centroids for %d / %d clusters",
        len(centroids),
        len(session.reid_clusters),
    )
    return centroids


def _assign_identity(
    features: np.ndarray,
    centroids: Dict[int, np.ndarray],
) -> Tuple[int, float]:
    """Assign a detection to the nearest ReID cluster centroid.

    Computes cosine similarity between the candidate feature vector and every
    cluster centroid, returning the cluster with the highest similarity.

    Args:
        features: DINOv3 CLS-token vector for the candidate crop, shape
            ``(feat_dim,)``.
        centroids: Mapping of cluster_id to L2-normalised centroid vector (as
            returned by :func:`_compute_cluster_centroids`).

    Returns:
        Tuple of ``(cluster_id, similarity)`` for the best-matching cluster.
        If *centroids* is empty, returns ``(-1, 0.0)``.
    """
    if not centroids:
        return -1, 0.0

    # Normalise the query vector for cosine similarity
    feat = features.astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm

    best_id = -1
    best_sim = -1.0

    for cluster_id, centroid in centroids.items():
        sim = float(np.dot(feat, centroid))
        if sim > best_sim:
            best_sim = sim
            best_id = cluster_id

    return best_id, best_sim


# ---------------------------------------------------------------------------
# Dual-proposer configuration (Path B & C)
# ---------------------------------------------------------------------------

_REFINE_THRESHOLD = float(os.getenv("INTERVIEW_REFINE_THRESHOLD", "0.3"))
_ENABLE_REFINEMENT = os.getenv("INTERVIEW_ENABLE_REFINEMENT", "true").lower() == "true"
_ENABLE_GRID_SEARCH = os.getenv("INTERVIEW_ENABLE_GRID_SEARCH", "true").lower() == "true"
_GRID_SCALE = float(os.getenv("INTERVIEW_GRID_SCALE", "0.10"))
_GRID_SIM_THRESHOLD = float(os.getenv("INTERVIEW_GRID_SIM_THRESHOLD", "0.5"))
_GRID_TOP_K = int(os.getenv("INTERVIEW_GRID_TOP_K", "5"))
_SEED_CHUNK_SIZE = int(os.getenv("INTERVIEW_SEED_CHUNK_SIZE", "100"))


def _get_sam3_image_model():
    """Import and return the Sam3Model singleton from seeding_common."""
    from seeding_common import _get_sam3_image_model as _get
    return _get()


def _refine_candidates_sam3(
    frames: Dict[int, Any],
    candidates: List[Tuple[int, np.ndarray, float]],
    prompt: str = "person",
    expand_frac: float = 0.2,
) -> List[Tuple[int, np.ndarray, float]]:
    """Refine candidate boxes using Sam3Model with text+box prompts.

    For each candidate, expands the box by *expand_frac* on each side,
    runs Sam3Model with combined text + box prompt, and extracts the tight
    bounding box from the best mask.

    Args:
        frames:      Mapping of frame_idx -> decoded PIL Image.
        candidates:  List of (frame_idx, box_xyxy, det_score).
        prompt:      Text prompt for Sam3Model (e.g., "person").
        expand_frac: Fraction to expand each side of the box.

    Returns:
        List of (frame_idx, refined_box_xyxy, det_score) for successful refinements.
    """
    import torch
    model, processor = _get_sam3_image_model()

    refined: List[Tuple[int, np.ndarray, float]] = []

    for frame_idx, box, det_score in candidates:
        pil_frame = frames.get(frame_idx)
        if pil_frame is None:
            continue

        w, h = pil_frame.size
        x1, y1, x2, y2 = box

        # Expand box
        bw, bh = x2 - x1, y2 - y1
        dx, dy = bw * expand_frac, bh * expand_frac
        ex1 = max(0, int(x1 - dx))
        ey1 = max(0, int(y1 - dy))
        ex2 = min(w, int(x2 + dx))
        ey2 = min(h, int(y2 + dy))

        if ex2 <= ex1 or ey2 <= ey1:
            continue

        try:
            inputs = processor(
                images=pil_frame,
                text=prompt,
                input_boxes=[[[ex1, ey1, ex2, ey2]]],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(DEVICE)

            with torch.inference_mode():
                outputs = model(**inputs)

            target_sizes = inputs.get("original_sizes")
            if target_sizes is not None:
                if hasattr(target_sizes, "tolist"):
                    target_sizes = target_sizes.tolist()
                # else already a plain list
            else:
                target_sizes = [[h, w]]

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=target_sizes,
            )[0]

            masks = results.get("masks", [])
            scores = results.get("scores", [])
            boxes_out = results.get("boxes", [])

            if not masks and not boxes_out:
                continue

            n_results = max(len(masks), len(boxes_out))
            if n_results == 0:
                continue

            best_idx = int(np.argmax([
                s.item() if hasattr(s, "item") else float(s) for s in scores
            ])) if scores else 0

            if best_idx < len(boxes_out):
                b = boxes_out[best_idx]
                tight = np.array(b.tolist() if hasattr(b, "tolist") else list(b), dtype=np.float32)
            elif best_idx < len(masks):
                mask = masks[best_idx]
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                ys, xs = np.where(mask > 0)
                if xs.size == 0:
                    continue
                tight = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
            else:
                continue

            refined.append((frame_idx, tight, det_score))

        except Exception as exc:
            logger.warning("Refinement failed for frame %d: %s", frame_idx, exc)
            continue

    return refined


# ---------------------------------------------------------------------------
# DINOv3 grid search fallback (Path C)
# ---------------------------------------------------------------------------

def _grid_search_fallback(
    frame,
    frame_idx: int,
    reference_features: np.ndarray,
    scale_frac: float = _GRID_SCALE,
    stride_frac: float = 0.5,
    top_k: int = _GRID_TOP_K,
    sim_threshold: float = _GRID_SIM_THRESHOLD,
) -> List[Tuple[np.ndarray, float]]:
    """Find person-like regions using DINOv3 feature similarity.

    Tiles the frame into a grid at a single scale, extracts DINOv3 CLS tokens,
    and compares to the mean feature vector of accepted crops.

    Args:
        frame:              PIL Image (or mock with .size and .crop).
        frame_idx:          Frame index (for logging only).
        reference_features: (N_ref, 1024) L2-normalized features of accepted crops.
        scale_frac:         Grid cell size as fraction of frame width/height.
        stride_frac:        Stride as fraction of cell size (0.5 = 50% overlap).
        top_k:              Maximum number of candidates to return.
        sim_threshold:      Minimum cosine similarity to be considered.

    Returns:
        List of (box_xyxy, similarity) sorted by descending similarity.
    """
    W, H = frame.size
    cw = max(16, int(W * scale_frac))
    ch = max(16, int(H * scale_frac))
    sx = max(8, int(cw * stride_frac))
    sy = max(8, int(ch * stride_frac))

    # Compute mean reference feature
    mean_ref = np.mean(reference_features, axis=0)
    norm = np.linalg.norm(mean_ref)
    if norm > 1e-8:
        mean_ref /= norm

    grid_crops = []
    grid_boxes: List[np.ndarray] = []

    for y0 in range(0, H - ch + 1, sy):
        for x0 in range(0, W - cw + 1, sx):
            grid_crops.append(frame.crop((x0, y0, x0 + cw, y0 + ch)))
            grid_boxes.append(np.array([x0, y0, x0 + cw, y0 + ch], dtype=np.float32))

    if not grid_crops:
        return []

    feats = extract_features(grid_crops, batch_size=32)  # (N, 1024)
    sims = feats @ mean_ref  # cosine sim (both L2-normed)

    candidates = []
    for j in range(len(grid_boxes)):
        if sims[j] >= sim_threshold:
            candidates.append((grid_boxes[j], float(sims[j])))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _score_and_accept_seed(
    box: np.ndarray,
    pil_frame,
    classifier,
    centroids: Dict[int, np.ndarray],
    threshold: float,
    source: str,
    frame_idx: int,
) -> Optional[Dict[str, Any]]:
    """Crop, extract features, score with MLP, and return seed dict if accepted.

    Shared helper for all three paths (A/B/C) to avoid code duplication.
    Returns None if the crop fails to pass the MLP threshold.
    """
    import torch
    from .dinov3_classifier import compute_crop_metadata

    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(pil_frame.width, x2), min(pil_frame.height, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = pil_frame.crop((x1, y1, x2, y2))
    feat = extract_features([crop])  # (1, 1024)
    meta = compute_crop_metadata(box, pil_frame.width, pil_frame.height)
    mlp_in = np.concatenate([feat, meta.reshape(1, -1)], axis=1)

    with torch.inference_mode():
        p = torch.sigmoid(
            classifier(torch.from_numpy(mlp_in).float())
        ).item()

    if p < threshold:
        return None

    identity, identity_sim = _assign_identity(feat[0], centroids)
    return {
        "frame_idx": int(frame_idx),
        "xyxy": box.tolist(),
        "confidence": round(p, 4),
        "identity": int(identity),
        "identity_similarity": round(float(identity_sim), 4),
        "source": source,
    }


def generate_seeds(
    session: InterviewSession,
    progress: JobProgress,
) -> Dict[str, Any]:
    """Generate dense seeds with dual-proposer pipeline.

    Three paths for maximum coverage:

      A. SAM3 text detection -> MLP quality gate (primary, all frames)
      B. Sam3Model box refinement for medium-confidence detections (fallback 1)
      C. DINOv3 grid search for frames with zero detections (fallback 2)

    Results are stored in ``session.seeds`` as a list of dicts::

        {
            "frame_idx": int,
            "xyxy": [x1, y1, x2, y2],
            "confidence": float,
            "identity": int,
            "identity_similarity": float,
            "source": str,              # "path_a", "path_b", or "path_c"
        }

    Args:
        session: The interview session (must be at REID phase or later,
            with a trained MLP model on disk).
        progress: :class:`JobProgress` handle for reporting status to the
            frontend polling loop.

    Returns:
        Summary dict with ``total_seeds``, ``identities``, and per-identity
        counts.

    Raises:
        RuntimeError: If the MLP model has not been trained yet or if no
            ReID clusters have been defined.
    """
    from .detection import (
        Sam3TextBasedDetector, nms_numpy, pad_boxes,
        _decode_frames_sequential,
    )
    from .dinov3_classifier import CropClassifier, compute_crop_metadata

    import torch

    # ---- Validate prerequisites ----
    progress.step = "Validating session state..."
    progress.current = 0

    state_dict = load_model(session.cache_key)
    if state_dict is None:
        raise RuntimeError(
            "No trained MLP model found. Complete the classification phase first."
        )
    if not session.reid_clusters:
        raise RuntimeError(
            "No ReID clusters found. Complete the ReID phase first."
        )

    # ---- Load models ----
    progress.step = "Loading models..."
    detector = Sam3TextBasedDetector()

    classifier = CropClassifier()
    classifier.load_state_dict(state_dict)
    classifier.eval()

    centroids = _compute_cluster_centroids(session)

    # ---- Compute accepted-crop reference features for grid search ----
    accepted = session.get_crops_by_label(CropLabel.ACCEPTED)
    ref_list = [c.features for c in accepted if c.features is not None]
    reference_features = (
        np.stack(ref_list) if ref_list
        else np.empty((0, 1024), dtype=np.float32)
    )

    # ---- Determine target frames ----
    interval = max(1, session.seed_config.frame_interval)
    uniform = set(range(0, session.frames_count, interval))
    change = set(session.change_keyframes) if session.embedding_complete else set()
    all_targets = sorted(uniform | change)
    total_frames = len(all_targets)

    progress.step = "Generating seeds..."
    progress.total = total_frames
    progress.current = 0

    logger.info(
        "Seed generation: scanning %d frames (interval=%d, threshold=%.2f, "
        "change_keyframes=%d, refinement=%s, grid_search=%s)",
        total_frames, interval, session.seed_config.confidence_threshold,
        len(change), _ENABLE_REFINEMENT, _ENABLE_GRID_SEARCH,
    )

    prompt_text = session.prompts[0] if session.prompts else "person"
    threshold = session.seed_config.confidence_threshold
    seeds: List[Dict[str, Any]] = []

    # ---- Process in chunks ----
    for chunk_start in range(0, total_frames, _SEED_CHUNK_SIZE):
        chunk_indices = all_targets[chunk_start:chunk_start + _SEED_CHUNK_SIZE]
        progress.step = (
            f"Decoding frames {chunk_start + 1}"
            f"-{chunk_start + len(chunk_indices)} / {total_frames}..."
        )

        # 1. Batch decode chunk via seek-based approach
        frames = _decode_frames_sequential(session.video_path, chunk_indices)

        no_seed_frames: set = set()  # frames with no Path A seed; fed to Path C
        medium_candidates: List[Tuple[int, np.ndarray, float]] = []

        # 2. Path A: SAM3 text detection per frame
        for fi, frame_idx in enumerate(chunk_indices):
            progress.current = chunk_start + fi + 1
            pil_frame = frames.get(frame_idx)
            if pil_frame is None:
                no_seed_frames.add(frame_idx)
                continue

            detections = detector.detect(prompt_text, pil_image=pil_frame)
            if not detections:
                no_seed_frames.add(frame_idx)
                continue

            boxes = np.array([d["xyxy"] for d in detections], dtype=np.float32)
            det_scores = np.array([d["score"] for d in detections], dtype=np.float32)
            boxes = pad_boxes(boxes, pil_frame.width, pil_frame.height)
            keep = nms_numpy(boxes, det_scores, iou_threshold=0.5)
            boxes, det_scores = boxes[keep], det_scores[keep]

            if len(boxes) == 0:
                no_seed_frames.add(frame_idx)
                continue

            # Crop + DINOv3 features + MLP
            crop_images = []
            valid_indices = []
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(pil_frame.width, x2), min(pil_frame.height, y2)
                if x2 > x1 and y2 > y1:
                    crop_images.append(pil_frame.crop((x1, y1, x2, y2)))
                    valid_indices.append(idx)

            if not crop_images:
                no_seed_frames.add(frame_idx)
                continue

            boxes = boxes[valid_indices]
            det_scores = det_scores[valid_indices]
            crop_features = extract_features(crop_images)
            metadata = np.array([
                compute_crop_metadata(b, pil_frame.width, pil_frame.height)
                for b in boxes
            ], dtype=np.float32)
            mlp_input = np.concatenate([crop_features, metadata], axis=1)

            with torch.inference_mode():
                probs = torch.sigmoid(
                    classifier(torch.from_numpy(mlp_input).float())
                ).squeeze(-1).cpu().numpy()

            frame_has_seed = False
            for i in range(len(boxes)):
                conf = float(probs[i]) if probs.ndim > 0 else float(probs)
                if conf >= threshold:
                    identity, identity_sim = _assign_identity(
                        crop_features[i], centroids,
                    )
                    seeds.append({
                        "frame_idx": int(frame_idx),
                        "xyxy": boxes[i].tolist(),
                        "confidence": round(conf, 4),
                        "identity": int(identity),
                        "identity_similarity": round(float(identity_sim), 4),
                        "source": "path_a",
                    })
                    frame_has_seed = True
                elif _ENABLE_REFINEMENT and conf >= _REFINE_THRESHOLD:
                    medium_candidates.append((frame_idx, boxes[i], det_scores[i]))

            if not frame_has_seed:
                no_seed_frames.add(frame_idx)

        # 3. Path B: Refine medium-confidence candidates
        if medium_candidates and _ENABLE_REFINEMENT:
            progress.step = f"Refining {len(medium_candidates)} candidates..."
            refined = _refine_candidates_sam3(
                frames, medium_candidates, prompt=prompt_text,
            )
            for frame_idx, box, _det_score in refined:
                pil_frame = frames.get(frame_idx)
                if pil_frame is None:
                    continue
                seed = _score_and_accept_seed(
                    box, pil_frame, classifier, centroids,
                    threshold, "path_b", frame_idx,
                )
                if seed is not None:
                    seeds.append(seed)
                    no_seed_frames.discard(frame_idx)

        # 4. Path C: Grid search for frames with zero seeds
        if (no_seed_frames and _ENABLE_GRID_SEARCH
                and reference_features.shape[0] > 0):
            progress.step = f"Grid search on {len(no_seed_frames)} frames..."
            for frame_idx in no_seed_frames:
                pil_frame = frames.get(frame_idx)
                if pil_frame is None:
                    continue
                grid_candidates = _grid_search_fallback(
                    pil_frame, frame_idx, reference_features,
                    scale_frac=_GRID_SCALE,
                    sim_threshold=_GRID_SIM_THRESHOLD,
                    top_k=_GRID_TOP_K,
                )
                if not grid_candidates:
                    continue
                # Refine grid candidates via Sam3Model
                grid_for_refine = [
                    (frame_idx, box, sim) for box, sim in grid_candidates
                ]
                refined_grid = (
                    _refine_candidates_sam3(
                        frames, grid_for_refine, prompt=prompt_text,
                    ) if _ENABLE_REFINEMENT else grid_for_refine
                )
                for fidx, box, _ in refined_grid:
                    pil_f = frames.get(fidx)
                    if pil_f is None:
                        continue
                    seed = _score_and_accept_seed(
                        box, pil_f, classifier, centroids,
                        threshold, "path_c", fidx,
                    )
                    if seed is not None:
                        seeds.append(seed)

    # ---- Finalise ----
    with session._lock:
        session.seeds = seeds
        session.advance_to(Phase.SEEDING)

    save_session(session)

    identity_counts: Dict[int, int] = {}
    for seed in seeds:
        ident = seed["identity"]
        identity_counts[ident] = identity_counts.get(ident, 0) + 1

    summary = {
        "total_seeds": len(seeds),
        "frames_scanned": total_frames,
        "identities": identity_counts,
    }

    logger.info(
        "Seed generation complete: %d seeds across %d identities",
        len(seeds), len(identity_counts),
    )
    progress.step = f"Done - {len(seeds)} seeds generated"
    return summary


def upload_seeds(
    session: InterviewSession,
    progress: JobProgress,
) -> Dict[str, Any]:
    """Upload seed regions to Label Studio with ``enabled=false`` keyframes.

    Each ReID identity becomes one Label Studio *videorectangle* region.
    Every seed detection for that identity becomes a keyframe entry in the
    region's ``sequence`` array, with ``enabled: false`` to prevent Label
    Studio from auto-interpolating between keyframes.

    The tracks structure expected by :func:`_build_prediction`::

        tracks = [
            {
                "track_id": identity_id,
                "label": "person",
                "sequence": [
                    {
                        "frame": frame_num,    # 1-based for LS
                        "xyxy": np.array([x1, y1, x2, y2]),
                        "enabled": False,
                    },
                    ...
                ],
            },
            ...
        ]

    Args:
        session: The interview session containing ``session.seeds`` (populated
            by :func:`generate_seeds`).
        progress: :class:`JobProgress` handle for status reporting.

    Returns:
        Upload result info including the number of tracks and keyframes
        pushed to Label Studio.

    Raises:
        RuntimeError: If no seeds have been generated yet.
        seeding_common.InitialSeedingError: If Label Studio connection fails.
    """
    progress.step = "Preparing upload..."
    progress.current = 0
    progress.total = 4

    if not session.seeds:
        raise RuntimeError(
            "No seeds to upload. Run seed generation first."
        )

    # ---- Group seeds by identity ----
    progress.step = "Grouping seeds by identity..."
    progress.current = 1

    identity_seeds: Dict[int, List[Dict[str, Any]]] = {}
    for seed in session.seeds:
        ident = seed["identity"]
        identity_seeds.setdefault(ident, []).append(seed)

    # Determine the label text for tracks
    label_text = session.prompts[0] if session.prompts else "person"

    # ---- Build track structures ----
    progress.step = "Building track structures..."
    progress.current = 2

    tracks: List[Dict[str, Any]] = []
    for identity_id, id_seeds in sorted(identity_seeds.items()):
        # Sort seeds by frame for a coherent sequence
        id_seeds_sorted = sorted(id_seeds, key=lambda s: s["frame_idx"])

        sequence: List[Dict[str, Any]] = []
        for seed in id_seeds_sorted:
            sequence.append({
                "frame": seed["frame_idx"] + 1,  # convert 0-based to 1-based
                "xyxy": np.array(seed["xyxy"], dtype=np.float32),
                "enabled": False,
            })

        tracks.append({
            "track_id": identity_id,
            "label": label_text,
            "sequence": sequence,
        })

    total_keyframes = sum(len(t["sequence"]) for t in tracks)
    logger.info(
        "Upload: %d tracks, %d total keyframes",
        len(tracks),
        total_keyframes,
    )

    # ---- Build LS prediction payload ----
    progress.step = "Building prediction payload..."
    progress.current = 3

    prediction = _build_prediction(
        tracks=tracks,
        width=session.width,
        height=session.height,
        frames_count=session.frames_count,
        fps=session.fps,
    )

    # ---- Connect and upload ----
    progress.step = "Uploading to Label Studio..."
    progress.current = 4

    ls_url = (
        os.getenv("LABEL_STUDIO_HOST")
        or os.getenv("LABEL_STUDIO_URL", "")
    )
    ls_api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
    ls = _build_ls_client(ls_url, ls_api_key)

    _upload_prediction(ls, session.task_id, prediction)

    # ---- Finalise ----
    result = {
        "tracks_uploaded": len(tracks),
        "total_keyframes": total_keyframes,
        "identities": list(identity_seeds.keys()),
        "model_version": prediction.get("model_version", "sam3-init-seed"),
    }

    with session._lock:
        session.upload_result = result
        session.advance_to(Phase.COMPLETE)

    save_session(session)

    logger.info(
        "Upload complete: %d tracks (%d keyframes) for task %d",
        len(tracks),
        total_keyframes,
        session.task_id,
    )
    progress.step = f"Done - uploaded {len(tracks)} tracks"
    return result
