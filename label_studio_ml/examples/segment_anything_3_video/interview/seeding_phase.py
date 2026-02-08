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
# Public API
# ---------------------------------------------------------------------------

def generate_seeds(
    session: InterviewSession,
    progress: JobProgress,
) -> Dict[str, Any]:
    """Generate dense seeds across the entire video.

    Pipeline executed on a background thread (via :func:`submit_job`):

    1. Sample frames at the user-configured interval
       (``session.seed_config.frame_interval``).
    2. Run :class:`Sam3TextBasedDetector` on each sampled frame to obtain
       candidate bounding boxes.
    3. Extract DINOv3 features for each candidate crop.
    4. Score each crop with the trained MLP classifier.
    5. Filter: keep only predictions whose confidence exceeds
       ``session.seed_config.confidence_threshold``.
    6. Per-frame NMS to remove duplicate / overlapping detections.
    7. Assign each surviving detection to the nearest ReID cluster centroid
       using DINOv3 cosine similarity.

    Results are stored in ``session.seeds`` as a list of dicts::

        {
            "frame_idx": int,
            "xyxy": [x1, y1, x2, y2],   # pixel coords
            "confidence": float,          # MLP confidence
            "identity": int,              # ReID cluster ID
            "identity_similarity": float, # cosine sim to centroid
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
    from .detection import Sam3TextBasedDetector, nms_numpy, pad_boxes
    from .dinov3_classifier import extract_features, CropClassifier, compute_crop_metadata

    import torch
    from PIL import Image

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

    # Reconstruct the MLP classifier and load trained weights
    # Default input_dim=1028: 1024 DINOv3 features + 4 metadata
    classifier = CropClassifier()
    classifier.load_state_dict(state_dict)
    classifier.eval()

    # Compute ReID cluster centroids
    centroids = _compute_cluster_centroids(session)

    # ---- Determine frames to scan ----
    interval = max(1, session.seed_config.frame_interval)
    frame_indices = list(range(0, session.frames_count, interval))
    total_frames = len(frame_indices)

    progress.step = "Generating seeds..."
    progress.total = total_frames
    progress.current = 0

    logger.info(
        "Seed generation: scanning %d frames (interval=%d, threshold=%.2f)",
        total_frames,
        interval,
        session.seed_config.confidence_threshold,
    )

    prompt_text = session.prompts[0] if session.prompts else "person"
    threshold = session.seed_config.confidence_threshold
    seeds: List[Dict[str, Any]] = []

    # ---- Main scan loop ----
    for scan_idx, frame_idx in enumerate(frame_indices):
        progress.current = scan_idx + 1
        if total_frames > 20 and scan_idx % max(1, total_frames // 20) == 0:
            progress.step = (
                f"Scanning frame {frame_idx} "
                f"({scan_idx + 1}/{total_frames})"
            )

        # 1. Read frame
        pil_frame = _read_frame_pyav(session.video_path, frame_idx)
        if pil_frame is None:
            logger.debug("Could not read frame %d; skipping", frame_idx)
            continue

        # 2. Detect candidate boxes
        detections = detector.detect(prompt_text, pil_image=pil_frame)
        if not detections:
            continue

        # Collect candidate boxes for NMS (detect returns list of dicts)
        boxes = np.array([d["xyxy"] for d in detections], dtype=np.float32)
        det_scores = np.array(
            [d["score"] for d in detections], dtype=np.float32,
        )

        # 3. Pad boxes slightly to capture full person context
        boxes = pad_boxes(boxes, pil_frame.width, pil_frame.height)

        # 4. NMS to suppress duplicates
        keep = nms_numpy(boxes, det_scores, iou_threshold=0.5)
        boxes = boxes[keep]
        det_scores = det_scores[keep]

        if len(boxes) == 0:
            continue

        # 5. Crop PIL images and extract DINOv3 features
        crop_images: List[Image.Image] = []
        valid_indices: List[int] = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(pil_frame.width, x2)
            y2 = min(pil_frame.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue  # skip degenerate crops
            crop_images.append(pil_frame.crop((x1, y1, x2, y2)))
            valid_indices.append(idx)

        if not crop_images:
            continue

        # Filter boxes/scores to only valid crops
        boxes = boxes[valid_indices]
        det_scores = det_scores[valid_indices]

        crop_features = extract_features(crop_images)  # (N, 1024)

        # 6. Compute metadata and concat for MLP (1024 + 4 = 1028 dims)
        metadata = np.array([
            compute_crop_metadata(box, pil_frame.width, pil_frame.height)
            for box in boxes
        ], dtype=np.float32)  # (N, 4)
        mlp_input = np.concatenate([crop_features, metadata], axis=1)  # (N, 1028)

        # 7. Classify with trained MLP
        with torch.inference_mode():
            feat_tensor = torch.from_numpy(mlp_input).float()
            logits = classifier(feat_tensor)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

        # 8. Filter by confidence threshold and assign identity
        for i in range(len(boxes)):
            confidence = float(probs[i]) if probs.ndim > 0 else float(probs)
            if confidence < threshold:
                continue

            identity, identity_sim = _assign_identity(
                crop_features[i], centroids,
            )

            seeds.append({
                "frame_idx": int(frame_idx),
                "xyxy": boxes[i].tolist(),
                "confidence": round(confidence, 4),
                "identity": int(identity),
                "identity_similarity": round(float(identity_sim), 4),
            })

    # ---- Finalise ----
    with session._lock:
        session.seeds = seeds
        session.advance_to(Phase.SEEDING)

    save_session(session)

    # Build summary
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
        len(seeds),
        len(identity_counts),
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
