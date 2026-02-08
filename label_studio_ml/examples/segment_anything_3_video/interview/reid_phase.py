"""ReID (Re-Identification) clustering for the Interview UI.

After all crops are classified (accepted/rejected), this module groups
accepted crops into identity clusters using fused DINOv3 + color histogram
features, then generates calibrated pairs for human verification. Pair
resolutions are applied with a burden-of-proof merge/split policy.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import ReID helpers from complete_reid.py in parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from complete_reid import _compute_hist, _rgb_to_hsv, _hist_intersection

from .state import (
    CropData, CropLabel, InterviewSession, Phase, ReIDPair,
)
from .cache_manager import save_session
from .background import JobProgress

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Feature Fusion
# ---------------------------------------------------------------------------

def compute_fused_similarity(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    dinov3_weight: float = 0.7,
    color_weight: float = 0.3,
) -> float:
    """Compute weighted combination of DINOv3 cosine similarity and color histogram intersection.

    Args:
        feat_a: DINOv3 CLS token embedding for crop A, shape (1024,).
        feat_b: DINOv3 CLS token embedding for crop B, shape (1024,).
        hist_a: Normalized HSV color histogram for crop A.
        hist_b: Normalized HSV color histogram for crop B.
        dinov3_weight: Weight for the cosine similarity component (default 0.7).
        color_weight: Weight for the histogram intersection component (default 0.3).

    Returns:
        Fused similarity score in [0, 1].
    """
    # Cosine similarity for DINOv3 features
    norm_a = float(np.linalg.norm(feat_a))
    norm_b = float(np.linalg.norm(feat_b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        cosine_sim = 0.0
    else:
        cosine_sim = float(np.dot(feat_a, feat_b) / (norm_a * norm_b))
    # Map from [-1, 1] to [0, 1]
    cosine_sim = 0.5 * (cosine_sim + 1.0)

    # Histogram intersection for color features
    color_sim = _hist_intersection(hist_a, hist_b)

    # Weighted combination
    fused = dinov3_weight * cosine_sim + color_weight * color_sim
    return max(0.0, min(1.0, fused))


# ---------------------------------------------------------------------------
# 2. Spherical K-Means
# ---------------------------------------------------------------------------

def spherical_kmeans(features: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    """Spherical K-Means clustering on L2-normalized features.

    Uses K-Means++ style initialization: the first centroid is chosen
    uniformly at random, and each subsequent centroid is sampled with
    probability proportional to (1 - cosine_similarity) to the nearest
    existing centroid.

    Iteration proceeds by assigning each point to the nearest centroid
    (highest cosine similarity), recomputing centroids as the mean of
    assigned points, and L2-normalizing them. Convergence is declared
    when assignments do not change between iterations.

    Args:
        features: (N, D) array of feature vectors (will be L2-normalized internally).
        k: Number of clusters.
        max_iter: Maximum number of iterations.

    Returns:
        Cluster assignments as an (N,) integer array with values in [0, k).
    """
    n, d = features.shape
    if n <= k:
        return np.arange(n, dtype=np.intp)

    # L2-normalize all feature vectors
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X = features / norms

    # K-Means++ initialization
    rng = np.random.RandomState(42)
    centroids = np.empty((k, d), dtype=np.float64)
    first_idx = rng.randint(0, n)
    centroids[0] = X[first_idx]

    for c in range(1, k):
        # Compute cosine similarity to nearest existing centroid
        sims = X @ centroids[:c].T  # (N, c)
        max_sim = sims.max(axis=1)  # (N,)
        # Distance proportional to (1 - max_sim), clipped for stability
        dists = np.maximum(1.0 - max_sim, 0.0)
        dist_sum = dists.sum()
        if dist_sum < 1e-12:
            # Fallback: pick randomly if all points are equidistant
            centroids[c] = X[rng.randint(0, n)]
        else:
            probs = dists / dist_sum
            idx = rng.choice(n, p=probs)
            centroids[c] = X[idx]

    # Iterative assignment and update
    assignments = np.full(n, -1, dtype=np.intp)
    for iteration in range(max_iter):
        # Assign each point to the nearest centroid (highest cosine sim)
        sims = X @ centroids.T  # (N, k)
        new_assignments = sims.argmax(axis=1)

        # Check convergence
        if np.array_equal(assignments, new_assignments):
            logger.debug("Spherical K-Means converged at iteration %d", iteration)
            break
        assignments = new_assignments

        # Update centroids
        for c in range(k):
            mask = assignments == c
            if mask.any():
                centroid = X[mask].mean(axis=0)
                norm = float(np.linalg.norm(centroid))
                if norm > 1e-8:
                    centroid /= norm
                centroids[c] = centroid
            # If no points assigned, keep the old centroid (avoids empty cluster)

    return assignments


# ---------------------------------------------------------------------------
# 3. Silhouette-based K estimation
# ---------------------------------------------------------------------------

def estimate_k(features: np.ndarray, k_range: Tuple[int, int] = (2, 10)) -> int:
    """Estimate optimal number of clusters using silhouette score heuristic.

    Runs spherical K-Means for each candidate K in the given range and
    picks the K with the highest average silhouette score. The silhouette
    score for a point measures how similar it is to its own cluster vs.
    the nearest neighboring cluster, using cosine distance.

    Args:
        features: (N, D) feature matrix.
        k_range: (min_k, max_k) inclusive range of K values to try.

    Returns:
        Optimal K value. Returns 1 if the dataset is too small for
        clustering or no valid K is found.
    """
    n = features.shape[0]
    min_k, max_k = k_range

    # Clamp range to data size
    max_k = min(max_k, n - 1)
    if max_k < min_k or n < 3:
        return max(1, min(n, min_k))

    # L2-normalize for cosine distance computation
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X = features / norms

    # Pairwise cosine distance matrix: dist = 1 - cosine_similarity
    sim_matrix = X @ X.T
    dist_matrix = 1.0 - sim_matrix

    best_k = min_k
    best_score = -1.0

    for k in range(min_k, max_k + 1):
        assignments = spherical_kmeans(features, k)
        n_actual_clusters = len(set(assignments))
        if n_actual_clusters < 2:
            continue

        # Compute average silhouette score
        silhouette_sum = 0.0
        for i in range(n):
            ci = assignments[i]

            # a(i): mean distance to points in same cluster
            same_mask = (assignments == ci)
            same_mask[i] = False
            n_same = same_mask.sum()
            if n_same == 0:
                continue
            a_i = float(dist_matrix[i][same_mask].mean())

            # b(i): min mean distance to points in any other cluster
            b_i = float("inf")
            for cj in range(k):
                if cj == ci:
                    continue
                other_mask = (assignments == cj)
                if not other_mask.any():
                    continue
                mean_dist = float(dist_matrix[i][other_mask].mean())
                if mean_dist < b_i:
                    b_i = mean_dist

            if b_i == float("inf"):
                continue

            denom = max(a_i, b_i)
            if denom > 1e-12:
                silhouette_sum += (b_i - a_i) / denom

        avg_silhouette = silhouette_sum / n
        logger.debug("K=%d silhouette=%.4f", k, avg_silhouette)

        if avg_silhouette > best_score:
            best_score = avg_silhouette
            best_k = k

    logger.info("Estimated optimal K=%d (silhouette=%.4f)", best_k, best_score)
    return best_k


# ---------------------------------------------------------------------------
# 4. Color histogram extraction for crops
# ---------------------------------------------------------------------------

def extract_crop_histograms(
    session: InterviewSession,
    accepted_crops: List[CropData],
    progress: JobProgress,
) -> Dict[str, np.ndarray]:
    """Extract HSV color histograms for accepted crops.

    Reads video frames via PyAV (_read_frame_pyav from seeding_common),
    crops to the bounding box stored in each CropData, and computes a
    normalized HSV histogram using _compute_hist from complete_reid.

    Args:
        session: The interview session (provides video_path and dimensions).
        accepted_crops: List of CropData with label == ACCEPTED.
        progress: JobProgress object for reporting extraction status.

    Returns:
        Mapping from crop_id to its normalized histogram array.
    """
    import seeding_common as base

    histograms: Dict[str, np.ndarray] = {}
    if not accepted_crops:
        return histograms

    # Group crops by frame to minimize video seeks
    frame_to_crops: Dict[int, List[CropData]] = {}
    for crop in accepted_crops:
        frame_to_crops.setdefault(crop.frame_idx, []).append(crop)

    total_frames = len(frame_to_crops)
    progress.step = "Extracting color histograms"
    progress.total = total_frames
    progress.current = 0

    bins = (8, 8, 8)
    video_path = session.video_path
    img_w, img_h = session.width, session.height

    for frame_count, (frame_idx, crops) in enumerate(sorted(frame_to_crops.items()), 1):
        pil_frame = base._read_frame_pyav(video_path, frame_idx)
        if pil_frame is None:
            logger.warning("Could not read frame %d for histogram extraction", frame_idx)
            progress.current = frame_count
            continue

        frame_rgb = np.array(pil_frame.convert("RGB"))
        h, w = frame_rgb.shape[:2]

        for crop in crops:
            x1, y1, x2, y2 = crop.xyxy.astype(int)
            # Clamp to frame bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            crop_rgb = frame_rgb[y1:y2, x1:x2]
            if crop_rgb.size == 0:
                logger.debug("Empty crop %s at frame %d", crop.crop_id, frame_idx)
                continue

            hist = _compute_hist(crop_rgb, bins)
            histograms[crop.crop_id] = hist

        progress.current = frame_count

    logger.info(
        "Extracted histograms for %d / %d accepted crops",
        len(histograms), len(accepted_crops),
    )
    return histograms


# ---------------------------------------------------------------------------
# 5. Pair Sampling (calibrated)
# ---------------------------------------------------------------------------

def sample_pairs(
    session: InterviewSession,
    clusters: Dict[int, List[str]],
    similarity_matrix: np.ndarray,
    crop_ids: List[str],
    max_pairs: int = 30,
) -> List[ReIDPair]:
    """Sample calibrated pairs from three pools for human verification.

    Pools:
        - Ambiguous (~60%): pairs from different clusters with borderline
          similarity (0.3 -- 0.7). These are the most informative for
          resolving cluster boundaries.
        - Confident same (~20%): high-similarity pairs within the same
          cluster (> 0.8). Serve as positive calibration anchors.
        - Confident different (~20%): low-similarity pairs from different
          clusters (< 0.3). Serve as negative calibration anchors.

    Pairs are interleaved (shuffled) so the human reviewer does not see
    blocks of one type, which could bias their responses.

    Args:
        session: Current interview session.
        clusters: Mapping from cluster_id to list of crop_ids in that cluster.
        similarity_matrix: (N, N) symmetric similarity matrix indexed by crop_ids.
        crop_ids: Ordered list of crop IDs corresponding to matrix rows/columns.
        max_pairs: Maximum total number of pairs to generate.

    Returns:
        List of ReIDPair objects, interleaved across pools.
    """
    id_to_idx = {cid: i for i, cid in enumerate(crop_ids)}
    id_to_cluster = {}
    for cid_int, members in clusters.items():
        for cid in members:
            id_to_cluster[cid] = cid_int

    ambiguous: List[ReIDPair] = []
    confident_same: List[ReIDPair] = []
    confident_diff: List[ReIDPair] = []

    cluster_ids_sorted = sorted(clusters.keys())

    # Collect confident_same candidates: pairs within the same cluster
    for cid_int in cluster_ids_sorted:
        members = clusters[cid_int]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a not in id_to_idx or b not in id_to_idx:
                    continue
                sim = float(similarity_matrix[id_to_idx[a], id_to_idx[b]])
                if sim > 0.8:
                    confident_same.append(ReIDPair(
                        pair_id=str(uuid.uuid4())[:12],
                        crop_id_a=a,
                        crop_id_b=b,
                        cluster_a=cid_int,
                        cluster_b=cid_int,
                        pool="confident_same",
                        similarity=sim,
                    ))

    # Collect cross-cluster candidates (ambiguous and confident_different)
    for i_idx, ci in enumerate(cluster_ids_sorted):
        for j_idx in range(i_idx + 1, len(cluster_ids_sorted)):
            cj = cluster_ids_sorted[j_idx]
            for a in clusters[ci]:
                for b in clusters[cj]:
                    if a not in id_to_idx or b not in id_to_idx:
                        continue
                    sim = float(similarity_matrix[id_to_idx[a], id_to_idx[b]])
                    pair_base = dict(
                        crop_id_a=a,
                        crop_id_b=b,
                        cluster_a=ci,
                        cluster_b=cj,
                        similarity=sim,
                    )
                    if 0.3 <= sim <= 0.7:
                        ambiguous.append(ReIDPair(
                            pair_id=str(uuid.uuid4())[:12],
                            pool="ambiguous",
                            **pair_base,
                        ))
                    elif sim < 0.3:
                        confident_diff.append(ReIDPair(
                            pair_id=str(uuid.uuid4())[:12],
                            pool="confident_different",
                            **pair_base,
                        ))

    # Budget allocation: ~60% ambiguous, ~20% confident_same, ~20% confident_different
    n_ambiguous = max(1, int(round(max_pairs * 0.6)))
    n_same = max(1, int(round(max_pairs * 0.2)))
    n_diff = max(1, int(round(max_pairs * 0.2)))

    # Shuffle each pool and take up to budget
    rng = random.Random(42)
    rng.shuffle(ambiguous)
    rng.shuffle(confident_same)
    rng.shuffle(confident_diff)

    selected_ambiguous = ambiguous[:n_ambiguous]
    selected_same = confident_same[:n_same]
    selected_diff = confident_diff[:n_diff]

    # Interleave: combine and shuffle so presentation is not blocked by pool
    all_pairs = selected_ambiguous + selected_same + selected_diff
    rng.shuffle(all_pairs)

    logger.info(
        "Sampled %d pairs: %d ambiguous, %d confident_same, %d confident_different",
        len(all_pairs), len(selected_ambiguous), len(selected_same), len(selected_diff),
    )
    return all_pairs


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

def run_reid_pipeline(
    session: InterviewSession,
    n_clusters: Optional[int],
    progress: JobProgress,
) -> Dict[str, Any]:
    """Full ReID pipeline for the interview workflow.

    Steps:
        1. Collect accepted crops that have DINOv3 features.
        2. Extract HSV color histograms for those crops.
        3. Compute a fused (DINOv3 + color) similarity matrix.
        4. Cluster using spherical K-Means (auto-estimating K if not provided).
        5. Generate calibrated pairs for human verification.
        6. Update session state with clusters and pairs.

    Args:
        session: The current interview session (must be in REID phase).
        n_clusters: User-specified number of identities, or None for auto.
        progress: JobProgress object for status reporting.

    Returns:
        Summary dict with keys: n_clusters, n_pairs, cluster_sizes.
    """
    progress.step = "Collecting accepted crops"
    progress.total = 6

    # Step 1: Collect accepted crops with DINOv3 features
    accepted = session.get_crops_by_label(CropLabel.ACCEPTED)
    featured = [c for c in accepted if c.features is not None]
    if len(featured) < 2:
        raise ValueError(
            f"Need at least 2 accepted crops with features for ReID, got {len(featured)}"
        )
    progress.current = 1
    logger.info("ReID pipeline: %d accepted crops with features", len(featured))

    # Step 2: Extract color histograms
    progress.step = "Extracting color histograms"
    histograms = extract_crop_histograms(session, featured, progress)
    progress.current = 2

    # Build ordered arrays for matrix computation
    crop_ids = [c.crop_id for c in featured]
    n = len(crop_ids)

    feature_matrix = np.stack([c.features for c in featured])  # (N, 1024)

    # Default histogram for crops where extraction failed
    default_hist = np.zeros_like(next(iter(histograms.values()))) if histograms else np.zeros(512)

    hist_list = [histograms.get(cid, default_hist) for cid in crop_ids]
    hist_matrix = np.stack(hist_list)  # (N, H)

    # Step 3: Compute fused similarity matrix
    progress.step = "Computing similarity matrix"
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = compute_fused_similarity(
                feature_matrix[i], feature_matrix[j],
                hist_matrix[i], hist_matrix[j],
            )
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    progress.current = 3

    # Step 4: Cluster with spherical K-Means
    progress.step = "Clustering identities"
    if n_clusters is None or n_clusters < 2:
        k = estimate_k(feature_matrix, k_range=(2, min(10, n - 1)))
    else:
        k = min(n_clusters, n - 1)
    k = max(2, k)

    assignments = spherical_kmeans(feature_matrix, k)
    progress.current = 4

    # Build cluster mapping: cluster_id -> [crop_ids]
    clusters: Dict[int, List[str]] = {}
    for idx, cid in enumerate(crop_ids):
        cluster_id = int(assignments[idx])
        clusters.setdefault(cluster_id, []).append(cid)
        # Also update the crop's reid_cluster_id
        crop = session.get_crop(cid)
        if crop is not None:
            crop.reid_cluster_id = cluster_id

    # Step 5: Generate calibrated pairs
    progress.step = "Generating verification pairs"
    pairs = sample_pairs(session, clusters, sim_matrix, crop_ids, max_pairs=30)
    progress.current = 5

    # Step 6: Update session state
    progress.step = "Saving session"
    session.reid_clusters = clusters
    session.reid_pairs = {p.pair_id: p for p in pairs}
    session.n_identities = len(clusters)
    session.touch()
    save_session(session)
    progress.current = 6

    cluster_sizes = {cid: len(members) for cid, members in clusters.items()}
    summary = {
        "n_clusters": len(clusters),
        "n_pairs": len(pairs),
        "cluster_sizes": cluster_sizes,
        "n_accepted": len(featured),
    }
    logger.info(
        "ReID pipeline complete: %d clusters, %d pairs, sizes=%s",
        summary["n_clusters"], summary["n_pairs"], cluster_sizes,
    )
    return summary


# ---------------------------------------------------------------------------
# 7. Apply resolutions (merge/split logic)
# ---------------------------------------------------------------------------

def apply_resolutions(
    session: InterviewSession,
    resolutions: Dict[str, str],
) -> Dict[str, Any]:
    """Apply human pair resolutions with burden-of-proof merge/split policy.

    Merge rules:
        - Need 2+ confirming "same" (Yes) pairs between the same two clusters
          to trigger a merge, OR 1 "same" pair if its similarity exceeds 0.85.
        - A single "different" (No) pair vetoes the merge entirely for that
          cluster pair, regardless of how many "same" pairs exist.
        - "unsure" pairs are treated as abstentions; at the end, any cluster
          pair that only has "unsure" evidence is left separate.

    The method tracks per-cluster-pair evidence as a dict keyed by
    (cluster_a, cluster_b) tuples, then executes merges that meet the
    threshold. Merged clusters are renumbered starting from 0.

    Args:
        session: Current interview session with reid_pairs populated.
        resolutions: Mapping from pair_id to "same", "different", or "unsure".

    Returns:
        Summary dict with keys: merges_executed, final_clusters, vetoed_pairs.
    """
    # Apply resolution labels to stored pairs
    for pair_id, resolution in resolutions.items():
        pair = session.reid_pairs.get(pair_id)
        if pair is not None:
            pair.resolution = resolution

    # Accumulate evidence per cluster pair
    # Use sorted tuple keys so (a, b) and (b, a) are treated identically
    evidence: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for pair in session.reid_pairs.values():
        if pair.resolution is None:
            continue
        ca, cb = pair.cluster_a, pair.cluster_b
        if ca == cb:
            # Same-cluster pair; no merge decision needed
            continue
        key = (min(ca, cb), max(ca, cb))
        if key not in evidence:
            evidence[key] = {"yes_count": 0, "no_count": 0, "unsure_count": 0, "max_sim": 0.0}

        if pair.resolution == "same":
            evidence[key]["yes_count"] += 1
            evidence[key]["max_sim"] = max(evidence[key]["max_sim"], pair.similarity)
        elif pair.resolution == "different":
            evidence[key]["no_count"] += 1
        else:  # "unsure"
            evidence[key]["unsure_count"] += 1

    # Decide which cluster pairs to merge
    merges_to_execute: List[Tuple[int, int]] = []
    vetoed: List[Tuple[int, int]] = []

    for (ca, cb), ev in evidence.items():
        if ev["no_count"] > 0:
            # Single "No" vetoes the merge
            vetoed.append((ca, cb))
            continue
        if ev["yes_count"] >= 2:
            merges_to_execute.append((ca, cb))
        elif ev["yes_count"] == 1 and ev["max_sim"] > 0.85:
            # High-confidence single pair is sufficient
            merges_to_execute.append((ca, cb))
        # Otherwise: insufficient evidence, leave separate

    # Execute merges using union-find for transitive closure
    # (if A merges with B and B merges with C, then A, B, C are one cluster)
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Initialize all existing cluster IDs
    for cid in session.reid_clusters:
        parent[cid] = cid

    for ca, cb in merges_to_execute:
        union(ca, cb)

    # Rebuild clusters from union-find roots
    new_clusters_raw: Dict[int, List[str]] = {}
    for old_cid, members in session.reid_clusters.items():
        root = find(old_cid)
        if root not in new_clusters_raw:
            new_clusters_raw[root] = []
        new_clusters_raw[root].extend(members)

    # Renumber clusters starting from 0
    new_clusters: Dict[int, List[str]] = {}
    root_to_new_id: Dict[int, int] = {}
    for new_id, (root, members) in enumerate(sorted(new_clusters_raw.items())):
        root_to_new_id[root] = new_id
        new_clusters[new_id] = members

    # Update crop reid_cluster_id to reflect new numbering
    for new_id, members in new_clusters.items():
        for cid in members:
            crop = session.get_crop(cid)
            if crop is not None:
                crop.reid_cluster_id = new_id

    # Update session
    session.reid_clusters = new_clusters
    session.n_identities = len(new_clusters)
    session.touch()
    save_session(session)

    summary = {
        "merges_executed": len(merges_to_execute),
        "vetoed_pairs": len(vetoed),
        "final_clusters": len(new_clusters),
        "cluster_sizes": {cid: len(m) for cid, m in new_clusters.items()},
    }
    logger.info(
        "Applied resolutions: %d merges, %d vetoed, %d final clusters",
        summary["merges_executed"], summary["vetoed_pairs"], summary["final_clusters"],
    )
    return summary
