"""DINOv3 feature extractor and MLP classifier for the Interview UI.

Provides lazy-loaded DINOv3 ViT-L backbone, CLS-token feature extraction,
2-layer MLP binary classifier with feature-level augmentation, training
loop with class weighting / uncertainty sampling, and dense spatial grid
feature search for Strategy B discovery.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .background import JobProgress
from .cache_manager import load_model, save_model, save_session
from .state import CropData, CropLabel, CropSource, InterviewSession, Phase

logger = logging.getLogger(__name__)

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ---------------------------------------------------------------------------
# DINOv3 singleton
# ---------------------------------------------------------------------------

_dinov3_model = None
_dinov3_processor = None


def _get_dinov3():
    """Lazy-load DINOv3 ViT-L backbone (frozen, bfloat16).

    Configurable via ``DINOV3_MODEL`` env var.  Defaults to
    ``facebook/dinov2-large`` (1024-dim CLS tokens).
    """
    global _dinov3_model, _dinov3_processor
    if _dinov3_model is None:
        from transformers import AutoImageProcessor, AutoModel
        model_name = os.getenv("DINOV3_MODEL", "facebook/dinov2-large")
        logger.info("Loading DINOv3 backbone from %s ...", model_name)
        _dinov3_model = (
            AutoModel.from_pretrained(model_name)
            .to(DEVICE, dtype=DTYPE)
            .eval()
        )
        for p in _dinov3_model.parameters():
            p.requires_grad = False
        _dinov3_processor = AutoImageProcessor.from_pretrained(model_name)
        logger.info("DINOv3 loaded on %s (%s)", DEVICE, DTYPE)
    return _dinov3_model, _dinov3_processor


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(crops: List[Image.Image], batch_size: int = 16) -> np.ndarray:
    """Extract DINOv3 CLS-token features from crop images.

    Returns: (N, 1024) float32 array, L2-normalized.
    """
    if not crops:
        return np.empty((0, 1024), dtype=np.float32)

    model, processor = _get_dinov3()
    all_features: List[np.ndarray] = []

    for start in range(0, len(crops), batch_size):
        batch_imgs = crops[start : start + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt")
        inputs = {k: v.to(DEVICE, dtype=DTYPE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        cls_tokens = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
        norms = np.maximum(np.linalg.norm(cls_tokens, axis=1, keepdims=True), 1e-8)
        all_features.append(cls_tokens / norms)

    return np.concatenate(all_features, axis=0)


# ---------------------------------------------------------------------------
# Crop metadata
# ---------------------------------------------------------------------------

def compute_crop_metadata(
    xyxy: np.ndarray, frame_width: int, frame_height: int
) -> np.ndarray:
    """Compute normalized crop metadata: [cx, cy, scale, aspect_ratio].

    - cx, cy: center position normalized to [0, 1]
    - scale: sqrt(area) / sqrt(frame_area)
    - aspect_ratio: width / height
    """
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0 / frame_width
    cy = (y1 + y2) / 2.0 / frame_height
    scale = math.sqrt(bw * bh) / math.sqrt(frame_width * frame_height)
    return np.array([cx, cy, scale, bw / bh], dtype=np.float32)


# ---------------------------------------------------------------------------
# MLP Classifier
# ---------------------------------------------------------------------------

class CropClassifier(nn.Module):
    """2-layer MLP for binary classification of crops.

    Input: 1028-dim (1024 DINOv3 + 4 metadata)
    Architecture: Linear(1028, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1)
    Output: logit (use BCEWithLogitsLoss for training, sigmoid for inference)
    """

    def __init__(self, input_dim: int = 1028, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: (B, 1028) -> (B, 1) logit."""
        return self.fc2(self.drop(self.relu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Feature-level augmentation
# ---------------------------------------------------------------------------

def augment_features(
    features: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply feature-level augmentation: MixUp, CutMix, RandomErasing.

    Augmented samples are appended to the original batch (originals preserved).
    Returns (augmented_features, augmented_labels).
    """
    n, d = features.shape
    if n < 2:
        return features, labels

    aug_feats: List[torch.Tensor] = []
    aug_labels: List[torch.Tensor] = []

    # MixUp: blend random pairs with lambda ~ Beta(0.2, 0.2)
    perm = torch.randperm(n)
    lam = torch.distributions.Beta(0.2, 0.2).sample((n,)).to(features.device).unsqueeze(1)
    aug_feats.append(lam * features + (1.0 - lam) * features[perm])
    aug_labels.append(lam.squeeze(1) * labels + (1.0 - lam.squeeze(1)) * labels[perm])

    # CutMix: swap random contiguous band of 20-40% of dimensions
    perm2 = torch.randperm(n)
    band_len = max(1, int(d * (0.2 + 0.2 * torch.rand(1).item())))
    start = torch.randint(0, max(1, d - band_len), (1,)).item()
    cutmix = features.clone()
    cutmix[:, start : start + band_len] = features[perm2, start : start + band_len]
    frac = band_len / d
    aug_feats.append(cutmix)
    aug_labels.append((1.0 - frac) * labels + frac * labels[perm2])

    # RandomErasing: zero out random 10-20% of feature dimensions
    n_erase = max(1, int(d * (0.1 + 0.1 * torch.rand(1).item())))
    mask = torch.ones_like(features)
    for i in range(n):
        mask[i, torch.randperm(d)[:n_erase]] = 0.0
    aug_feats.append(features * mask)
    aug_labels.append(labels.clone())

    return (
        torch.cat([features] + aug_feats, dim=0),
        torch.cat([labels] + aug_labels, dim=0),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_single_frame(video_path: str, frame_idx: int) -> Optional[Image.Image]:
    """Read a single frame by index via PyAV, return as PIL RGB Image."""
    import av
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        if frame_idx > 0 and stream.time_base:
            container.seek(int(frame_idx / fps / stream.time_base), stream=stream)
        current_idx = 0
        for frame in container.decode(video=0):
            if current_idx >= frame_idx:
                return frame.to_image()
            current_idx += 1
        return None
    finally:
        container.close()


def _ensure_crop_features(
    session: InterviewSession,
    crop_ids: List[str],
    progress: Optional[JobProgress] = None,
) -> None:
    """Extract and cache DINOv3 features for crops that don't have them yet."""
    missing = [cid for cid in crop_ids if session.crops[cid].features is None]
    if not missing:
        return
    if progress:
        progress.step = f"Extracting DINOv3 features for {len(missing)} crops"

    from collections import defaultdict
    frame_to_cids: Dict[int, List[str]] = defaultdict(list)
    for cid in missing:
        frame_to_cids[session.crops[cid].frame_idx].append(cid)

    processed = 0
    for frame_idx in sorted(frame_to_cids.keys()):
        cids = frame_to_cids[frame_idx]
        pil_frame = _read_single_frame(session.video_path, frame_idx)
        if pil_frame is None:
            logger.warning("Could not decode frame %d", frame_idx)
            continue

        crop_imgs: List[Image.Image] = []
        crop_id_order: List[str] = []
        for cid in cids:
            crop = session.crops[cid]
            x1, y1, x2, y2 = crop.xyxy.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(pil_frame.width, x2), min(pil_frame.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_imgs.append(pil_frame.crop((x1, y1, x2, y2)))
            crop_id_order.append(cid)

        if not crop_imgs:
            continue
        feats = extract_features(crop_imgs)
        for i, cid in enumerate(crop_id_order):
            c = session.crops[cid]
            c.features = feats[i]
            c.metadata = compute_crop_metadata(c.xyxy, session.width, session.height)
        processed += len(crop_id_order)
        if progress:
            progress.current = processed
            progress.total = len(missing)


def _build_feature_matrix(session: InterviewSession, crop_ids: List[str]) -> torch.Tensor:
    """Build (N, 1028) feature matrix: DINOv3 (1024) + metadata (4)."""
    rows = []
    for cid in crop_ids:
        crop = session.crops[cid]
        feat = crop.features if crop.features is not None else np.zeros(1024, dtype=np.float32)
        meta = crop.metadata if crop.metadata is not None else np.zeros(4, dtype=np.float32)
        rows.append(np.concatenate([feat, meta]))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def _overlaps_any(box: np.ndarray, existing: List[np.ndarray], threshold: float) -> bool:
    """True if box overlaps any existing box above the IoU threshold."""
    return any(_compute_iou(box, eb) >= threshold for eb in existing)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_classifier(
    session: InterviewSession, progress: JobProgress
) -> Dict[str, Any]:
    """Train the MLP classifier on labeled crops.

    1. Collect accepted (positive) and rejected (negative) crops
    2. Build feature matrix (1028-dim: DINOv3 + metadata)
    3. Apply class weights for imbalance
    4. Train for 20 epochs with AdamW lr=1e-3
    5. Score all unlabeled crops -> uncertainty sampling
    6. uncertainty = 1.0 - abs(2 * sigmoid(logit) - 1.0)
    7. Save model to cache, update session stats

    Target: <2s per training cycle on RTX 6000 Ada.
    """
    progress.step = "Collecting labelled crops"
    accepted = session.get_crops_by_label(CropLabel.ACCEPTED)
    rejected = session.get_crops_by_label(CropLabel.REJECTED)
    n_pos, n_neg = len(accepted), len(rejected)

    if n_pos == 0 or n_neg == 0:
        logger.warning("Need >= 1 pos and >= 1 neg (got %d / %d)", n_pos, n_neg)
        return {"accuracy": 0.0, "n_pos": n_pos, "n_neg": n_neg,
                "epochs": 0, "pending_scored": 0, "mean_uncertainty": 0.5}

    # Ensure features extracted for all crops
    _ensure_crop_features(session, list(session.crops.keys()), progress)

    # Build training data
    progress.step = "Building training data"
    train_ids = [c.crop_id for c in accepted] + [c.crop_id for c in rejected]
    y_np = np.array([1.0] * n_pos + [0.0] * n_neg, dtype=np.float32)

    X_train = _build_feature_matrix(session, train_ids)
    y_train = torch.tensor(y_np, dtype=torch.float32)

    # Inverse-frequency class weights
    w_pos = len(y_np) / (2.0 * max(n_pos, 1))
    w_neg = len(y_np) / (2.0 * max(n_neg, 1))
    sample_weights = torch.where(y_train == 1.0, torch.tensor(w_pos), torch.tensor(w_neg))

    # Init model (or load cached)
    model = CropClassifier()
    cached_sd = load_model(session.cache_key)
    if cached_sd is not None:
        try:
            model.load_state_dict(cached_sd)
        except Exception as e:
            logger.warning("Could not load cached model: %s", e)
    model.train()

    device = torch.device(DEVICE)
    model, X_train, y_train, sample_weights = (
        model.to(device), X_train.to(device), y_train.to(device), sample_weights.to(device)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_epochs = 20
    progress.step = "Training classifier"
    progress.total = n_epochs
    best_acc = 0.0

    for epoch in range(n_epochs):
        X_aug, y_aug = augment_features(X_train, y_train)
        n_aug_extra = X_aug.shape[0] - X_train.shape[0]
        aug_w = torch.cat([sample_weights, torch.ones(n_aug_extra, device=device)])

        optimizer.zero_grad()
        logits = model(X_aug).squeeze(-1)
        loss = (F.binary_cross_entropy_with_logits(logits, y_aug, reduction="none") * aug_w).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(model(X_train).squeeze(-1)) >= 0.5).float()
            best_acc = max(best_acc, (preds == y_train).float().mean().item())
        progress.current = epoch + 1

    # Score pending crops
    progress.step = "Scoring pending crops"
    model.eval()
    pending = session.get_crops_by_label(CropLabel.PENDING)
    pending_scored = 0
    mean_unc = 0.5

    if pending:
        pids = [c.crop_id for c in pending]
        with torch.no_grad():
            probs = torch.sigmoid(model(_build_feature_matrix(session, pids).to(device)).squeeze(-1)).cpu().numpy()
        uncertainties = 1.0 - np.abs(2.0 * probs - 1.0)
        for i, cid in enumerate(pids):
            session.crops[cid].uncertainty = float(uncertainties[i])
        pending_scored = len(pids)
        mean_unc = float(np.mean(uncertainties))

    # Persist
    save_model(session.cache_key, model.cpu().state_dict())
    session.model_trained = True
    session.training_epochs += n_epochs
    session.training_accuracy = best_acc
    session.touch()
    save_session(session)

    result = {"accuracy": best_acc, "n_pos": n_pos, "n_neg": n_neg,
              "epochs": n_epochs, "pending_scored": pending_scored,
              "mean_uncertainty": mean_unc}
    logger.info("Training complete: %s", result)
    return result


# ---------------------------------------------------------------------------
# Feature search (Strategy B helper)
# ---------------------------------------------------------------------------

def run_feature_search(
    session: InterviewSession, progress: JobProgress
) -> List[CropData]:
    """Dense spatial grid feature search.

    1. Define sliding window grid across sampled frames at scales [0.05, 0.10, 0.15]
    2. Extract DINOv3 features for each grid cell
    3. Compare to confirmed positives using cosine similarity
    4. Return top-K most similar cells that don't overlap existing crops
    """
    import uuid

    progress.step = "Preparing feature search"
    accepted = session.get_crops_by_label(CropLabel.ACCEPTED)
    if not accepted:
        logger.warning("Feature search requires at least one accepted crop")
        return []

    _ensure_crop_features(session, [c.crop_id for c in accepted], progress)
    pos_feats = [c.features for c in accepted if c.features is not None]
    if not pos_feats:
        return []

    mean_feat = np.mean(np.stack(pos_feats), axis=0)
    norm = np.linalg.norm(mean_feat)
    if norm > 1e-8:
        mean_feat /= norm

    # Existing bboxes per frame for overlap filtering
    existing_per_frame: Dict[int, List[np.ndarray]] = {}
    for crop in session.crops.values():
        existing_per_frame.setdefault(crop.frame_idx, []).append(crop.xyxy)

    W, H = session.width, session.height
    scales = [0.05, 0.10, 0.15]
    top_k, sim_thresh, iou_thresh = 50, 0.5, 0.3
    candidates: List[Tuple[float, CropData]] = []

    frames = session.sampled_frames
    if not frames:
        total = session.frames_count or 1
        step = max(1, total // 10)
        frames = list(range(0, total, step))[:10]

    progress.total = len(frames)

    for fi, frame_idx in enumerate(frames):
        progress.step = f"Feature search: frame {fi + 1}/{len(frames)}"
        progress.current = fi

        pil_frame = _read_single_frame(session.video_path, frame_idx)
        if pil_frame is None:
            continue

        grid_crops: List[Image.Image] = []
        grid_boxes: List[np.ndarray] = []

        for scale in scales:
            cw, ch = max(16, int(W * scale)), max(16, int(H * scale))
            sx, sy = max(8, cw // 2), max(8, ch // 2)
            for y0 in range(0, H - ch + 1, sy):
                for x0 in range(0, W - cw + 1, sx):
                    box = np.array([x0, y0, x0 + cw, y0 + ch], dtype=np.float32)
                    if _overlaps_any(box, existing_per_frame.get(frame_idx, []), iou_thresh):
                        continue
                    grid_crops.append(pil_frame.crop((x0, y0, x0 + cw, y0 + ch)))
                    grid_boxes.append(box)

        if not grid_crops:
            continue

        feats = extract_features(grid_crops, batch_size=32)
        sims = feats @ mean_feat  # cosine sim (both L2-normed)

        for j in range(len(grid_crops)):
            if sims[j] < sim_thresh:
                continue
            candidates.append((float(sims[j]), CropData(
                crop_id=str(uuid.uuid4())[:12],
                frame_idx=frame_idx,
                xyxy=grid_boxes[j],
                score=float(sims[j]),
                label=CropLabel.PENDING,
                source=CropSource.FEATURE_SEARCH,
                prompt="feature_search",
                uncertainty=0.5,
                features=feats[j],
                metadata=compute_crop_metadata(grid_boxes[j], W, H),
            )))

    candidates.sort(key=lambda x: -x[0])
    results = [cd for _, cd in candidates[:top_k]]
    for cd in results:
        session.add_crop(cd)

    progress.step = f"Feature search complete: {len(results)} new candidates"
    progress.current = progress.total
    logger.info("Feature search found %d candidates from %d frames", len(results), len(frames))
    save_session(session)
    return results
