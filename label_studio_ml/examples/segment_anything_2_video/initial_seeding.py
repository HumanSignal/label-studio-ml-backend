from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from joblib import Memory
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.client import LabelStudio
from PIL import Image
from scipy.ndimage import median_filter
from torchvision.ops import box_iou

# Grounding DINO
from groundingdino.util.inference import load_model, predict


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class InitialSeedingError(Exception):
    """Custom error type for initial seeding pipeline."""


@dataclass
class KeyframeDetection:
    frame_idx: int  # 0-based
    xyxy: np.ndarray  # shape (4,)
    score: float
    label: str
    track_id: Optional[int] = None


def _ensure_meta_text_placeholder(result: Dict[str, Any]) -> None:
    """Ensure result['meta']['text'] exists and has at least 'id:'."""
    meta = result.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        result["meta"] = meta
    raw_text = meta.get("text")
    texts: List[str] = []
    if isinstance(raw_text, str):
        texts = [raw_text]
    elif isinstance(raw_text, list):
        texts = [t for t in raw_text if isinstance(t, str)]
    if not texts or all(not t.strip() for t in texts):
        meta["text"] = "id:"


def _build_ls_client(ls_url: str, ls_api_key: str):
    if not ls_api_key or ls_api_key.strip() == "" or ls_api_key == "your_api_key":
        raise InitialSeedingError(
            "LABEL_STUDIO_API_KEY is required. "
            "Provide it via --ls-api-key or the LABEL_STUDIO_API_KEY env var."
        )

    os.environ.setdefault("LABEL_STUDIO_URL", ls_url)
    os.environ.setdefault("LABEL_STUDIO_API_KEY", ls_api_key)

    logger.info("Connecting to Label Studio at %s", ls_url)
    client = LabelStudio(base_url=ls_url, api_key=ls_api_key, timeout=600)
    logger.info("Connected to Label Studio")
    return client


def _fetch_task(ls, project_id: int, task_id: int) -> Dict[str, Any]:
    logger.info("Fetching task %s from project %s", task_id, project_id)
    task_obj = ls.tasks.get(task_id)
    if not task_obj:
        raise InitialSeedingError(f"Task {task_id} not found")

    task_project = getattr(task_obj, "project", None)
    if task_project is not None and task_project != project_id:
        logger.warning(
            "Task %s belongs to project %s (not %s)",
            getattr(task_obj, "id", task_id),
            task_project,
            project_id,
        )

    task = {"id": task_obj.id, "data": getattr(task_obj, "data", {})}
    logger.info("Task fetched: %s", task.get("id"))
    return task


def _fetch_annotation(ls, annotation_id: int) -> Any:
    logger.info("Fetching annotation %s", annotation_id)
    ann = ls.annotations.get(id=annotation_id)
    if not ann:
        raise InitialSeedingError(f"Annotation {annotation_id} not found")

    result = getattr(ann, "result", None)
    if result is None:
        raise InitialSeedingError(f"Annotation {annotation_id} has no regions")

    logger.info(
        "Annotation fetched: id=%s with %d regions", getattr(ann, "id", annotation_id), len(result or [])
    )
    return ann


def _detect_video_key(task_data: Dict[str, Any]) -> Tuple[str, str]:
    preferred_keys = ["video", "video_url", "video_path"]
    for key in preferred_keys:
        if key in task_data and isinstance(task_data[key], str):
            return key, task_data[key]

    for key, value in task_data.items():
        if not isinstance(value, str):
            continue
        lower = value.lower()
        if lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            return key, value

    raise InitialSeedingError(
        "Could not detect video field in task data. "
        "Ensure your task has a field like 'video' with a video URL/path."
    )


def _get_video_path(task: Dict[str, Any]) -> Tuple[str, str]:
    data = task.get("data") or {}
    key, video_url = _detect_video_key(data)
    logger.info("Using video field '%s' with URL %s", key, video_url)

    if not video_url.startswith("http") and video_url.startswith("/"):
        host = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL")
        if host:
            from urllib.parse import urljoin

            video_url = urljoin(host.rstrip("/"), video_url)
            logger.info("Resolved relative video URL to %s", video_url)

    logger.info("Downloading/caching video via get_local_path…")
    local_path = get_local_path(video_url, task_id=task["id"])
    if not os.path.exists(local_path):
        raise InitialSeedingError(f"Video file not found after download: {local_path}")

    size_mb = os.path.getsize(local_path) / 1024**2
    logger.info("Video cached at: %s (%.2f MB)", local_path, size_mb)
    return local_path, key


def _build_sam2_predictor():
    try:
        from sam2.build_sam import build_sam2  # type: ignore[import]
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise InitialSeedingError(
            "SAM2 backend requested but 'sam2' package is not available in this environment."
        ) from exc

    device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model_config = os.getenv("MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
    model_checkpoint = os.getenv("MODEL_CHECKPOINT", "sam2.1_hiera_large.pt")

    root_dir = os.getcwd()
    cand_app = os.path.join(root_dir, "checkpoints", model_checkpoint)
    cand_sam2 = os.path.join("/sam2", "checkpoints", model_checkpoint)
    if os.path.exists(cand_app):
        sam2_checkpoint = cand_app
    elif os.path.exists(cand_sam2):
        sam2_checkpoint = cand_sam2
    else:
        raise InitialSeedingError(
            f"SAM2 checkpoint '{model_checkpoint}' not found. Checked: {cand_app} and {cand_sam2}."
        )

    logger.info(
        "Initializing SAM2ImagePredictor (DEVICE=%s, CONFIG=%s, CHECKPOINT=%s)",
        device,
        model_config,
        sam2_checkpoint,
    )

    sam2_model = build_sam2(model_config, sam2_checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


def _global_pool_embed(embed: torch.Tensor) -> torch.Tensor:
    if embed.ndim == 4:
        return embed.mean(dim=[2, 3])
    if embed.ndim == 3:
        return embed.mean(dim=[1, 2])
    return embed


def _compute_sam2_frame_embeddings(
    video_id: str,
    video_path: str,
    batch_size: int,
    cache_dir: str,
) -> np.ndarray:
    memory = Memory(cache_dir, verbose=0)

    @memory.cache
    def _inner(video_id_cached: str, video_path_cached: str, batch_size_cached: int) -> np.ndarray:
        predictor = _build_sam2_predictor()
        cap = cv2.VideoCapture(video_path_cached)
        if not cap.isOpened():
            raise InitialSeedingError(f"Could not open video file: {video_path_cached}")

        embeds: List[np.ndarray] = []
        frames: List[np.ndarray] = []
        try:
            while True:
                success, frame_bgr = cap.read()
                if not success or frame_bgr is None:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

                if len(frames) >= batch_size:
                    embeds.append(_embed_batch(predictor, frames))
                    frames = []

            if frames:
                embeds.append(_embed_batch(predictor, frames))
        finally:
            cap.release()

        if not embeds:
            raise InitialSeedingError("No frames read from video for embedding computation")

        stacked = np.concatenate(embeds, axis=0).astype("float16")
        logger.info("Computed SAM2 embeddings for %d frames (shape=%s)", stacked.shape[0], stacked.shape)
        return stacked

    return _inner(video_id, video_path, batch_size)


def _embed_batch(predictor, frames: List[np.ndarray]) -> np.ndarray:
    out: List[np.ndarray] = []
    for frame in frames:
        predictor.set_image(frame)
        features = getattr(predictor, "features", {}) or {}
        embed = (
            features.get("image_embed")
            or features.get("image_embeddings")
            or features.get("image_embedding")
            or None
        )
        if embed is None:
            raise InitialSeedingError("SAM2 predictor did not return image embeddings")
        pooled = _global_pool_embed(embed)
        out.append(pooled.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_change_scores(embeds: np.ndarray) -> np.ndarray:
    if embeds.ndim != 2:
        raise InitialSeedingError(f"Expected embeddings with shape [T, D], got {embeds.shape}")
    norm = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8
    norm_embeds = embeds / norm
    T_len = norm_embeds.shape[0]
    diff = np.zeros(T_len, dtype=np.float32)
    diff[1:] = np.linalg.norm(norm_embeds[1:] - norm_embeds[:-1], axis=1)
    return diff


def smooth_change_scores(diff: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return median_filter(diff, size=kernel_size)


def uniform_indices(T_len: int, K: int) -> List[int]:
    return sorted({int(round(i * T_len / K)) for i in range(max(K, 1))})


def top_change_indices(smooth_diff: np.ndarray, max_candidates: int, min_spacing: int) -> List[int]:
    T_len = len(smooth_diff)
    idx_sorted = np.argsort(-smooth_diff)
    chosen: List[int] = []
    for idx in idx_sorted:
        if len(chosen) >= max_candidates:
            break
        if all(abs(idx - c) >= min_spacing for c in chosen):
            chosen.append(int(idx))
    return sorted(chosen)


def select_keyframes(T_len: int, frac: float, smooth_diff: np.ndarray, min_spacing: int = 30) -> List[int]:
    K = max(1, int(frac * T_len))
    base = set(uniform_indices(T_len, K))
    changed = set(top_change_indices(smooth_diff, max_candidates=3 * K, min_spacing=min_spacing))
    merged = sorted(base.union(changed))
    if len(merged) > K:
        step = max(1, len(merged) // K)
        merged = merged[::step][:K]
    return sorted(merged)


class GroundingDINOHelper:
    def __init__(self):
        repo_root = os.getenv("GROUNDINGDINO_REPO_PATH", "/GroundingDINO")
        config_name = os.getenv("GROUNDING_DINO_CONFIG", "GroundingDINO_SwinT_OGC.py")
        weights_name = os.getenv("GROUNDING_DINO_WEIGHTS", "gdino_swint_darpa-ir-v1-1k_20_1.pth")

        self.prompt = self._resolve_prompt()
        self.device = os.getenv("GROUNDING_DINO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.startswith("cuda") and torch.cuda.is_available()

        config_path = os.path.join(repo_root, "groundingdino", "config", config_name)
        weights_path = os.path.join(repo_root, "weights", weights_name)

        if not os.path.exists(config_path):
            raise InitialSeedingError(f"Grounding DINO config not found at {config_path}")
        if not os.path.exists(weights_path):
            raise InitialSeedingError(f"Grounding DINO weights not found at {weights_path}")

        logger.info(
            "Loading Grounding DINO from config '%s' and weights '%s' on device '%s'",
            config_path,
            weights_path,
            self.device,
        )
        self.model = load_model(model_config_path=config_path, model_checkpoint_path=weights_path, device=self.device)
        self.transform = T.Compose(
            [
                T.RandomResize([int(os.getenv("GROUNDING_DINO_BASE_SIZE", "800"))], max_size=int(os.getenv("GROUNDING_DINO_MAX_SIZE", "1333"))),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _resolve_prompt() -> str:
        prompt = os.getenv("GROUNDING_DINO_PROMPT")
        if prompt:
            prompt = prompt.strip()
        if not prompt:
            labels = os.getenv("GROUNDING_DINO_LABELS", "person")
            label_list = [token.strip() for token in labels.split(",") if token.strip()]
            prompt = ". ".join(label_list) + "."
        if not prompt.endswith("."):
            prompt = prompt + "."
        return prompt

    def infer_frame(
        self,
        frame: np.ndarray,
        *,
        prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[KeyframeDetection]:
        prompt_final = (prompt or self.prompt).strip()
        if not prompt_final.endswith("."):
            prompt_final = prompt_final + "."

        tensor, _ = self.transform(Image.fromarray(frame).convert("RGB"), None)
        tensor = tensor.to(self.device)

        box_threshold = float(box_threshold) if box_threshold is not None else float(
            os.getenv("GROUNDING_DINO_BOX_THRESHOLD", 0.35)
        )
        text_threshold = float(text_threshold) if text_threshold is not None else float(
            os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", 0.25)
        )

        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    boxes, scores, phrases = predict(
                        model=self.model,
                        image=tensor,
                        caption=prompt_final,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        device=self.device,
                    )
            else:
                boxes, scores, phrases = predict(
                    model=self.model,
                    image=tensor,
                    caption=prompt_final,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=self.device,
                )

        if boxes.numel() == 0:
            return []

        h, w = frame.shape[:2]
        xyxy = boxes_to_xyxy(boxes, w, h)
        scores_np = scores.cpu().numpy().astype(np.float32)
        detections: List[KeyframeDetection] = []
        for i in range(xyxy.shape[0]):
            detections.append(
                KeyframeDetection(
                    frame_idx=-1,  # to be filled by caller
                    xyxy=xyxy[i],
                    score=float(scores_np[i]),
                    label="object",
                )
            )
        return detections


def boxes_to_xyxy(boxes_cxcywh: torch.Tensor, width: int, height: int) -> np.ndarray:
    xyxy = torch.zeros_like(boxes_cxcywh)
    xyxy[:, 0] = boxes_cxcywh[:, 0] - 0.5 * boxes_cxcywh[:, 2]
    xyxy[:, 1] = boxes_cxcywh[:, 1] - 0.5 * boxes_cxcywh[:, 3]
    xyxy[:, 2] = boxes_cxcywh[:, 0] + 0.5 * boxes_cxcywh[:, 2]
    xyxy[:, 3] = boxes_cxcywh[:, 1] + 0.5 * boxes_cxcywh[:, 3]
    xyxy = xyxy * torch.tensor([width, height, width, height], device=boxes_cxcywh.device)
    return xyxy.cpu().numpy().astype(np.float32)


def xyxy_to_percent(xyxy: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = xyxy
    x0 = max(0.0, min(float(width - 1), float(x0)))
    y0 = max(0.0, min(float(height - 1), float(y0)))
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    return (x0 / width) * 100.0, (y0 / height) * 100.0, (w / width) * 100.0, (h / height) * 100.0


def _track_between_keyframes(
    predictor_factory,
    cap: cv2.VideoCapture,
    start_idx: int,
    end_idx: int,
    start_dets: List[KeyframeDetection],
    end_dets: List[KeyframeDetection],
    iou_threshold: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[KeyframeDetection]]:
    """
    Returns:
        track_segments: list of dicts with keys track_id, sequence (list of frame boxes)
        unmatched_end: end detections not matched (used to spawn new tracks later)
    """
    if end_idx <= start_idx:
        return [], end_dets

    predictor = predictor_factory()
    # Load segment frames into memory (inclusive)
    frames: List[np.ndarray] = []
    for fidx in range(start_idx, end_idx + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        success, frame_bgr = cap.read()
        if not success or frame_bgr is None:
            logger.warning("Failed to read frame %d during tracking segment", fidx)
            frames.append(None)
            continue
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    track_segments: List[Dict[str, Any]] = []
    end_boxes = np.stack([d.xyxy for d in end_dets], axis=0) if end_dets else np.zeros((0, 4), dtype=np.float32)

    for det in start_dets:
        sequence: List[Dict[str, Any]] = []
        prev_box = det.xyxy.copy()
        last_seen = start_idx

        for local_idx, frame in enumerate(frames):
            frame_id = start_idx + local_idx
            if frame is None:
                continue
            predictor.set_image(frame)
            box = prev_box
            try:
                masks, scores, _ = predictor.predict(
                    box=box.astype(np.float32),
                    multimask_output=True,
                    return_logits=False,
                )
                if masks.size > 0 and scores.size > 0:
                    best_idx = int(np.argmax(scores))
                    mask = masks[best_idx]
                    ys, xs = np.where(mask > 0)
                    if xs.size > 0 and ys.size > 0:
                        x0_tight = int(xs.min())
                        x1_tight = int(xs.max()) + 1
                        y0_tight = int(ys.min())
                        y1_tight = int(ys.max()) + 1
                        prev_box = np.array([x0_tight, y0_tight, x1_tight, y1_tight], dtype=np.float32)
                        last_seen = frame_id
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("SAM2 predict failed on frame %d: %s", frame_id, exc)

            sequence.append({"frame": frame_id + 1, "xyxy": prev_box.copy()})

        match_idx = -1
        if end_boxes.shape[0] > 0:
            ious = box_iou(
                torch.from_numpy(prev_box[None, :]),
                torch.from_numpy(end_boxes),
            ).numpy()[0]
            if ious.size > 0:
                match_idx = int(np.argmax(ious))
                if ious[match_idx] < iou_threshold:
                    match_idx = -1

        track_segments.append(
            {
                "track_id": det.track_id,
                "sequence": sequence,
                "end_match": match_idx,
                "last_seen": last_seen,
                "final_box": prev_box.copy(),
            }
        )

    unmatched_end = []
    matched_end_indices = {seg["end_match"] for seg in track_segments if seg["end_match"] >= 0}
    for idx, det in enumerate(end_dets):
        if idx not in matched_end_indices:
            unmatched_end.append(det)

    return track_segments, unmatched_end


def _merge_tracks_across_pairs(
    keyframes: List[int],
    detections_by_frame: Dict[int, List[KeyframeDetection]],
    video_path: str,
) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise InitialSeedingError(f"Could not open video file: {video_path}")

    # Initialize tracks from first keyframe
    global_tracks: Dict[int, List[Dict[str, Any]]] = {}
    active: Dict[int, Dict[str, Any]] = {}
    next_global_id = 0

    if keyframes:
        first_kf = keyframes[0]
        for det in detections_by_frame.get(first_kf, []):
            det.track_id = next_global_id
            seq_entry = {"frame": first_kf + 1, "xyxy": det.xyxy.copy()}
            global_tracks[next_global_id] = [seq_entry]
            active[next_global_id] = {"last_box": det.xyxy.copy(), "last_frame": first_kf}
            next_global_id += 1

    total_pairs = max(0, len(keyframes) - 1)
    for pair_idx, (start_idx, end_idx) in enumerate(zip(keyframes[:-1], keyframes[1:]), 1):
        # Attach track ids to start detections (use active mapping)
        start_dets = detections_by_frame.get(start_idx, [])
        for det in start_dets:
            if det.track_id is None:
                # Find nearest active by IoU with last_box
                best_id = None
                best_iou = 0.0
                for tid, state in active.items():
                    if state["last_frame"] != start_idx:
                        continue
                    iou = float(
                        box_iou(
                            torch.from_numpy(det.xyxy[None, :]),
                            torch.from_numpy(state["last_box"][None, :]),
                        ).numpy()[0, 0]
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_id = tid
                if best_id is None:
                    best_id = next_global_id
                    next_global_id += 1
                    global_tracks[best_id] = []
                det.track_id = best_id
        end_dets = detections_by_frame.get(end_idx, [])

        track_segments, unmatched_end = _track_between_keyframes(
            predictor_factory=_build_sam2_predictor,
            cap=cap,
            start_idx=start_idx,
            end_idx=end_idx,
            start_dets=start_dets,
            end_dets=end_dets,
        )

        # Update global tracks with segment sequences and propagate IDs
        end_assigned = set()
        for seg in track_segments:
            tid = seg["track_id"]
            if tid is None:
                tid = next_global_id
                next_global_id += 1
                global_tracks.setdefault(tid, [])
            seq_entries = []
            for item in seg["sequence"]:
                seq_entries.append({"frame": item["frame"], "xyxy": item["xyxy"]})
            global_tracks.setdefault(tid, []).extend(seq_entries)
            active[tid] = {"last_box": seg["final_box"], "last_frame": seg["last_seen"]}

            # Assign matching end detection to this track id
            if seg["end_match"] >= 0 and seg["end_match"] < len(end_dets):
                matched = end_dets[seg["end_match"]]
                matched.track_id = tid
                end_assigned.add(seg["end_match"])

        # Spawn new tracks for unmatched detections on end keyframe
        for idx, det in enumerate(end_dets):
            if idx in end_assigned:
                continue
            tid = det.track_id if det.track_id is not None else next_global_id
            if det.track_id is None:
                next_global_id += 1
            det.track_id = tid
            global_tracks.setdefault(tid, []).append({"frame": end_idx + 1, "xyxy": det.xyxy.copy()})
            active[tid] = {"last_box": det.xyxy.copy(), "last_frame": end_idx}

        pct = 100.0 * float(pair_idx) / float(total_pairs if total_pairs else 1)
        logger.info(
            "Keyframe tracking progress: %d/%d pairs (%.1f%%)",
            pair_idx,
            total_pairs,
            pct,
        )

    cap.release()
    tracks: List[Dict[str, Any]] = []
    for gid, seq in global_tracks.items():
        # Deduplicate frames (keep last occurrence)
        seen_frames = {}
        for entry in seq:
            seen_frames[entry["frame"]] = entry["xyxy"]
        merged_seq = [{"frame": f, "xyxy": b} for f, b in sorted(seen_frames.items())]
        tracks.append({"track_id": gid, "sequence": merged_seq})
    return tracks


def _build_prediction(tracks: List[Dict[str, Any]], width: int, height: int, frames_count: int) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for tr in tracks:
        seq_items = []
        for item in tr["sequence"]:
            x_pct, y_pct, w_pct, h_pct = xyxy_to_percent(item["xyxy"], width, height)
            seq_items.append(
                {
                    "frame": int(item["frame"]),
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                    "enabled": True,
                }
            )

        if not seq_items:
            continue

        results.append(
            {
                "id": f"auto-track-{tr['track_id']}",
                "type": "videorectangle",
                "from_name": "box",
                "to_name": "video",
                "score": 1.0,
                "value": {
                    "sequence": seq_items,
                    "framesCount": frames_count,
                },
                "meta": {"text": "id:"},
            }
        )
        _ensure_meta_text_placeholder(results[-1])

    prediction = {"result": results, "score": 1.0, "model_version": "initial-seeding"}
    return prediction


def _upload_prediction(ls, task_id: int, prediction: Dict[str, Any]):
    try:
        result = ls.predictions.create(
            task=task_id,
            score=prediction.get("score", 0.0),
            model_version=prediction.get("model_version", "initial-seeding"),
            result=prediction.get("result", []),
        )
        pred_id = getattr(result, "id", None)
        if pred_id is not None:
            logger.info("Upload complete, prediction id=%s", pred_id)
        else:
            logger.info("Upload request completed (no prediction id in response)")
    except Exception as exc:  # pragma: no cover - defensive
        msg = str(exc)
        if "504" in msg:
            logger.warning("Received 504 from LS during prediction upload; assuming it succeeded.")
        else:
            logger.error("Failed to upload prediction: %s", msg)


def _read_frame(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    success, frame_bgr = cap.read()
    if not success or frame_bgr is None:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _detect_keyframes(
    video_path: str,
    cache_dir: str,
    cache_key: str,
    embedding_batch: int,
    keyframe_frac: float,
    min_spacing: int,
) -> Tuple[List[int], int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise InitialSeedingError(f"Could not open video file: {video_path}")
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    embeds = _compute_sam2_frame_embeddings(cache_key, video_path, embedding_batch, cache_dir)
    if embeds.shape[0] != frames_count:
        logger.warning(
            "Embedding frame count (%d) does not match video frames (%d); proceeding with min length",
            embeds.shape[0],
            frames_count,
        )
        frames_count = min(frames_count, embeds.shape[0])
        embeds = embeds[:frames_count]

    diff = compute_change_scores(embeds)
    smooth = smooth_change_scores(diff, kernel_size=5)
    keyframes = select_keyframes(frames_count, keyframe_frac, smooth, min_spacing=min_spacing)
    logger.info("Selected %d keyframes out of %d total frames", len(keyframes), frames_count)
    return keyframes, width, height, frames_count


def _run_grounding_dino_on_keyframes(
    video_path: str,
    keyframes: List[int],
    prompt: Optional[str],
) -> Dict[int, List[KeyframeDetection]]:
    dino = GroundingDINOHelper()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise InitialSeedingError(f"Could not open video file: {video_path}")

    detections: Dict[int, List[KeyframeDetection]] = {}
    try:
        for idx, frame_idx in enumerate(keyframes, 1):
            frame = _read_frame(cap, frame_idx)
            if frame is None:
                logger.warning("Failed to read keyframe %d", frame_idx)
                continue
            dets = dino.infer_frame(frame, prompt=prompt)
            for d in dets:
                d.frame_idx = frame_idx
            detections[frame_idx] = dets

            pct = 100.0 * float(idx) / float(len(keyframes))
            logger.info(
                "Grounding DINO progress: %d/%d keyframes (%.1f%%)",
                idx,
                len(keyframes),
                pct,
            )
    finally:
        cap.release()

    return detections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initial seeding pipeline using SAM2 + Grounding DINO and upload to Label Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ls-url", required=True, help="Label Studio URL (e.g., https://app.heartex.com)")
    parser.add_argument("--ls-api-key", required=True, help="Label Studio API key")
    parser.add_argument("--project", type=int, required=True, help="Project ID (for validation/logging)")
    parser.add_argument("--task", type=int, required=True, help="Task ID associated with the annotation")
    parser.add_argument("--annotation", type=int, required=True, help="Annotation ID to use as source")
    parser.add_argument(
        "--embedding-batch",
        type=int,
        default=int(os.getenv("EMBED_BATCH", "8")),
        help="Batch size for SAM2 embedding computation",
    )
    parser.add_argument(
        "--keyframe-frac",
        type=float,
        default=0.1,
        help="Fraction of frames to keep as keyframes (default 0.1 => 10%)",
    )
    parser.add_argument(
        "--min-spacing",
        type=int,
        default=30,
        help="Minimum spacing between high-change keyframes to avoid shake oversampling",
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
        help="Class-of-interest prompt for Grounding DINO (overrides env/default).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("=" * 80)
    logger.info("🚀 INITIAL SEEDING STARTED")
    logger.info("=" * 80)
    logger.info("Parameters: project=%s task=%s annotation=%s", args.project, args.task, args.annotation)

    exit_code = 0
    try:
        ls = _build_ls_client(args.ls_url, args.ls_api_key)
        task = _fetch_task(ls, args.project, args.task)
        _ = _fetch_annotation(ls, args.annotation)

        video_path, _video_key = _get_video_path(task)
        keyframes, width, height, frames_count = _detect_keyframes(
            video_path=video_path,
            cache_dir=args.cache_dir,
            cache_key=f"{task['id']}",
            embedding_batch=args.embedding_batch,
            keyframe_frac=args.keyframe_frac,
            min_spacing=args.min_spacing,
        )
        detections = _run_grounding_dino_on_keyframes(video_path, keyframes, args.prompt)

        tracks = _merge_tracks_across_pairs(
            keyframes=keyframes,
            detections_by_frame=detections,
            video_path=video_path,
        )

        prediction = _build_prediction(tracks, width, height, frames_count=frames_count)
        _upload_prediction(ls, args.task, prediction)

        logger.info("=" * 80)
        logger.info("✅ INITIAL SEEDING COMPLETED")
        logger.info("=" * 80)
    except InitialSeedingError as e:
        logger.error("❌ Initial seeding error: %s", e)
        exit_code = 1
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        exit_code = 130
    except Exception as e:  # pragma: no cover - unexpected errors
        logger.error("❌ Unexpected error: %s", e, exc_info=True)
        exit_code = 1
    finally:
        if exit_code != 0:
            logger.info("=" * 80)
            logger.info("❌ INITIAL SEEDING FAILED (exit code: %s)", exit_code)
            logger.info("=" * 80)

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
