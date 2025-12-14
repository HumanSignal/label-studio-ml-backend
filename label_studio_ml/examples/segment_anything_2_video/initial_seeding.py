from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import cv2
import numpy as np
import torch
import groundingdino.datasets.transforms as T
from joblib import Memory
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.client import LabelStudio
from PIL import Image
from torchvision.ops import box_iou, nms

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
    """Compute SAM2 embeddings with joblib caching keyed only by video_id (ignores code changes)."""
    memory = Memory(cache_dir, verbose=0)

    # Use ignore parameter to cache only by video_id, not by function source code
    @memory.cache(ignore=["video_path_arg", "batch_size_arg", "predictor_builder"])
    def _cached_compute(video_id_key: str, video_path_arg: str, batch_size_arg: int, predictor_builder) -> np.ndarray:
        from tqdm import tqdm

        predictor = predictor_builder()
        cap = cv2.VideoCapture(video_path_arg)
        if not cap.isOpened():
            raise InitialSeedingError(f"Could not open video file: {video_path_arg}")

        # Get total frame count for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / fps if fps > 0 else 0
        logger.info(
            "Processing video: %d frames, %.1f FPS, %.1f sec duration",
            total_frames, fps, duration_sec
        )

        # Suppress SAM2 per-frame logging during batch processing
        # SAM2 uses logging.info() directly, so we suppress root logger AND its handlers
        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_handler_levels = [(h, h.level) for h in root_logger.handlers]
        root_logger.setLevel(logging.WARNING)
        for h in root_logger.handlers:
            h.setLevel(logging.WARNING)

        try:
            embeds: List[np.ndarray] = []
            frames: List[np.ndarray] = []

            with tqdm(total=total_frames, desc="Embedding frames", unit="frame") as pbar:
                while True:
                    success, frame_bgr = cap.read()
                    if not success or frame_bgr is None:
                        break
                    frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

                    if len(frames) >= batch_size_arg:
                        embeds.append(_embed_batch(predictor, frames))
                        pbar.update(len(frames))
                        frames = []

                if frames:
                    embeds.append(_embed_batch(predictor, frames))
                    pbar.update(len(frames))

            cap.release()

            if not embeds:
                raise InitialSeedingError("No frames read from video for embedding computation")

            stacked = np.concatenate(embeds, axis=0).astype("float16")
            logger.info("Computed SAM2 embeddings for %d frames (shape=%s)", stacked.shape[0], stacked.shape)
            return stacked
        finally:
            # Restore root logger and handler levels
            root_logger.setLevel(original_level)
            for h, lvl in original_handler_levels:
                h.setLevel(lvl)

    return _cached_compute(video_id, video_path, batch_size, _build_sam2_predictor)


def _extract_sam2_image_embedding(predictor) -> torch.Tensor:
    """Extract SAM2 image embedding using the predictor's recorded features."""
    features = getattr(predictor, "_features", None)
    if not isinstance(features, dict):
        raise InitialSeedingError(
            f"SAM2 predictor features should be a dict, got {type(features).__name__}"
        )
    if "image_embed" not in features:
        raise InitialSeedingError(
            f"SAM2 predictor did not return 'image_embed'; available keys: {list(features.keys())}"
        )
    embed = features["image_embed"]
    if isinstance(embed, (list, tuple)) and embed:
        embed = embed[0]
    if not isinstance(embed, torch.Tensor):
        raise InitialSeedingError(f"'image_embed' expected torch.Tensor, got {type(embed).__name__}")
    return embed


def _embed_batch(predictor, frames: List[np.ndarray]) -> np.ndarray:
    out: List[np.ndarray] = []
    for frame in frames:
        predictor.set_image(frame)
        embed = _extract_sam2_image_embedding(predictor)
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


def _median_filter_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:

    if kernel_size < 1:
        raise InitialSeedingError(f"kernel_size must be positive, got {kernel_size}")
    if kernel_size % 2 == 0:
        kernel_size += 1  # enforce odd window for symmetric padding

    pad = kernel_size // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    try:
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
        return np.median(windows, axis=-1).astype(values.dtype, copy=False)
    except AttributeError:
        # Fallback for older NumPy versions
        filtered = np.empty_like(values)
        for i in range(len(values)):
            start = i
            end = i + kernel_size
            filtered[i] = np.median(padded[start:end])
        return filtered


def smooth_change_scores(diff: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return _median_filter_1d(diff, kernel_size=kernel_size)


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

        nms_iou = float(os.getenv("GROUNDING_DINO_NMS_IOU", "0.5"))
        if xyxy.shape[0] > 0 and nms_iou > 0:
            keep = nms(
                torch.from_numpy(xyxy),
                torch.from_numpy(scores_np),
                nms_iou,
            )
            keep_idx = keep.cpu().numpy().astype(int).tolist()
            xyxy = xyxy[keep_idx]
            scores_np = scores_np[keep_idx]
            if isinstance(phrases, (list, tuple)):
                phrases = [phrases[i] for i in keep_idx]

        detections: List[KeyframeDetection] = []
        for i in range(xyxy.shape[0]):
            label = phrases[i] if i < len(phrases) and phrases[i] else "object"
            detections.append(
                KeyframeDetection(
                    frame_idx=-1,  # to be filled by caller
                    xyxy=xyxy[i],
                    score=float(scores_np[i]),
                    label=label,
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


def _process_keyframe_pair_worker(args: Tuple) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Worker function for parallel keyframe pair processing.
    Each worker creates its own SAM2 predictor and processes one keyframe pair.
    
    Args:
        args: Tuple of (pair_idx, video_path, start_idx, end_idx, start_dets_data, end_dets_data)
              where dets_data are serializable dicts (not KeyframeDetection objects)
    
    Returns:
        Tuple of (pair_idx, segments, unmatched_dets_data)
    """
    pair_idx, video_path, start_idx, end_idx, start_dets_data, end_dets_data = args
    
    # Suppress SAM2 logging in worker process (spawned processes don't inherit log settings)
    # SAM2 uses logging.info() directly to root logger, so we must:
    # 1. Set root logger level to WARNING
    # 2. Add a NullHandler if no handlers exist (spawned processes start with none)
    # 3. Set all existing handler levels to WARNING
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    if not root_logger.handlers:
        root_logger.addHandler(logging.NullHandler())
    for h in root_logger.handlers:
        h.setLevel(logging.WARNING)
    
    # Reconstruct KeyframeDetection objects from serializable data
    start_dets = [
        KeyframeDetection(
            xyxy=np.array(d["xyxy"], dtype=np.float32),
            score=d["score"],
            label=d["label"],
            frame_idx=d["frame_idx"],
            track_id=d["track_id"],
        )
        for d in start_dets_data
    ]
    end_dets = [
        KeyframeDetection(
            xyxy=np.array(d["xyxy"], dtype=np.float32),
            score=d["score"],
            label=d["label"],
            frame_idx=d["frame_idx"],
            track_id=d["track_id"],
        )
        for d in end_dets_data
    ]
    
    # Create fresh predictor for this process
    predictor = _build_sam2_predictor()
    
    segments, unmatched = _track_between_keyframes(
        predictor=predictor,
        video_path=video_path,
        start_idx=start_idx,
        end_idx=end_idx,
        start_dets=start_dets,
        end_dets=end_dets,
    )
    
    # Convert unmatched back to serializable format
    unmatched_data = [
        {
            "xyxy": d.xyxy.tolist(),
            "score": d.score,
            "label": d.label,
            "frame_idx": d.frame_idx,
            "track_id": d.track_id,
        }
        for d in unmatched
    ]
    
    return pair_idx, segments, unmatched_data


def _track_between_keyframes(
    predictor,
    video_path: str,
    start_idx: int,
    end_idx: int,
    start_dets: List[KeyframeDetection],
    end_dets: List[KeyframeDetection],
    iou_threshold: float = 0.3,
) -> Tuple[List[Dict[str, Any]], List[KeyframeDetection]]:
    """
    Track detections between two keyframes using SAM2 mask prediction.
    Optimized: batches all box predictions per frame into single GPU call.
    Uses torch.inference_mode() and autocast for faster inference.
    
    Returns:
        track_segments: list of dicts with keys track_id, sequence (list of frame boxes)
        unmatched_end: end detections not matched (used to spawn new tracks later)
    """
    if end_idx <= start_idx or not start_dets:
        return [], end_dets

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], end_dets

    try:
        # Track state: current box for each detection
        track_boxes = [det.xyxy.copy() for det in start_dets]
        track_last_seen = [start_idx] * len(start_dets)
        track_enabled = [True] * len(start_dets)
        sequences = [[] for _ in start_dets]
        n_dets = len(start_dets)

        # Use inference mode and autocast for faster GPU inference
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for frame_id in range(start_idx, end_idx + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                success, frame_bgr = cap.read()
                if not success or frame_bgr is None:
                    for i in range(n_dets):
                        sequences[i].append({
                            "frame": frame_id + 1,
                            "xyxy": track_boxes[i].copy(),
                            "enabled": track_enabled[i],
                        })
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                predictor.set_image(frame_rgb)

                # Batch predict all boxes at once for better GPU utilization
                if n_dets > 0:
                    input_boxes = np.stack([tb.astype(np.float32) for tb in track_boxes], axis=0)
                    try:
                        masks, scores, _ = predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_boxes,
                            multimask_output=False,
                        )
                        if masks.ndim == 4:
                            masks = masks.squeeze(1)
                        
                        for i in range(n_dets):
                            frame_enabled = True
                            if i < masks.shape[0]:
                                mask = masks[i]
                                ys, xs = np.where(mask > 0)
                                if xs.size > 0 and ys.size > 0:
                                    track_boxes[i] = np.array([
                                        int(xs.min()), int(ys.min()),
                                        int(xs.max()) + 1, int(ys.max()) + 1
                                    ], dtype=np.float32)
                                    track_last_seen[i] = frame_id
                                    track_enabled[i] = True
                                else:
                                    frame_enabled = False
                                    track_enabled[i] = False
                            else:
                                frame_enabled = track_enabled[i]
                            
                            sequences[i].append({
                                "frame": frame_id + 1,
                                "xyxy": track_boxes[i].copy(),
                                "enabled": frame_enabled,
                            })
                    except Exception:
                        for i in range(n_dets):
                            sequences[i].append({
                                "frame": frame_id + 1,
                                "xyxy": track_boxes[i].copy(),
                                "enabled": track_enabled[i],
                            })

        # Match final boxes to end detections using IoU with center-distance fallback
        end_boxes = np.stack([d.xyxy for d in end_dets], axis=0) if end_dets else np.zeros((0, 4), dtype=np.float32)
        track_segments = []
        matched_end_indices = set()  # Track which end detections are already matched
        for i, det in enumerate(start_dets):
            match_idx = -1
            if end_boxes.shape[0] > 0 and track_enabled[i]:
                ious = box_iou(
                    torch.from_numpy(track_boxes[i][None, :]),
                    torch.from_numpy(end_boxes),
                ).numpy()[0]
                if ious.size > 0:
                    # Find best unmatched end detection
                    for best in np.argsort(-ious):
                        if best not in matched_end_indices and ious[best] >= iou_threshold:
                            match_idx = int(best)
                            matched_end_indices.add(match_idx)
                            break
                    
                    # Fallback: center-distance matching if IoU failed
                    if match_idx < 0:
                        track_center = np.array([(track_boxes[i][0] + track_boxes[i][2]) / 2,
                                                  (track_boxes[i][1] + track_boxes[i][3]) / 2])
                        end_centers = np.stack([(end_boxes[:, 0] + end_boxes[:, 2]) / 2,
                                                 (end_boxes[:, 1] + end_boxes[:, 3]) / 2], axis=1)
                        dists = np.linalg.norm(end_centers - track_center, axis=1)
                        # Use video dimensions for distance threshold (10% of diagonal)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        _, sample_frame = cap.read()
                        if sample_frame is not None:
                            h, w = sample_frame.shape[:2]
                        else:
                            h, w = 1080, 1920  # fallback
                        dist_threshold = 0.1 * np.sqrt(h**2 + w**2)
                        for best in np.argsort(dists):
                            if best not in matched_end_indices and dists[best] < dist_threshold:
                                match_idx = int(best)
                                matched_end_indices.add(match_idx)
                                break

            track_segments.append({
                "track_id": det.track_id,
                "sequence": sequences[i],
                "end_match": match_idx,
                "last_seen": track_last_seen[i],
                "final_box": track_boxes[i].copy(),
            })

        unmatched_end = []
        matched_indices = {seg["end_match"] for seg in track_segments if seg["end_match"] >= 0}
        for idx, det in enumerate(end_dets):
            if idx not in matched_indices:
                unmatched_end.append(det)

        return track_segments, unmatched_end
    finally:
        cap.release()


def _merge_tracks_across_pairs(
    keyframes: List[int],
    detections_by_frame: Dict[int, List[KeyframeDetection]],
    video_path: str,
    num_workers: int = 8,
) -> List[Dict[str, Any]]:
    """
    Merge tracks across keyframe pairs with parallel SAM2 tracking.
    
    Args:
        keyframes: List of keyframe indices
        detections_by_frame: Detections for each keyframe
        video_path: Path to video file
        num_workers: Number of parallel workers for tracking
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    if len(keyframes) < 2:
        # No pairs to track
        tracks = []
        if keyframes:
            first_kf = keyframes[0]
            for i, det in enumerate(detections_by_frame.get(first_kf, [])):
                tracks.append({
                    "track_id": i,
                    "sequence": [{"frame": first_kf + 1, "xyxy": det.xyxy.copy()}]
                })
        return tracks

    # Suppress SAM2 logging during tracking
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handler_levels = [(h, h.level) for h in root_logger.handlers]
    root_logger.setLevel(logging.WARNING)
    for h in root_logger.handlers:
        h.setLevel(logging.WARNING)

    try:
        # Initialize track_labels dict early so Phase 2 can use it
        track_labels: Dict[int, str] = {}
        
        # Phase 1: Assign initial track IDs to first keyframe detections
        next_global_id = 0
        first_kf = keyframes[0]
        for det in detections_by_frame.get(first_kf, []):
            det.track_id = next_global_id
            track_labels[det.track_id] = det.label  # Store label immediately
            next_global_id += 1

        # Phase 2: Assign track IDs to start detections only
        # End detection IDs will be assigned during merge based on matching
        # Also store labels for all start detections
        for start_idx in keyframes[:-1]:
            start_dets = detections_by_frame.get(start_idx, [])
            for det in start_dets:
                if det.track_id is None:
                    det.track_id = next_global_id
                    next_global_id += 1
                # Store label for this track if not already stored
                if det.track_id not in track_labels:
                    track_labels[det.track_id] = det.label

        # Phase 3: Build work items for parallel processing
        # Convert KeyframeDetection objects to serializable dicts for multiprocessing
        keyframe_pairs = list(zip(keyframes[:-1], keyframes[1:]))
        work_items = []
        for pair_idx, (start_idx, end_idx) in enumerate(keyframe_pairs):
            start_dets = detections_by_frame.get(start_idx, [])
            end_dets = detections_by_frame.get(end_idx, [])
            # Serialize detections for cross-process transfer
            start_dets_data = [
                {
                    "xyxy": d.xyxy.tolist(),
                    "score": d.score,
                    "label": d.label,
                    "frame_idx": d.frame_idx,
                    "track_id": d.track_id,
                }
                for d in start_dets
            ]
            end_dets_data = [
                {
                    "xyxy": d.xyxy.tolist(),
                    "score": d.score,
                    "label": d.label,
                    "frame_idx": d.frame_idx,
                    "track_id": d.track_id,
                }
                for d in end_dets
            ]
            work_items.append((pair_idx, video_path, start_idx, end_idx, start_dets_data, end_dets_data))

        # Phase 4: Process pairs in parallel using multiprocessing
        # Each process creates its own SAM2 predictor and CUDA context
        # Use NVIDIA MPS for concurrent GPU kernel execution (start MPS daemon externally)
        all_results = [None] * len(work_items)
        # Use provided num_workers, capped by number of work items
        n_workers = min(len(work_items), max(1, num_workers))
        
        logger.info("Starting SAM2 tracking for %d keyframe pairs with %d workers...", len(work_items), n_workers)
        
        # Use spawn method for CUDA compatibility
        ctx = mp.get_context("spawn")
        
        if n_workers > 1 and len(work_items) > 1:
            # Parallel processing with progress tracking
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                futures = {executor.submit(_process_keyframe_pair_worker, item): item[0] for item in work_items}
                
                from tqdm import tqdm as tqdm_iter
                for future in tqdm_iter(futures, desc="SAM2 tracking pairs", unit="pair", total=len(futures)):
                    try:
                        pair_idx, segments, unmatched_data = future.result()
                        # Reconstruct KeyframeDetection objects from serialized data
                        unmatched = [
                            KeyframeDetection(
                                xyxy=np.array(d["xyxy"], dtype=np.float32),
                                score=d["score"],
                                label=d["label"],
                                frame_idx=d["frame_idx"],
                                track_id=d["track_id"],
                            )
                            for d in unmatched_data
                        ]
                        all_results[pair_idx] = (segments, unmatched)
                    except Exception as e:
                        logger.error("Worker failed for pair %d: %s", futures[future], e)
                        # Fallback: empty result
                        all_results[futures[future]] = ([], [])
        else:
            # Sequential fallback
            for item in tqdm(work_items, desc="SAM2 tracking pairs", unit="pair"):
                pair_idx, video_path_item, start_idx, end_idx, start_dets_data, end_dets_data = item
                # Reconstruct detections
                start_dets = [
                    KeyframeDetection(
                        xyxy=np.array(d["xyxy"], dtype=np.float32),
                        score=d["score"],
                        label=d["label"],
                        frame_idx=d["frame_idx"],
                        track_id=d["track_id"],
                    )
                    for d in start_dets_data
                ]
                end_dets = [
                    KeyframeDetection(
                        xyxy=np.array(d["xyxy"], dtype=np.float32),
                        score=d["score"],
                        label=d["label"],
                        frame_idx=d["frame_idx"],
                        track_id=d["track_id"],
                    )
                    for d in end_dets_data
                ]
                predictor = _build_sam2_predictor()
                segments, unmatched = _track_between_keyframes(
                    predictor=predictor,
                    video_path=video_path_item,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_dets=start_dets,
                    end_dets=end_dets,
                )
                all_results[pair_idx] = (segments, unmatched)

        # Phase 5: Merge all results into global tracks
        global_tracks: Dict[int, List[Dict[str, Any]]] = {}
        
        # Add first keyframe entries (labels already stored in Phase 1 and 2)
        for det in detections_by_frame.get(first_kf, []):
            global_tracks[det.track_id] = [{"frame": first_kf + 1, "xyxy": det.xyxy.copy()}]

        # Process results in order
        for pair_idx, (start_idx, end_idx) in enumerate(keyframe_pairs):
            segments, unmatched = all_results[pair_idx]
            end_dets = detections_by_frame.get(end_idx, [])
            
            end_assigned = set()
            for seg in segments:
                tid = seg["track_id"]
                if tid is not None:
                    global_tracks.setdefault(tid, []).extend(seg["sequence"])
                    
                    # Update end detection track ID if matched
                    if seg["end_match"] >= 0 and seg["end_match"] < len(end_dets):
                        end_dets[seg["end_match"]].track_id = tid
                        end_assigned.add(seg["end_match"])

            # Add unmatched end detections as new track entries
            # IMPORTANT: Only spawn new tracks in the first 10% of keyframes to limit fragmentation
            # After that, unmatched detections are ignored (assumed to be new objects entering scene)
            spawn_cutoff = max(1, int(len(keyframe_pairs) * 0.1))
            for idx, det in enumerate(end_dets):
                if idx not in end_assigned:
                    # Only spawn new tracks early in the video
                    if pair_idx < spawn_cutoff:
                        if det.track_id is None:
                            det.track_id = next_global_id
                            next_global_id += 1
                        global_tracks.setdefault(det.track_id, []).append({
                            "frame": end_idx + 1, "xyxy": det.xyxy.copy()
                        })
                        # Store label for new tracks
                        if det.track_id not in track_labels:
                            track_labels[det.track_id] = det.label

        # Phase 6: Deduplicate, trim disabled tails, and format output
        tracks: List[Dict[str, Any]] = []
        for gid, seq in global_tracks.items():
            seen_frames = {}
            for entry in seq:
                # Store full entry (xyxy + enabled) keyed by frame
                seen_frames[entry["frame"]] = {
                    "xyxy": entry["xyxy"],
                    "enabled": entry.get("enabled", True),
                }
            merged_seq = [
                {"frame": f, "xyxy": data["xyxy"], "enabled": data["enabled"]}
                for f, data in sorted(seen_frames.items())
            ]
            
            # Trim trailing disabled frames (track ended but wasn't properly terminated)
            # Find last enabled frame and truncate there
            last_enabled_idx = -1
            for i, item in enumerate(merged_seq):
                if item["enabled"]:
                    last_enabled_idx = i
            
            if last_enabled_idx >= 0:
                # Keep up to last enabled frame, mark the last one as end of track
                merged_seq = merged_seq[:last_enabled_idx + 1]
                if merged_seq:
                    merged_seq[-1]["enabled"] = False  # Mark last frame as disabled (track ends here)
            
            # Skip tracks with no enabled frames
            if not merged_seq or last_enabled_idx < 0:
                continue
            
            tracks.append({
                "track_id": gid,
                "sequence": merged_seq,
                "label": track_labels.get(gid, "object"),
            })
        
        return tracks
    finally:
        # Restore logger levels
        root_logger.setLevel(original_level)
        for h, lvl in original_handler_levels:
            h.setLevel(lvl)


def _build_prediction(tracks: List[Dict[str, Any]], width: int, height: int, frames_count: int, fps: float) -> Dict[str, Any]:
    duration = frames_count / fps if fps > 0 else 0.0
    results: List[Dict[str, Any]] = []
    for tr in tracks:
        seq_items = []
        for item in tr["sequence"]:
            x_pct, y_pct, w_pct, h_pct = xyxy_to_percent(item["xyxy"], width, height)
            frame_num = int(item["frame"])
            seq_items.append(
                {
                    "frame": frame_num,
                    "x": x_pct,
                    "y": y_pct,
                    "width": w_pct,
                    "height": h_pct,
                    "enabled": item.get("enabled", True),
                    "rotation": 0,
                    "time": (frame_num - 1) / fps if fps > 0 else 0.0,
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
                "origin": "manual",
                "value": {
                    "sequence": seq_items,
                    "framesCount": frames_count,
                    "duration": duration,
                    "labels": [tr.get("label", "object")],
                },
                "meta": {"text": "id:"},
            }
        )
        _ensure_meta_text_placeholder(results[-1])

    prediction = {"result": results, "score": 1.0, "model_version": "gdinosam2-init-seed"}
    return prediction


def _upload_prediction(ls, task_id: int, prediction: Dict[str, Any]):
    try:
        result = ls.predictions.create(
            task=task_id,
            score=prediction.get("score", 0.0),
            model_version=prediction.get("model_version", "gdinosam2-init-seed"),
            result=prediction.get("result", []),
        )
        pred_id = getattr(result, "id", None)
        if pred_id is not None:
            logger.info("Upload complete, prediction id=%s", pred_id)
        else:
            logger.info("Upload request completed (no prediction id in response)")
    except Exception as exc:  # pragma: no cover - defensive
        msg = str(exc)
        err_no = getattr(exc, "errno", None)
        if "504" in msg:
            logger.warning("Received 504 from LS during prediction upload; assuming it succeeded.")
        else:
            if err_no is not None:
                logger.error("Failed to upload prediction (errno=%s): %s", err_no, msg)
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
) -> Tuple[List[int], int, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise InitialSeedingError(f"Could not open video file: {video_path}")
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
    return keyframes, width, height, frames_count, fps


def _run_grounding_dino_on_keyframes(
    video_path: str,
    keyframes: List[int],
    prompt: Optional[str],
) -> Dict[int, List[KeyframeDetection]]:
    from tqdm import tqdm

    dino = GroundingDINOHelper()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise InitialSeedingError(f"Could not open video file: {video_path}")

    detections: Dict[int, List[KeyframeDetection]] = {}
    try:
        for frame_idx in tqdm(keyframes, desc="Grounding DINO keyframes", unit="kf"):
            frame = _read_frame(cap, frame_idx)
            if frame is None:
                logger.warning("Failed to read keyframe %d", frame_idx)
                continue
            dets = dino.infer_frame(frame, prompt=prompt)
            for d in dets:
                d.frame_idx = frame_idx
            detections[frame_idx] = dets
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.getenv("SAM2_NUM_WORKERS", "4")),
        help="Number of parallel workers for SAM2 tracking (default: 4, or SAM2_NUM_WORKERS env var). "
             "Set to 1 for sequential processing. For best GPU utilization with multiple workers, "
             "enable NVIDIA MPS: 'sudo nvidia-smi -c 3 && nvidia-cuda-mps-control -d'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Save prediction to JSON file instead of uploading to Label Studio. "
             "Useful for validating prediction format before full run.",
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
        keyframes, width, height, frames_count, fps = _detect_keyframes(
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
            num_workers=args.num_workers,
        )

        prediction = _build_prediction(tracks, width, height, frames_count=frames_count, fps=fps)
        
        if args.dry_run:
            # Save prediction to file for validation
            import json
            output_file = f"prediction_task_{args.task}.json"
            with open(output_file, "w") as f:
                json.dump(prediction, f, indent=2)
            logger.info("=" * 80)
            logger.info("🔍 DRY RUN: Prediction saved to %s", output_file)
            logger.info("Validate with: python validate_prediction.py --ls-url %s --ls-api-key <key> --task %s --prediction-file %s --upload",
                        args.ls_url, args.task, output_file)
            logger.info("=" * 80)
        else:
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
