from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import torch
from joblib import Memory
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.client import LabelStudio
from PIL import Image
from torchvision.ops import nms

logger = logging.getLogger(__name__)


class InitialSeedingError(Exception):
    pass


def _ensure_groundingdino_importable() -> None:
    try:
        import groundingdino  # noqa: F401
        return
    except ModuleNotFoundError:
        local_repo = os.path.join(os.path.dirname(__file__), "grounding_dino")
        if os.path.isdir(local_repo) and local_repo not in sys.path:
            sys.path.insert(0, local_repo)

    try:
        import groundingdino  # noqa: F401
    except ModuleNotFoundError as exc:
        raise InitialSeedingError(
            "GroundingDINO is not available. Install it or ensure the local "
            "'grounding_dino' directory is on PYTHONPATH."
        ) from exc


@dataclass
class KeyframeDetection:
    frame_idx: int
    xyxy: np.ndarray
    score: float
    label: str
    track_id: Optional[int] = None


def _ensure_meta_text_placeholder(result: Dict[str, Any]) -> None:
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


def _manual_download_video(url: str, dest_path: str) -> None:
    """Manually download video with Authorization header if needed."""
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Token {api_key}"
    
    logger.info("Starting manual download from %s to %s", url, dest_path)
    try:
        with requests.get(url, headers=headers, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Manual download completed")
    except Exception as e:
        logger.error("Manual download failed: %s", e)
        # Clean up partial file
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise


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
    
    # Check for empty or missing file
    if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
        logger.warning("Cached video file is empty (0 bytes). Removing and attempting manual download...")
        try:
            os.remove(local_path)
        except OSError:
            pass
            
        try:
            _manual_download_video(video_url, local_path)
        except Exception:
            # If manual failed, we already logged it. 
            # We can try get_local_path one last time as last resort or just fail.
            # But likely if manual failed, get_local_path won't work either if it's network/auth.
            pass

    if not os.path.exists(local_path):
        raise InitialSeedingError(f"Video file not found after download: {local_path}")

    size_mb = os.path.getsize(local_path) / 1024**2
    logger.info("Video cached at: %s (%.2f MB)", local_path, size_mb)
    
    if size_mb == 0:
        raise InitialSeedingError(f"Video file is empty (0 bytes) after download attempts: {local_path}")

    return local_path, key


def _build_sam2_predictor():
    try:
        from sam2.build_sam import build_sam2  # type: ignore[import]
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
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


def _extract_sam2_image_embedding(predictor) -> torch.Tensor:
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


def _compute_sam2_frame_embeddings(
    video_id: str,
    video_path: str,
    batch_size: int,
    cache_dir: str,
) -> np.ndarray:
    memory = Memory(cache_dir, verbose=0)

    @memory.cache(ignore=["video_path_arg", "batch_size_arg", "predictor_builder"])
    def _cached_compute(video_id_key: str, video_path_arg: str, batch_size_arg: int, predictor_builder) -> np.ndarray:
        from tqdm import tqdm

        predictor = predictor_builder()
        cap = cv2.VideoCapture(video_path_arg)
        if not cap.isOpened():
            raise InitialSeedingError(f"Could not open video file: {video_path_arg}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / fps if fps > 0 else 0
        logger.info(
            "Processing video: %d frames, %.1f FPS, %.1f sec duration",
            total_frames,
            fps,
            duration_sec,
        )

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
            logger.info(
                "Computed SAM2 embeddings for %d frames (shape=%s)",
                stacked.shape[0],
                stacked.shape,
            )
            return stacked
        finally:
            root_logger.setLevel(original_level)
            for h, lvl in original_handler_levels:
                h.setLevel(lvl)

    return _cached_compute(video_id, video_path, batch_size, _build_sam2_predictor)


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
        kernel_size += 1

    pad = kernel_size // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    try:
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
        return np.median(windows, axis=-1).astype(values.dtype, copy=False)
    except AttributeError:
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
        idx = np.linspace(0, len(merged) - 1, num=K, dtype=int)
        merged = [merged[i] for i in idx]
    return sorted(set(merged))


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


class GroundingDINOHelper:
    def __init__(self):
        _ensure_groundingdino_importable()
        import groundingdino.datasets.transforms as T

        from groundingdino.util.inference import load_model, predict

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
        self._predict = predict
        self.transform = T.Compose(
            [
                T.RandomResize(
                    [int(os.getenv("GROUNDING_DINO_BASE_SIZE", "800"))],
                    max_size=int(os.getenv("GROUNDING_DINO_MAX_SIZE", "1333")),
                ),
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
                    boxes, scores, phrases = self._predict(
                        model=self.model,
                        image=tensor,
                        caption=prompt_final,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        device=self.device,
                    )
            else:
                boxes, scores, phrases = self._predict(
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
                    frame_idx=-1,
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


def xyxy_to_percent(xyxy: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = xyxy
    x0 = max(0.0, min(float(width - 1), float(x0)))
    y0 = max(0.0, min(float(height - 1), float(y0)))
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    return (x0 / width) * 100.0, (y0 / height) * 100.0, (w / width) * 100.0, (h / height) * 100.0


def _build_prediction(
    tracks: List[Dict[str, Any]],
    width: int,
    height: int,
    frames_count: int,
    fps: float,
) -> Dict[str, Any]:
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
                    "labels": [tr.get("label") or "object"],
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
    except Exception as exc:  # pragma: no cover
        msg = str(exc)
        err_no = getattr(exc, "errno", None)
        if "504" in msg:
            logger.warning("Received 504 from LS during prediction upload; assuming it succeeded.")
        else:
            if err_no is not None:
                logger.error("Failed to upload prediction (errno=%s): %s", err_no, msg)
            else:
                logger.error("Failed to upload prediction: %s", msg)
