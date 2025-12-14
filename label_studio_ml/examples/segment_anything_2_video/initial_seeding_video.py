from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

import seeding_common as base

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _build_sam2_video_predictor():
    try:
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise base.InitialSeedingError(
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
        raise base.InitialSeedingError(
            f"SAM2 checkpoint '{model_checkpoint}' not found. Checked: {cand_app} and {cand_sam2}."
        )

    logger.info(
        "Initializing SAM2VideoPredictor (DEVICE=%s, CONFIG=%s, CHECKPOINT=%s)",
        device,
        model_config,
        sam2_checkpoint,
    )
    return build_sam2_video_predictor(model_config, sam2_checkpoint, device=device)


def _clear_jpeg_dir(dir_path: str) -> None:
    if not os.path.isdir(dir_path):
        return
    for name in os.listdir(dir_path):
        if name.lower().endswith((".jpg", ".jpeg")):
            try:
                os.remove(os.path.join(dir_path, name))
            except FileNotFoundError:
                continue


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _extract_all_frames_to_temp_total(
    *,
    video_path: str,
    temp_total_dir: str,
    frames_count: int,
    jpeg_quality: int,
) -> None:
    if os.path.isdir(temp_total_dir):
        existing = [f for f in os.listdir(temp_total_dir) if f.lower().endswith(".jpg")]
        if len(existing) == frames_count:
            logger.info("temp_total already has %d frames; skipping extraction", frames_count)
            return
        if existing:
            shutil.rmtree(temp_total_dir)

    os.makedirs(temp_total_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise base.InitialSeedingError(f"Could not open video file: {video_path}")

    try:
        logger.info("Extracting %d frames to temp_total=%s", frames_count, temp_total_dir)
        frame_idx = 0
        while frame_idx < frames_count:
            success, frame_bgr = cap.read()
            if not success or frame_bgr is None:
                break
            out_path = os.path.join(temp_total_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            frame_idx += 1

        if frame_idx != frames_count:
            raise base.InitialSeedingError(
                f"Frame extraction incomplete: wrote {frame_idx}/{frames_count} frames"
            )
    finally:
        cap.release()


def _prepare_segment_dir(
    *,
    temp_total_dir: str,
    temp_dir: str,
    start_idx: int,
    end_idx: int,
) -> int:
    if end_idx < start_idx:
        raise base.InitialSeedingError(f"Invalid segment: start_idx={start_idx} end_idx={end_idx}")

    os.makedirs(temp_dir, exist_ok=True)
    _clear_jpeg_dir(temp_dir)

    num_frames = end_idx - start_idx + 1
    for local_idx, global_idx in enumerate(range(start_idx, end_idx + 1)):
        src = os.path.join(temp_total_dir, f"{global_idx:06d}.jpg")
        dst = os.path.join(temp_dir, f"{local_idx:05d}.jpg")
        if not os.path.exists(src):
            raise base.InitialSeedingError(f"Missing frame in temp_total: {src}")
        _link_or_copy(src, dst)

    return num_frames


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _mask_logits_to_xyxy(mask_logits: torch.Tensor) -> Optional[np.ndarray]:
    mask = (mask_logits > 0.0).detach().to("cpu")
    while mask.ndim > 2:
        mask = mask.squeeze(0)

    mask_np = mask.numpy().astype(np.uint8)
    ys, xs = np.where(mask_np > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _track_segment(
    *,
    predictor,
    temp_dir: str,
    start_idx: int,
    end_idx: int,
    start_dets: List[base.KeyframeDetection],
) -> List[Dict[str, Any]]:
    if end_idx < start_idx:
        return []
    if not start_dets:
        return []

    num_frames = end_idx - start_idx + 1
    offload_video_to_cpu = _get_env_bool("SAM2_OFFLOAD_VIDEO_TO_CPU", True)

    inference_state = predictor.init_state(
        video_path=temp_dir,
        offload_video_to_cpu=offload_video_to_cpu,
    )
    predictor.reset_state(inference_state)

    local_obj_ids: List[int] = []
    last_box_by_obj: Dict[int, np.ndarray] = {}
    label_by_obj: Dict[int, str] = {}

    for local_id, det in enumerate(start_dets):
        local_obj_ids.append(local_id)
        last_box_by_obj[local_id] = det.xyxy.copy()
        label_by_obj[local_id] = det.label
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=local_id,
            box=det.xyxy,
        )

    sequences: Dict[int, List[Dict[str, Any]]] = {oid: [] for oid in local_obj_ids}

    use_cuda = torch.cuda.is_available() and os.getenv("DEVICE", "cuda").startswith("cuda")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()

    with torch.inference_mode(), autocast_ctx:
        for frame_local, obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=num_frames,
        ):
            global_frame = start_idx + int(frame_local)
            for i, obj_id in enumerate(obj_ids):
                obj_id_int = int(obj_id)
                bbox = _mask_logits_to_xyxy(mask_logits[i])
                enabled = True
                if bbox is not None:
                    last_box_by_obj[obj_id_int] = bbox
                    enabled = True
                else:
                    enabled = False

                sequences[obj_id_int].append(
                    {
                        "frame": global_frame,
                        "xyxy": last_box_by_obj[obj_id_int].copy(),
                        "enabled": enabled,
                    }
                )

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
        tracklets.append(
            {
                "local_id": obj_id,
                "track_id": int(global_track_id),
                "label": label_by_obj.get(obj_id, "object"),
                "start_frame": start_idx,
                "end_frame": end_idx,
                "start_box": start_box,
                "end_box": end_box,
                "visible_at_end": visible_at_end,
                "sequence": seq,
            }
        )

    return tracklets


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
        kept.append(
            {
                "frame": last_frame,
                "xyxy": np.array(last_item["xyxy"], dtype=np.float32).copy(),
                "enabled": True,
            }
        )

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

        merged_seq = merged_seq[: last_enabled_idx + 1]

        tracks.append(
            {
                "track_id": int(tid),
                "sequence": merged_seq,
                "label": track_labels.get(int(tid), "object"),
            }
        )

    return tracks


def _build_prediction_zero_based(
    *,
    tracks: List[Dict[str, Any]],
    width: int,
    height: int,
    frames_count: int,
    fps: float,
) -> Dict[str, Any]:
    converted: List[Dict[str, Any]] = []
    use_sparse = _get_env_bool("SPARSE_SEQUENCE", True)
    for tr in tracks:
        seq_0b = tr.get("sequence", [])
        if use_sparse:
            seq_0b = _sparsify_track_sequence_zero_based(sequence=seq_0b, frames_count=frames_count)
        seq_1b = []
        for item in seq_0b:
            seq_1b.append(
                {
                    "frame": int(item["frame"]) + 1,
                    "xyxy": np.array(item["xyxy"], dtype=np.float32),
                    "enabled": bool(item.get("enabled", True)),
                }
            )
        converted.append(
            {
                "track_id": int(tr.get("track_id", 0)),
                "sequence": seq_1b,
                "label": tr.get("label", "object"),
            }
        )

    return base._build_prediction(converted, width, height, frames_count=frames_count, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initial seeding pipeline using SAM2 video predictor per segment and upload to Label Studio",
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
        help="Fraction of frames to keep as keyframes (default 0.1 => 10%%)",
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
        "--dry-run",
        action="store_true",
        help="Save prediction to JSON file instead of uploading to Label Studio.",
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    exit_code = 0
    temp_root: Optional[str] = None
    try:
        ls = base._build_ls_client(args.ls_url, args.ls_api_key)
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

        keyframes = sorted({int(k) for k in keyframes} | {0, frames_count - 1})
        detections_by_frame = base._run_grounding_dino_on_keyframes(video_path, keyframes, args.prompt)
        for k in keyframes:
            detections_by_frame.setdefault(k, [])

        temp_root = os.path.abspath(f"./temp_init_seeding_video_task_{args.task}")
        temp_total_dir = os.path.join(temp_root, "temp_total")
        temp_segments_root = os.path.join(temp_root, "temp_segments", "worker_0")
        os.makedirs(temp_segments_root, exist_ok=True)

        jpeg_quality = int(os.getenv("FRAME_JPEG_QUALITY", "95"))
        _extract_all_frames_to_temp_total(
            video_path=video_path,
            temp_total_dir=temp_total_dir,
            frames_count=frames_count,
            jpeg_quality=jpeg_quality,
        )

        predictor = _build_sam2_video_predictor()

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

            seg_dir = os.path.join(temp_segments_root, f"seg_{start_kf:06d}_{end_kf:06d}")
            _ = _prepare_segment_dir(
                temp_total_dir=temp_total_dir,
                temp_dir=seg_dir,
                start_idx=start_kf,
                end_idx=end_kf,
            )
            try:
                tracklets = _track_segment(
                    predictor=predictor,
                    temp_dir=seg_dir,
                    start_idx=start_kf,
                    end_idx=end_kf,
                    start_dets=start_dets,
                )
            finally:
                shutil.rmtree(seg_dir, ignore_errors=True)

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

        tracks = _finalize_tracks(global_tracks=global_tracks, track_labels=track_labels)
        prediction = _build_prediction_zero_based(
            tracks=tracks,
            width=width,
            height=height,
            frames_count=frames_count,
            fps=fps,
        )

        if args.dry_run:
            output_file = f"prediction_task_{args.task}.json"
            with open(output_file, "w") as f:
                json.dump(prediction, f, indent=2)
            logger.info("DRY RUN: Prediction saved to %s", output_file)
            logger.info(
                "Validate with: python validate_prediction.py --ls-url %s --ls-api-key <key> --task %s --prediction-file %s --upload",
                args.ls_url,
                args.task,
                output_file,
            )
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
    finally:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
