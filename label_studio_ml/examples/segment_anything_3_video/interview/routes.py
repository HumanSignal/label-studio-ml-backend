"""Flask Blueprint for Interview UI — all REST endpoints.

Serves the SPA at /interview and exposes /interview/api/* for backend ops.
Long-running operations return 202 with a job_id for polling.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request, send_from_directory, abort

from .state import (
    CropData, CropLabel, CropSource, InterviewSession, Phase,
    create_session, get_session, get_or_create_session, list_sessions,
    delete_session, SeedConfig,
)
from .cache_manager import (
    cache_exists, list_project_caches, save_session, load_session,
    delete_cache, save_model, load_model,
)
from .background import submit_job, get_job_progress, get_job_result

logger = logging.getLogger(__name__)

# Blueprint with static files served from interview/static/
interview_bp = Blueprint(
    "interview",
    __name__,
    static_folder="static",
    static_url_path="",
    url_prefix="/interview",
)


@interview_bp.after_request
def _fix_passthrough(response):
    """Convert direct-passthrough file responses to buffered responses.

    The label_studio_ml logging middleware calls response.get_data() on
    every response, which crashes on send_from_directory's streaming
    responses with 'RuntimeError: Attempted implicit sequence conversion
    but the response object is in direct passthrough mode'.
    """
    if response.direct_passthrough:
        response.direct_passthrough = False
    return response


# ===========================================================================
# SPA entry point
# ===========================================================================

@interview_bp.route("/")
@interview_bp.route("/index.html")
def index():
    return send_from_directory(interview_bp.static_folder, "index.html")


# ===========================================================================
# Session endpoints
# ===========================================================================

@interview_bp.route("/api/session/init", methods=["POST"])
def session_init():
    """Create or find a session. Check cache, fetch video info."""
    data = request.get_json(force=True)
    project_id = data.get("project_id")
    task_id = data.get("task_id")
    annotation_id = data.get("annotation_id")

    if not project_id or not task_id:
        return jsonify({"error": "project_id and task_id are required"}), 400

    project_id = int(project_id)
    task_id = int(task_id)
    annotation_id = int(annotation_id) if annotation_id else None

    cache_key = f"p{project_id}_t{task_id}"

    # Check for existing caches
    has_cache = cache_exists(cache_key)
    project_caches = list_project_caches(project_id)
    other_caches = [c for c in project_caches if c.get("cache_key") != cache_key]

    options = []
    if has_cache:
        options.extend(["resume", "build_on", "fresh"])
    elif other_caches:
        for oc in other_caches:
            options.append(f"use_from_{oc['task_id']}")
        options.append("fresh")
    else:
        options.append("fresh")

    return jsonify({
        "cache_key": cache_key,
        "has_cache": has_cache,
        "other_caches": other_caches,
        "options": options,
    })


@interview_bp.route("/api/session/resume", methods=["POST"])
def session_resume():
    """Resume, Build On, or start Fresh from cache."""
    data = request.get_json(force=True)
    project_id = int(data["project_id"])
    task_id = int(data["task_id"])
    annotation_id = data.get("annotation_id")
    annotation_id = int(annotation_id) if annotation_id else None
    mode = data.get("mode", "fresh")  # resume, build_on, fresh, use_from_<task_id>

    cache_key = f"p{project_id}_t{task_id}"

    if mode == "resume":
        session = load_session(cache_key)
        if session is None:
            return jsonify({"error": "No cache found to resume"}), 404
        # Register in memory
        from .state import _sessions, _registry_lock
        with _registry_lock:
            _sessions[session.session_id] = session
        return jsonify({"session_id": session.session_id, **session.stats()})

    elif mode == "build_on":
        session = load_session(cache_key)
        if session is None:
            return jsonify({"error": "No cache found to build on"}), 404
        session.phase = Phase.DETECTION
        from .state import _sessions, _registry_lock
        with _registry_lock:
            _sessions[session.session_id] = session
        return jsonify({"session_id": session.session_id, **session.stats()})

    elif mode.startswith("use_from_"):
        source_task_id = int(mode.split("_")[-1])
        source_key = f"p{project_id}_t{source_task_id}"
        source = load_session(source_key)
        if source is None:
            return jsonify({"error": f"No cache found for task {source_task_id}"}), 404

        # Create new session importing features/model/labels from source
        session = create_session(project_id, task_id, annotation_id)
        session.prompts = list(source.prompts)
        session.model_trained = source.model_trained
        session.training_epochs = source.training_epochs
        session.training_accuracy = source.training_accuracy
        session.phase = Phase.DETECTION

        # Copy features if source has model.pt
        source_model = load_model(source_key)
        if source_model:
            save_model(session.cache_key, source_model)
            session.model_trained = True

        return jsonify({"session_id": session.session_id, **session.stats()})

    else:
        # Fresh start
        delete_cache(cache_key, project_id)
        session = create_session(project_id, task_id, annotation_id)
        return jsonify({"session_id": session.session_id, **session.stats()})


@interview_bp.route("/api/session/<session_id>/status", methods=["GET"])
def session_status(session_id: str):
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(session.stats())


@interview_bp.route("/api/session/<session_id>/video_info", methods=["POST"])
def session_video_info(session_id: str):
    """Fetch video info for the session's task. Must be called after init."""
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def _fetch(progress):
        progress.step = "Fetching video info..."
        progress.total = 3

        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from seeding_common import (
            _build_ls_client, _fetch_task, _get_video_path,
            _get_video_info_pyav,
        )

        progress.step = "Connecting to Label Studio..."
        progress.current = 1
        ls_url = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL", "")
        ls_api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
        ls = _build_ls_client(ls_url, ls_api_key)

        progress.step = "Fetching task data..."
        progress.current = 2
        task = _fetch_task(ls, session.project_id, session.task_id)
        video_path, video_key = _get_video_path(task)

        progress.step = "Reading video metadata..."
        progress.current = 3
        width, height, frames_count, fps = _get_video_info_pyav(video_path)

        with session._lock:
            session.video_path = video_path
            session.video_key = video_key
            session.width = width
            session.height = height
            session.frames_count = frames_count
            session.fps = fps
            session.touch()

        return {
            "video_path": video_path,
            "width": width,
            "height": height,
            "frames_count": frames_count,
            "fps": fps,
        }

    job_id = submit_job(_fetch, name="fetch_video_info")
    return jsonify({"job_id": job_id}), 202


# ===========================================================================
# Detection endpoints (Phase 1)
# ===========================================================================

@interview_bp.route("/api/detect/start", methods=["POST"])
def detect_start():
    """Start detection job: sample frames, detect, NMS, extract features, cluster."""
    data = request.get_json(force=True)
    session_id = data["session_id"]
    prompt = data.get("prompt", "person")

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def _detect(progress):
        from .detection import run_detection_pipeline
        return run_detection_pipeline(session, prompt, progress)

    job_id = submit_job(_detect, name="detection")
    return jsonify({"job_id": job_id}), 202


@interview_bp.route("/api/detect/crops", methods=["GET"])
def detect_crops():
    """List crops with filtering and sorting."""
    session_id = request.args.get("session_id")
    filter_label = request.args.get("filter", "all")
    sort_by = request.args.get("sort", "uncertainty")  # uncertainty, cluster, frame
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 50))

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    crops = list(session.crops.values())

    # Filter
    if filter_label != "all":
        try:
            label = CropLabel(filter_label)
            crops = [c for c in crops if c.label == label]
        except ValueError:
            pass

    # Sort
    if sort_by == "uncertainty":
        crops.sort(key=lambda c: -c.uncertainty)
    elif sort_by == "cluster":
        crops.sort(key=lambda c: (c.cluster_id or 9999, -c.uncertainty))
    elif sort_by == "frame":
        crops.sort(key=lambda c: (c.frame_idx, c.xyxy[0] if c.xyxy is not None else 0))

    total = len(crops)
    crops = crops[offset:offset + limit]

    return jsonify({
        "total": total,
        "offset": offset,
        "limit": limit,
        "crops": [c.to_dict() for c in crops],
    })


@interview_bp.route("/api/detect/frame/<int:frame_idx>", methods=["GET"])
def detect_frame(frame_idx: int):
    """Serve a frame as JPEG."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        abort(404)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from seeding_common import _read_frame_pyav

    pil_img = _read_frame_pyav(session.video_path, frame_idx)
    if pil_img is None:
        abort(404)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.getvalue(), 200, {"Content-Type": "image/jpeg"}


@interview_bp.route("/api/detect/frame/<int:frame_idx>/annotated", methods=["GET"])
def detect_frame_annotated(frame_idx: int):
    """Frame with color-coded boxes drawn."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        abort(404)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from seeding_common import _read_frame_pyav

    pil_img = _read_frame_pyav(session.video_path, frame_idx)
    if pil_img is None:
        abort(404)

    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil_img)

    color_map = {
        CropLabel.ACCEPTED: "#00ff00",
        CropLabel.REJECTED: "#ff0000",
        CropLabel.PENDING: "#ffff00",
    }
    source_color_override = {
        CropSource.HUMAN_DRAWN: "#ff8800",
        CropSource.FEATURE_SEARCH: "#aa00ff",
    }

    for crop in session.get_crops_by_frame(frame_idx):
        color = source_color_override.get(crop.source, color_map.get(crop.label, "#ffff00"))
        x1, y1, x2, y2 = crop.xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.getvalue(), 200, {"Content-Type": "image/jpeg"}


@interview_bp.route("/api/detect/crop/<crop_id>/image", methods=["GET"])
def detect_crop_image(crop_id: str):
    """Serve a cropped box as JPEG."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        abort(404)

    crop = session.get_crop(crop_id)
    if crop is None:
        abort(404)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from seeding_common import _read_frame_pyav

    pil_img = _read_frame_pyav(session.video_path, crop.frame_idx)
    if pil_img is None:
        abort(404)

    x1, y1, x2, y2 = [int(round(v)) for v in crop.xyxy]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(pil_img.width, x2)
    y2 = min(pil_img.height, y2)
    cropped = pil_img.crop((x1, y1, x2, y2))

    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf.getvalue(), 200, {"Content-Type": "image/jpeg"}


@interview_bp.route("/api/detect/label", methods=["POST"])
def detect_label():
    """Batch label crops (accept/reject)."""
    data = request.get_json(force=True)
    session_id = data["session_id"]
    labels = data.get("labels", {})  # {crop_id: "accepted" | "rejected"}

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    updated = 0
    for crop_id, label_str in labels.items():
        try:
            label = CropLabel(label_str)
            if session.label_crop(crop_id, label):
                updated += 1
        except ValueError:
            pass

    save_session(session)
    return jsonify({"updated": updated, **session.stats()})


@interview_bp.route("/api/detect/draw", methods=["POST"])
def detect_draw():
    """Add a human-drawn box (Draw Mode) — auto-accepted."""
    data = request.get_json(force=True)
    session_id = data["session_id"]
    frame_idx = int(data["frame_idx"])
    xyxy = data["xyxy"]  # [x1, y1, x2, y2] in pixel coords

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    import numpy as np
    import uuid

    crop = CropData(
        crop_id=str(uuid.uuid4())[:12],
        frame_idx=frame_idx,
        xyxy=np.array(xyxy, dtype=np.float32),
        score=1.0,
        label=CropLabel.ACCEPTED,
        source=CropSource.HUMAN_DRAWN,
        prompt="human_drawn",
    )
    session.add_crop(crop)
    save_session(session)

    return jsonify({"crop": crop.to_dict(), **session.stats()})


@interview_bp.route("/api/detect/train", methods=["POST"])
def detect_train():
    """Start training job: train MLP, score unlabeled, uncertainty sort."""
    data = request.get_json(force=True)
    session_id = data["session_id"]

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def _train(progress):
        from .dinov3_classifier import train_classifier
        return train_classifier(session, progress)

    job_id = submit_job(_train, name="train_classifier")
    return jsonify({"job_id": job_id}), 202


@interview_bp.route("/api/detect/training_status", methods=["GET"])
def detect_training_status():
    """Training progress, accuracy, next batch."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    return jsonify({
        "model_trained": session.model_trained,
        "training_epochs": session.training_epochs,
        "training_accuracy": session.training_accuracy,
        "pending_crops": len(session.get_crops_by_label(CropLabel.PENDING)),
    })


@interview_bp.route("/api/detect/recall_strategy", methods=["POST"])
def detect_recall_strategy():
    """Start recall gap job: multi_prompt / feature_search."""
    data = request.get_json(force=True)
    session_id = data["session_id"]
    strategy = data.get("strategy")  # "multi_prompt" or "feature_search"
    extra_prompts = data.get("prompts", [])

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def _recall(progress):
        from .detection import run_recall_strategy
        return run_recall_strategy(session, strategy, extra_prompts, progress)

    job_id = submit_job(_recall, name=f"recall_{strategy}")
    return jsonify({"job_id": job_id}), 202


# ===========================================================================
# ReID endpoints (Phase 2)
# ===========================================================================

@interview_bp.route("/api/reid/start", methods=["POST"])
def reid_start():
    """Start clustering job."""
    data = request.get_json(force=True)
    session_id = data["session_id"]
    n_clusters = data.get("n_clusters")  # optional, auto-estimated if None

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def _cluster(progress):
        from .reid_phase import run_reid_pipeline
        return run_reid_pipeline(session, n_clusters, progress)

    job_id = submit_job(_cluster, name="reid_clustering")
    return jsonify({"job_id": job_id}), 202


@interview_bp.route("/api/reid/clusters", methods=["GET"])
def reid_clusters():
    """Cluster list + pairs for resolution."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    clusters_info = {}
    for cid, crop_ids in session.reid_clusters.items():
        clusters_info[str(cid)] = {
            "crop_ids": crop_ids,
            "count": len(crop_ids),
        }

    # Get unresolved pairs
    unresolved = [
        p.__dict__ for p in session.reid_pairs.values()
        if p.resolution is None
    ]

    return jsonify({
        "clusters": clusters_info,
        "n_identities": session.n_identities,
        "unresolved_pairs": unresolved,
        "total_pairs": len(session.reid_pairs),
    })


@interview_bp.route("/api/reid/resolve", methods=["POST"])
def reid_resolve():
    """Submit pair resolutions (same/different/unsure)."""
    data = request.get_json(force=True)
    session_id = data["session_id"]
    resolutions = data.get("resolutions", {})  # {pair_id: "same"|"different"|"unsure"}

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    from .reid_phase import apply_resolutions
    result = apply_resolutions(session, resolutions)
    save_session(session)

    return jsonify(result)


@interview_bp.route("/api/reid/pair/<pair_id>/frames", methods=["GET"])
def reid_pair_frames(pair_id: str):
    """Frame + crop data for a specific pair."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    pair = session.reid_pairs.get(pair_id)
    if pair is None:
        return jsonify({"error": "Pair not found"}), 404

    crop_a = session.get_crop(pair.crop_id_a)
    crop_b = session.get_crop(pair.crop_id_b)

    return jsonify({
        "pair": pair.__dict__,
        "crop_a": crop_a.to_dict() if crop_a else None,
        "crop_b": crop_b.to_dict() if crop_b else None,
    })


# ===========================================================================
# Seeding + Upload endpoints (Phase 3)
# ===========================================================================

@interview_bp.route("/api/seeds/generate", methods=["POST"])
def seeds_generate():
    """Start seed generation job."""
    data = request.get_json(force=True)
    session_id = data["session_id"]

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def _generate(progress):
        from .seeding_phase import generate_seeds
        return generate_seeds(session, progress)

    job_id = submit_job(_generate, name="seed_generation")
    return jsonify({"job_id": job_id}), 202


@interview_bp.route("/api/seeds/list", methods=["GET"])
def seeds_list():
    """Seed list with identity assignments."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    # Summarize seeds by identity
    identity_summary = {}
    for seed in session.seeds:
        identity = seed.get("identity", "unknown")
        if identity not in identity_summary:
            identity_summary[identity] = {"count": 0, "frames": []}
        identity_summary[identity]["count"] += 1
        identity_summary[identity]["frames"].append(seed.get("frame_idx", 0))

    return jsonify({
        "total_seeds": len(session.seeds),
        "identities": identity_summary,
        "seed_config": {
            "frame_interval": session.seed_config.frame_interval,
            "confidence_threshold": session.seed_config.confidence_threshold,
        },
    })


@interview_bp.route("/api/seeds/config", methods=["GET"])
def seeds_config_get():
    """Current seed generation config."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    return jsonify({
        "frame_interval": session.seed_config.frame_interval,
        "confidence_threshold": session.seed_config.confidence_threshold,
    })


@interview_bp.route("/api/seeds/config", methods=["PUT"])
def seeds_config_put():
    """Update seed generation config."""
    data = request.get_json(force=True)
    session_id = data["session_id"]
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    if "frame_interval" in data:
        session.seed_config.frame_interval = int(data["frame_interval"])
    if "confidence_threshold" in data:
        session.seed_config.confidence_threshold = float(data["confidence_threshold"])

    session.touch()
    save_session(session)

    return jsonify({
        "frame_interval": session.seed_config.frame_interval,
        "confidence_threshold": session.seed_config.confidence_threshold,
    })


@interview_bp.route("/api/seeds/upload", methods=["POST"])
def seeds_upload():
    """Upload seed regions to LS with enabled=false keyframes."""
    data = request.get_json(force=True)
    session_id = data["session_id"]

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def _upload(progress):
        from .seeding_phase import upload_seeds
        return upload_seeds(session, progress)

    job_id = submit_job(_upload, name="seed_upload")
    return jsonify({"job_id": job_id}), 202


# ===========================================================================
# Shared endpoints
# ===========================================================================

@interview_bp.route("/api/job/<job_id>/progress", methods=["GET"])
def job_progress(job_id: str):
    """Poll background job status + progress."""
    progress = get_job_progress(job_id)
    if progress is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(progress)


@interview_bp.route("/api/session/<session_id>/save", methods=["POST"])
def session_save(session_id: str):
    """Manually save session to cache."""
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    save_session(session)
    return jsonify({"saved": True})
