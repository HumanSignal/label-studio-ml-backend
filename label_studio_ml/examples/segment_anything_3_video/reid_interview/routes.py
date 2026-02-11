"""Flask Blueprint for Standalone ReID Interview UI.

Serves the SPA at /ReID-Interview and exposes /ReID-Interview/api/* endpoints.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
from collections import OrderedDict
from typing import Any, Dict

from flask import Blueprint, jsonify, request, send_from_directory, abort

from .state import create_session, get_session, delete_session

logger = logging.getLogger(__name__)

# Ensure sibling modules are importable
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


# ---------------------------------------------------------------------------
# LRU frame cache (separate from interview's cache)
# ---------------------------------------------------------------------------
_FRAME_CACHE_SIZE = int(os.getenv("REID_FRAME_CACHE_SIZE", "64"))
_frame_cache: OrderedDict = OrderedDict()
_frame_cache_lock = threading.Lock()


def _get_cached_frame(video_path: str, frame_idx: int):
    key = (video_path, frame_idx)
    with _frame_cache_lock:
        if key in _frame_cache:
            _frame_cache.move_to_end(key)
            return _frame_cache[key]
    return None


def _put_cached_frame(video_path: str, frame_idx: int, pil_img):
    key = (video_path, frame_idx)
    with _frame_cache_lock:
        _frame_cache[key] = pil_img
        _frame_cache.move_to_end(key)
        while len(_frame_cache) > _FRAME_CACHE_SIZE:
            _frame_cache.popitem(last=False)


def _read_frame_cached(video_path: str, frame_idx: int):
    cached = _get_cached_frame(video_path, frame_idx)
    if cached is not None:
        return cached
    from seeding_common import _read_frame_pyav
    pil_img = _read_frame_pyav(video_path, frame_idx)
    if pil_img is not None:
        _put_cached_frame(video_path, frame_idx, pil_img)
    return pil_img


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------
reid_interview_bp = Blueprint(
    "reid_interview",
    __name__,
    static_folder="static",
    static_url_path="",
    url_prefix="/ReID-Interview",
)


@reid_interview_bp.after_request
def _fix_passthrough(response):
    """Convert direct-passthrough to buffered (same fix as interview)."""
    if response.direct_passthrough:
        response.direct_passthrough = False
    return response


# ===========================================================================
# SPA entry point
# ===========================================================================

@reid_interview_bp.route("/")
@reid_interview_bp.route("/index.html")
def index():
    return send_from_directory(reid_interview_bp.static_folder, "index.html")


# ===========================================================================
# Session endpoints
# ===========================================================================

@reid_interview_bp.route("/api/session/init", methods=["POST"])
def session_init():
    """Create a ReID interview session. All three IDs required."""
    data = request.get_json(force=True)
    project_id = data.get("project_id")
    task_id = data.get("task_id")
    annotation_id = data.get("annotation_id")

    if not project_id or not task_id:
        return jsonify({"error": "project_id and task_id are required"}), 400
    if not annotation_id:
        return jsonify({"error": "annotation_id is required for ReID Interview"}), 400

    try:
        project_id = int(project_id)
        task_id = int(task_id)
        annotation_id = int(annotation_id)
    except (TypeError, ValueError):
        return jsonify({"error": "IDs must be integers"}), 400

    session = create_session(project_id, task_id, annotation_id)
    return jsonify({"session_id": session.session_id, **session.stats()})


@reid_interview_bp.route("/api/session/<session_id>/status", methods=["GET"])
def session_status(session_id: str):
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(session.stats())


# ===========================================================================
# Load pipeline (background job)
# ===========================================================================

@reid_interview_bp.route("/api/load", methods=["POST"])
def load_pipeline():
    """Start the full ReID pipeline as a background job."""
    data = request.get_json(force=True)
    session_id = data.get("session_id")

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    from interview.background import submit_job

    def _run_pipeline(progress):
        from .pipeline import run_reid_pipeline
        with session._lock:
            session.phase = "loading"
        return run_reid_pipeline(session, progress)

    job_id = submit_job(_run_pipeline, name="reid_pipeline")
    return jsonify({"job_id": job_id}), 202


# ===========================================================================
# Job progress polling
# ===========================================================================

@reid_interview_bp.route("/api/job/<job_id>/progress", methods=["GET"])
def job_progress(job_id: str):
    from interview.background import get_job_progress
    progress = get_job_progress(job_id)
    if progress is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(progress)


# ===========================================================================
# Crop + frame serving
# ===========================================================================

@reid_interview_bp.route("/api/crops", methods=["GET"])
def list_crops():
    """List all crops with track and cluster info."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    crop_list = []
    for crop in session.crops.values():
        d = crop.to_dict()
        d["cluster_id"] = session.per_crop_identity.get(crop.crop_id)
        crop_list.append(d)

    return jsonify({
        "total": len(crop_list),
        "crops": crop_list,
    })


@reid_interview_bp.route("/api/crop/<crop_id>/image", methods=["GET"])
def crop_image(crop_id: str):
    """Serve a cropped bounding box as JPEG."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        abort(404)

    crop = session.crops.get(crop_id)
    if crop is None:
        abort(404)

    pil_img = _read_frame_cached(session.video_path, crop.frame_idx)
    if pil_img is None:
        abort(404)

    x1, y1, x2, y2 = [int(round(v)) for v in crop.xyxy_px]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(pil_img.width, x2)
    y2 = min(pil_img.height, y2)
    cropped = pil_img.crop((x1, y1, x2, y2))

    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf.getvalue(), 200, {
        "Content-Type": "image/jpeg",
        "Cache-Control": "public, max-age=86400",
    }


@reid_interview_bp.route("/api/frame/<int:frame_idx>", methods=["GET"])
def serve_frame(frame_idx: int):
    """Serve a raw video frame as JPEG."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        abort(404)

    pil_img = _read_frame_cached(session.video_path, frame_idx)
    if pil_img is None:
        abort(404)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.getvalue(), 200, {
        "Content-Type": "image/jpeg",
        "Cache-Control": "public, max-age=86400",
    }


# ===========================================================================
# Pairs endpoints
# ===========================================================================

@reid_interview_bp.route("/api/pairs/next", methods=["GET"])
def pairs_next():
    """Get the next batch of pairs for the swipe UI."""
    session_id = request.args.get("session_id")
    batch_size = int(request.args.get("batch_size", 10))

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    # Find unresolved pairs
    unresolved = []
    for p in session.pairs:
        if p.pair_id not in session.resolutions:
            unresolved.append(p.to_dict())
        if len(unresolved) >= batch_size:
            break

    return jsonify({
        "pairs": unresolved,
        "total_pairs": len(session.pairs),
        "resolved_count": len(session.resolutions),
        "remaining": len(session.pairs) - len(session.resolutions),
    })


@reid_interview_bp.route("/api/pairs/resolve", methods=["POST"])
def pairs_resolve():
    """Submit a pair resolution (same/different/unsure)."""
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    pair_id = data.get("pair_id")
    resolution = data.get("resolution")

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    if resolution not in ("same", "different", "unsure"):
        return jsonify({"error": "resolution must be same, different, or unsure"}), 400

    # Record resolution
    with session._lock:
        session.resolutions[pair_id] = resolution
        session.pairs_resolved_count = len(session.resolutions)
        session.touch()

    # Apply cluster update
    from .pipeline import apply_pair_resolution, compute_accuracy

    # Find the pair info
    pair_info = None
    for p in session.pairs:
        if p.pair_id == pair_id:
            pair_info = p
            break

    if pair_info:
        new_clusters = apply_pair_resolution(session.clusters, {
            "pair_id": pair_id,
            "crop_id_a": pair_info.crop_id_a,
            "crop_id_b": pair_info.crop_id_b,
            "resolution": resolution,
        })
        with session._lock:
            session.clusters = new_clusters
            session.n_clusters = len(new_clusters)
            # Update per-crop identity from new clusters
            for cid, members in new_clusters.items():
                for m in members:
                    session.per_crop_identity[m] = cid

    # Compute accuracy on calibration pairs
    accuracy = compute_accuracy(session.resolutions, session.calibration_answers)

    return jsonify({
        "resolved_count": len(session.resolutions),
        "remaining": len(session.pairs) - len(session.resolutions),
        "n_clusters": session.n_clusters,
        "accuracy": round(accuracy, 1),
        "clusters": {str(k): v for k, v in session.clusters.items()},
    })


# ===========================================================================
# Cluster visualization
# ===========================================================================

@reid_interview_bp.route("/api/clusters", methods=["GET"])
def clusters_data():
    """Current cluster state for the live map."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    nodes = []
    for cid, members in session.clusters.items():
        for crop_id in members:
            crop = session.crops.get(crop_id)
            nodes.append({
                "id": crop_id,
                "cluster": cid,
                "track": crop.track_region_id if crop else None,
                "frame": crop.frame_idx if crop else None,
            })

    return jsonify({
        "clusters": {str(k): v for k, v in session.clusters.items()},
        "n_clusters": session.n_clusters,
        "nodes": nodes,
    })


# ===========================================================================
# Write-back endpoints
# ===========================================================================

@reid_interview_bp.route("/api/writeback/preview", methods=["GET"])
def writeback_preview():
    """Dry-run preview of annotation mutations."""
    session_id = request.args.get("session_id")
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    from .annotation_writeback import compute_writeback_preview

    # Fetch original annotation
    from seeding_common import _build_ls_client, _fetch_annotation
    ls_url = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL", "")
    ls_api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
    ls = _build_ls_client(ls_url, ls_api_key)
    ann = _fetch_annotation(ls, session.annotation_id)
    ann_result = getattr(ann, "result", []) or []

    preview = compute_writeback_preview(
        ann_result, session.per_crop_identity,
        session.width, session.height,
    )
    return jsonify(preview)


@reid_interview_bp.route("/api/writeback", methods=["POST"])
def writeback_execute():
    """Execute write-back as a background job."""
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    mode = data.get("mode", "prediction")  # "prediction" or "update"

    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    from interview.background import submit_job

    def _run_writeback(progress):
        from .annotation_writeback import execute_writeback
        from seeding_common import (
            _build_ls_client, _fetch_annotation, _upload_prediction,
        )

        progress.step = "Fetching annotation..."
        progress.total = 3
        progress.current = 1

        ls_url = os.getenv("LABEL_STUDIO_HOST") or os.getenv("LABEL_STUDIO_URL", "")
        ls_api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
        ls = _build_ls_client(ls_url, ls_api_key)
        ann = _fetch_annotation(ls, session.annotation_id)
        ann_result = getattr(ann, "result", []) or []

        progress.step = "Computing mutations..."
        progress.current = 2
        new_result = execute_writeback(session, ann_result, session.per_crop_identity, progress)

        progress.step = "Uploading to Label Studio..."
        progress.current = 3

        if mode == "update":
            ls.annotations.update(
                id=session.annotation_id,
                result=new_result,
            )
        else:
            prediction = {
                "result": new_result,
                "score": 1.0,
                "model_version": f"reid-interview-ann-{session.annotation_id}",
            }
            _upload_prediction(ls, session.task_id, prediction)

        with session._lock:
            session.phase = "complete"
            session.touch()

        return {"mode": mode, "n_regions": len(new_result)}

    job_id = submit_job(_run_writeback, name="reid_writeback")
    return jsonify({"job_id": job_id}), 202
