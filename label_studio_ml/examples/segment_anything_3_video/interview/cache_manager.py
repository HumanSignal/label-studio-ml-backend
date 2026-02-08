"""Disk cache for interview sessions.

Persists session state under /data/adapters/{cache_key}/ so that
sessions can survive container restarts and be reused across tasks.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .state import (
    CropData, CropLabel, CropSource, InterviewSession, Phase,
    ReIDPair, SeedConfig,
)

logger = logging.getLogger(__name__)

CACHE_ROOT = os.getenv("INTERVIEW_CACHE_ROOT", "/data/adapters")
PROJECT_INDEX_FILE = "_project_index.json"

_index_lock = threading.Lock()


def _cache_dir(cache_key: str) -> Path:
    return Path(CACHE_ROOT) / cache_key


def cache_exists(cache_key: str) -> bool:
    return (_cache_dir(cache_key) / "config.json").is_file()


def list_project_caches(project_id: int) -> List[Dict[str, Any]]:
    """Find all caches belonging to a project via the project index."""
    index = _read_project_index()
    return index.get(str(project_id), [])


# ---------------------------------------------------------------------------
# Project index
# ---------------------------------------------------------------------------

def _index_path() -> Path:
    return Path(CACHE_ROOT) / PROJECT_INDEX_FILE


def _read_project_index() -> Dict[str, List[Dict[str, Any]]]:
    path = _index_path()
    if not path.is_file():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read project index: %s", e)
        return {}


def _write_project_index(index: Dict[str, List[Dict[str, Any]]]) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2)
    tmp.rename(path)


def _update_project_index(project_id: int, cache_key: str, task_id: int, phase: str) -> None:
    with _index_lock:
        index = _read_project_index()
        pid = str(project_id)
        entries = index.get(pid, [])

        # Update or insert
        found = False
        for entry in entries:
            if entry.get("cache_key") == cache_key:
                entry["phase"] = phase
                entry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                found = True
                break
        if not found:
            entries.append({
                "cache_key": cache_key,
                "task_id": task_id,
                "phase": phase,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

        index[pid] = entries
        _write_project_index(index)


def _remove_from_project_index(project_id: int, cache_key: str) -> None:
    with _index_lock:
        index = _read_project_index()
        pid = str(project_id)
        entries = index.get(pid, [])
        entries = [e for e in entries if e.get("cache_key") != cache_key]
        if entries:
            index[pid] = entries
        else:
            index.pop(pid, None)
        _write_project_index(index)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_session(session: InterviewSession) -> None:
    """Persist session state to disk."""
    d = _cache_dir(session.cache_key)
    d.mkdir(parents=True, exist_ok=True)

    # config.json — lightweight session metadata
    config = {
        "session_id": session.session_id,
        "project_id": session.project_id,
        "task_id": session.task_id,
        "annotation_id": session.annotation_id,
        "cache_key": session.cache_key,
        "phase": session.phase.value,
        "video_path": session.video_path,
        "video_key": session.video_key,
        "width": session.width,
        "height": session.height,
        "frames_count": session.frames_count,
        "fps": session.fps,
        "prompts": session.prompts,
        "sampled_frames": session.sampled_frames,
        "model_trained": session.model_trained,
        "training_epochs": session.training_epochs,
        "training_accuracy": session.training_accuracy,
        "n_identities": session.n_identities,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "seed_config": {
            "frame_interval": session.seed_config.frame_interval,
            "confidence_threshold": session.seed_config.confidence_threshold,
        },
    }
    _write_json(d / "config.json", config)

    # crops_metadata.json — all crop data (excluding numpy arrays)
    crops_meta = {cid: c.to_dict() for cid, c in session.crops.items()}
    _write_json(d / "crops_metadata.json", crops_meta)

    # labels.json — just the labels for quick access
    labels = {cid: c.label.value for cid, c in session.crops.items()}
    _write_json(d / "labels.json", labels)

    # prompts.json
    _write_json(d / "prompts.json", session.prompts)

    # features.npz — DINOv3 features (float16 for space)
    _save_features(d, session)

    # clusters.json — ReID data
    reid_data = {
        "clusters": {str(k): v for k, v in session.reid_clusters.items()},
        "pairs": {pid: _pair_to_dict(p) for pid, p in session.reid_pairs.items()},
    }
    _write_json(d / "clusters.json", reid_data)

    # Update project index
    _update_project_index(
        session.project_id, session.cache_key, session.task_id, session.phase.value
    )

    logger.info("Saved session %s to %s", session.session_id, d)


def load_session(cache_key: str) -> Optional[InterviewSession]:
    """Load session state from disk cache."""
    d = _cache_dir(cache_key)
    config_path = d / "config.json"
    if not config_path.is_file():
        return None

    config = _read_json(config_path)
    if config is None:
        return None

    session = InterviewSession(
        session_id=config.get("session_id", ""),
        project_id=config.get("project_id", 0),
        task_id=config.get("task_id", 0),
        annotation_id=config.get("annotation_id"),
        cache_key=cache_key,
        video_path=config.get("video_path", ""),
        video_key=config.get("video_key", ""),
        width=config.get("width", 0),
        height=config.get("height", 0),
        frames_count=config.get("frames_count", 0),
        fps=config.get("fps", 30.0),
        phase=Phase(config.get("phase", "init")),
        prompts=config.get("prompts", []),
        sampled_frames=config.get("sampled_frames", []),
        model_trained=config.get("model_trained", False),
        training_epochs=config.get("training_epochs", 0),
        training_accuracy=config.get("training_accuracy", 0.0),
        n_identities=config.get("n_identities", 0),
        created_at=config.get("created_at", time.time()),
        updated_at=config.get("updated_at", time.time()),
    )

    sc = config.get("seed_config", {})
    session.seed_config = SeedConfig(
        frame_interval=sc.get("frame_interval", 5),
        confidence_threshold=sc.get("confidence_threshold", 0.8),
    )

    # Load crops
    crops_meta = _read_json(d / "crops_metadata.json")
    if crops_meta:
        for cid, cdata in crops_meta.items():
            session.crops[cid] = CropData.from_dict(cdata)

    # Load features
    _load_features(d, session)

    # Load ReID data
    reid_data = _read_json(d / "clusters.json")
    if reid_data:
        clusters = reid_data.get("clusters", {})
        session.reid_clusters = {int(k): v for k, v in clusters.items()}
        pairs = reid_data.get("pairs", {})
        session.reid_pairs = {pid: _pair_from_dict(pdata) for pid, pdata in pairs.items()}

    logger.info("Loaded session %s from %s (phase=%s)", session.session_id, d, session.phase.value)
    return session


def delete_cache(cache_key: str, project_id: Optional[int] = None) -> bool:
    """Remove a cache directory and its project index entry."""
    import shutil
    d = _cache_dir(cache_key)
    if d.is_dir():
        shutil.rmtree(d)
        logger.info("Deleted cache %s", cache_key)

    if project_id is not None:
        _remove_from_project_index(project_id, cache_key)

    return True


def save_model(cache_key: str, state_dict: dict) -> None:
    """Save MLP classifier state_dict to disk."""
    import torch
    d = _cache_dir(cache_key)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, d / "model.pt")
    logger.info("Saved MLP model to %s", d / "model.pt")


def load_model(cache_key: str) -> Optional[dict]:
    """Load MLP classifier state_dict from disk."""
    import torch
    path = _cache_dir(cache_key) / "model.pt"
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.rename(path)


def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def _save_features(d: Path, session: InterviewSession) -> None:
    """Save DINOv3 features and metadata arrays to npz."""
    ids = []
    feats = []
    metas = []
    for cid, crop in session.crops.items():
        if crop.features is not None:
            ids.append(cid)
            feats.append(crop.features)
            if crop.metadata is not None:
                metas.append(crop.metadata)

    if feats:
        feat_arr = np.stack(feats).astype(np.float16)
        meta_arr = np.stack(metas).astype(np.float16) if metas and len(metas) == len(feats) else np.array([])
        np.savez_compressed(
            d / "features.npz",
            ids=np.array(ids, dtype=object),
            features=feat_arr,
            metadata=meta_arr,
        )


def _load_features(d: Path, session: InterviewSession) -> None:
    """Load DINOv3 features back into session crops."""
    path = d / "features.npz"
    if not path.is_file():
        return
    try:
        data = np.load(path, allow_pickle=True)
        ids = data["ids"]
        feats = data["features"].astype(np.float32)
        metas = data.get("metadata")
        if metas is not None and metas.size > 0:
            metas = metas.astype(np.float32)

        for i, cid in enumerate(ids):
            cid_str = str(cid)
            crop = session.crops.get(cid_str)
            if crop is not None:
                crop.features = feats[i]
                if metas is not None and metas.size > 0 and i < len(metas):
                    crop.metadata = metas[i]
    except Exception as e:
        logger.warning("Failed to load features from %s: %s", path, e)


def _pair_to_dict(pair: ReIDPair) -> Dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "crop_id_a": pair.crop_id_a,
        "crop_id_b": pair.crop_id_b,
        "cluster_a": pair.cluster_a,
        "cluster_b": pair.cluster_b,
        "pool": pair.pool,
        "similarity": pair.similarity,
        "resolution": pair.resolution,
    }


def _pair_from_dict(d: Dict[str, Any]) -> ReIDPair:
    return ReIDPair(
        pair_id=d["pair_id"],
        crop_id_a=d["crop_id_a"],
        crop_id_b=d["crop_id_b"],
        cluster_a=d["cluster_a"],
        cluster_b=d["cluster_b"],
        pool=d["pool"],
        similarity=d["similarity"],
        resolution=d.get("resolution"),
    )
