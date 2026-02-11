"""In-memory session state for the standalone ReID Interview UI.

Manages ReIDInterviewSession objects with thread-safe access.
Fully independent from the interview/state.py registry.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackInfo:
    """Parsed LS VideoRectangle track."""
    region_id: str
    sequence: List[Dict[str, Any]]  # raw LS keyframe dicts
    labels: List[str]
    meta_text: str
    frames_count: int = 0
    duration: float = 0.0


@dataclass
class CropInfo:
    """A single keyframe crop extracted from a track."""
    crop_id: str                    # "{region_id}_f{frame}"
    track_region_id: str
    frame_idx: int                  # 1-based (LS convention)
    x_pct: float
    y_pct: float
    w_pct: float
    h_pct: float
    xyxy_px: np.ndarray             # pixel coords [x1, y1, x2, y2]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crop_id": self.crop_id,
            "track_region_id": self.track_region_id,
            "frame_idx": self.frame_idx,
            "x_pct": self.x_pct,
            "y_pct": self.y_pct,
            "w_pct": self.w_pct,
            "h_pct": self.h_pct,
            "xyxy_px": self.xyxy_px.tolist(),
        }


@dataclass
class ReIDPairInfo:
    """A pair of crops for identity comparison."""
    pair_id: str
    crop_id_a: str
    crop_id_b: str
    track_a: str                    # region_id
    track_b: str                    # region_id
    similarity: float
    pool: str                       # "warmup", "calibration", "ambiguous"
    difficulty: int                 # 0=easy, 1=medium, 2=hard
    model_prediction: str           # "same" or "different"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "crop_id_a": self.crop_id_a,
            "crop_id_b": self.crop_id_b,
            "track_a": self.track_a,
            "track_b": self.track_b,
            "similarity": self.similarity,
            "pool": self.pool,
            "difficulty": self.difficulty,
            "model_prediction": self.model_prediction,
        }


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class ReIDInterviewSession:
    """Full state for one ReID interview session."""
    session_id: str
    project_id: int
    task_id: int
    annotation_id: int              # REQUIRED
    cache_key: str = ""

    # Video info
    video_path: str = ""
    video_key: str = ""
    width: int = 0
    height: int = 0
    frames_count: int = 0
    fps: float = 30.0

    # Annotation data
    tracks: List[TrackInfo] = field(default_factory=list)
    crops: Dict[str, CropInfo] = field(default_factory=dict)

    # Embedding data
    features: Dict[str, np.ndarray] = field(default_factory=dict)
    histograms: Dict[str, np.ndarray] = field(default_factory=dict)
    similarity_matrix: Optional[np.ndarray] = None
    crop_id_list: List[str] = field(default_factory=list)

    # Clustering
    clusters: Dict[int, List[str]] = field(default_factory=dict)
    n_clusters: int = 0

    # Pairs
    pairs: List[ReIDPairInfo] = field(default_factory=list)
    resolutions: Dict[str, str] = field(default_factory=dict)
    auto_resolved: Dict[str, str] = field(default_factory=dict)
    calibration_answers: Dict[str, str] = field(default_factory=dict)

    # Per-crop identity assignment (from clustering + human resolution)
    per_crop_identity: Dict[str, int] = field(default_factory=dict)

    # Gamification
    pairs_resolved_count: int = 0

    # Phase
    phase: str = "landing"          # landing, loading, reviewing, writeback, complete

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def touch(self) -> None:
        self.updated_at = time.time()

    def stats(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "phase": self.phase,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "annotation_id": self.annotation_id,
            "n_tracks": len(self.tracks),
            "n_crops": len(self.crops),
            "n_clusters": self.n_clusters,
            "n_pairs": len(self.pairs),
            "pairs_resolved": self.pairs_resolved_count,
            "video_frames": self.frames_count,
        }


# ---------------------------------------------------------------------------
# Session registry (in-memory, thread-safe) — separate from interview
# ---------------------------------------------------------------------------

_sessions: Dict[str, ReIDInterviewSession] = {}
_registry_lock = threading.Lock()


def create_session(project_id: int, task_id: int, annotation_id: Optional[int] = None) -> ReIDInterviewSession:
    """Create and register a new ReID interview session."""
    if annotation_id is None:
        raise ValueError("annotation_id is required for ReID Interview sessions")

    session_id = str(uuid.uuid4())[:12]
    cache_key = f"reid_p{project_id}_t{task_id}_a{annotation_id}"
    session = ReIDInterviewSession(
        session_id=session_id,
        project_id=project_id,
        task_id=task_id,
        annotation_id=annotation_id,
        cache_key=cache_key,
    )
    with _registry_lock:
        _sessions[session_id] = session
    logger.info("Created ReID session %s (cache_key=%s)", session_id, cache_key)
    return session


def get_session(session_id: str) -> Optional[ReIDInterviewSession]:
    """Retrieve a session by ID."""
    with _registry_lock:
        return _sessions.get(session_id)


def delete_session(session_id: str) -> bool:
    """Remove a session from the registry."""
    with _registry_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            logger.info("Deleted ReID session %s", session_id)
            return True
        return False


def list_sessions() -> List[Dict[str, Any]]:
    """Return summary of all active sessions."""
    with _registry_lock:
        return [s.stats() for s in _sessions.values()]
