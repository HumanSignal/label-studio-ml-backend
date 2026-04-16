"""Per-task video decoding state: local video path + frame reader.

Kept separate from model.py so the SAM2 import doesn't pollute unit tests that
exercise the cache/prewarm logic without CUDA.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2

logger = logging.getLogger(__name__)


@dataclass
class VideoHandle:
    task_id: str
    local_path: str
    width: int
    height: int
    frame_count: int
    fps: float
    lock: threading.RLock = field(default_factory=threading.RLock)
    _reader: Optional["cv2.VideoCapture"] = None
    _last_frame_idx: int = -1

    def read_frame(self, frame_idx: int):
        with self.lock:
            if self._reader is None:
                self._reader = cv2.VideoCapture(self.local_path)
                if not self._reader.isOpened():
                    raise RuntimeError(f"failed to open video: {self.local_path}")
            # Seek only on non-sequential access; sequential read is much faster.
            if frame_idx != self._last_frame_idx + 1:
                self._reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = self._reader.read()
            if not success:
                raise RuntimeError(f"failed to read frame {frame_idx} from {self.local_path}")
            self._last_frame_idx = frame_idx
            return frame  # BGR numpy array

    def close(self):
        with self.lock:
            if self._reader is not None:
                self._reader.release()
                self._reader = None


class VideoRegistry:
    """Maps task_id to a VideoHandle. One handle per task, created on first use."""

    def __init__(self):
        self._handles: Dict[str, VideoHandle] = {}
        self._lock = threading.RLock()

    def get_or_create(self, task_id: str, local_path: str) -> VideoHandle:
        with self._lock:
            handle = self._handles.get(task_id)
            if handle is not None and handle.local_path == local_path:
                return handle
            if handle is not None:
                handle.close()

            cap = cv2.VideoCapture(local_path)
            if not cap.isOpened():
                raise RuntimeError(f"failed to open video: {local_path}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            cap.release()

            handle = VideoHandle(
                task_id=task_id,
                local_path=local_path,
                width=width,
                height=height,
                frame_count=frame_count,
                fps=fps,
            )
            self._handles[task_id] = handle
            logger.info(
                "video registered task_id=%s path=%s w=%s h=%s frames=%s fps=%.2f",
                task_id, local_path, width, height, frame_count, fps,
            )
            return handle

    def drop(self, task_id: str) -> None:
        with self._lock:
            handle = self._handles.pop(task_id, None)
            if handle is not None:
                handle.close()
