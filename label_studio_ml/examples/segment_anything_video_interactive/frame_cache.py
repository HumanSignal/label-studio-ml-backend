"""Per-task frame embedding cache with spatial eviction and bounded memory.

Design (see design discussion):
- Sticky: once a frame is encoded its embedding stays until a cap forces eviction.
- Per-task and global byte caps; per-task frame-count cap.
- Spatial eviction: frames furthest from the current frame are dropped first.
- Global eviction: whole task caches are dropped when idle past TTL before
  per-frame eviction kicks in.
- Encoding is serialized per task via a single background worker thread, so we
  never contend on the GPU.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


MAX_CACHED_FRAMES_PER_TASK = _env_int("MAX_CACHED_FRAMES_PER_TASK", 500)
MAX_TASK_CACHE_MB = _env_int("MAX_TASK_CACHE_MB", 2048)
MAX_GLOBAL_CACHE_MB = _env_int("MAX_GLOBAL_CACHE_MB", 8192)
TASK_CACHE_TTL_SECONDS = _env_int("TASK_CACHE_TTL_SECONDS", 1800)


def _sizeof_embedding(embedding) -> int:
    """Best-effort byte size. Works for torch tensors, numpy arrays, dicts of either."""
    if embedding is None:
        return 0
    if isinstance(embedding, dict):
        return sum(_sizeof_embedding(v) for v in embedding.values())
    if isinstance(embedding, (list, tuple)):
        return sum(_sizeof_embedding(v) for v in embedding)
    nbytes = getattr(embedding, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    element_size = getattr(embedding, "element_size", None)
    numel = getattr(embedding, "numel", None)
    if callable(element_size) and callable(numel):
        return int(element_size()) * int(numel())
    return sys.getsizeof(embedding)


@dataclass
class _TaskCache:
    task_id: str
    embeddings: Dict[int, object] = field(default_factory=dict)
    pending: Dict[int, Future] = field(default_factory=dict)
    bytes_used: int = 0
    last_access: float = field(default_factory=time.time)
    current_frame: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


class FrameCache:
    """Thread-safe per-task frame-embedding cache with a shared encode worker pool."""

    def __init__(
        self,
        max_frames_per_task: int = MAX_CACHED_FRAMES_PER_TASK,
        max_task_mb: int = MAX_TASK_CACHE_MB,
        max_global_mb: int = MAX_GLOBAL_CACHE_MB,
        ttl_seconds: int = TASK_CACHE_TTL_SECONDS,
    ):
        self.max_frames_per_task = max_frames_per_task
        self.max_task_bytes = max_task_mb * 1024 * 1024
        self.max_global_bytes = max_global_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds

        self._tasks: Dict[str, _TaskCache] = {}
        self._global_lock = threading.RLock()
        # One worker per task is created lazily below; a single shared pool keeps
        # resource bookkeeping simple. Concurrency per task is enforced by the
        # per-task lock around encode jobs.
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="frame-encoder")

        self._evictions_total = 0

    # --- public API ------------------------------------------------------

    def touch(self, task_id: str, current_frame: int) -> None:
        task = self._get_or_create(task_id)
        with task.lock:
            task.current_frame = current_frame
            task.last_access = time.time()

    def has(self, task_id: str, frame_idx: int) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        with task.lock:
            return frame_idx in task.embeddings

    def get(self, task_id: str, frame_idx: int) -> Optional[object]:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        with task.lock:
            task.last_access = time.time()
            return task.embeddings.get(frame_idx)

    def submit(
        self,
        task_id: str,
        frame_indices: Iterable[int],
        encode_fn: Callable[[int], object],
    ) -> Tuple[List[int], List[int]]:
        """Schedule encoding for the given frames. Returns (already_cached, scheduled)."""
        task = self._get_or_create(task_id)
        already: List[int] = []
        scheduled: List[int] = []
        with task.lock:
            for idx in frame_indices:
                if idx in task.embeddings:
                    already.append(idx)
                    continue
                if idx in task.pending:
                    scheduled.append(idx)
                    continue
                future = self._pool.submit(self._encode_and_store, task_id, idx, encode_fn)
                task.pending[idx] = future
                scheduled.append(idx)
            task.last_access = time.time()
        return already, scheduled

    def ensure_encoded(
        self,
        task_id: str,
        frame_idx: int,
        encode_fn: Callable[[int], object],
        timeout: Optional[float] = None,
    ) -> object:
        """Block until frame_idx is encoded; schedule inline if not already pending."""
        task = self._get_or_create(task_id)
        with task.lock:
            if frame_idx in task.embeddings:
                task.last_access = time.time()
                return task.embeddings[frame_idx]
            future = task.pending.get(frame_idx)
            if future is None:
                future = self._pool.submit(self._encode_and_store, task_id, frame_idx, encode_fn)
                task.pending[frame_idx] = future
        future.result(timeout=timeout)
        with task.lock:
            return task.embeddings[frame_idx]

    def stats(self) -> dict:
        with self._global_lock:
            total_bytes = sum(t.bytes_used for t in self._tasks.values())
            return {
                "tasks": len(self._tasks),
                "frames_total": sum(len(t.embeddings) for t in self._tasks.values()),
                "bytes_total": total_bytes,
                "evictions_total": self._evictions_total,
                "per_task": {
                    tid: {
                        "frames": len(t.embeddings),
                        "pending": len(t.pending),
                        "bytes": t.bytes_used,
                        "current_frame": t.current_frame,
                        "last_access": t.last_access,
                    }
                    for tid, t in self._tasks.items()
                },
            }

    def drop_task(self, task_id: str) -> None:
        with self._global_lock:
            self._tasks.pop(task_id, None)

    # --- internals -------------------------------------------------------

    def _get_or_create(self, task_id: str) -> _TaskCache:
        with self._global_lock:
            task = self._tasks.get(task_id)
            if task is None:
                task = _TaskCache(task_id=task_id)
                self._tasks[task_id] = task
            return task

    def _encode_and_store(self, task_id: str, frame_idx: int, encode_fn: Callable[[int], object]) -> None:
        try:
            embedding = encode_fn(frame_idx)
        except Exception:
            logger.exception("encode failed task=%s frame=%s", task_id, frame_idx)
            task = self._tasks.get(task_id)
            if task is not None:
                with task.lock:
                    task.pending.pop(frame_idx, None)
            raise

        task = self._tasks.get(task_id)
        if task is None:
            return

        with task.lock:
            task.pending.pop(frame_idx, None)
            task.embeddings[frame_idx] = embedding
            task.bytes_used += _sizeof_embedding(embedding)
            task.last_access = time.time()

        self._enforce_caps(task_id)

    def _enforce_caps(self, task_id: str) -> None:
        self._evict_expired_tasks()
        self._evict_from_task_until_under_cap(task_id)
        self._evict_globally_until_under_cap(skip_task_id=task_id)

    def _evict_expired_tasks(self) -> None:
        now = time.time()
        expired: List[str] = []
        with self._global_lock:
            for tid, t in self._tasks.items():
                if now - t.last_access > self.ttl_seconds and len(t.embeddings) > 0:
                    expired.append(tid)
            for tid in expired:
                t = self._tasks.pop(tid, None)
                if t is not None:
                    self._evictions_total += len(t.embeddings)
                    logger.info(
                        "evict task task_id=%s frames=%s bytes=%s reason=ttl",
                        tid, len(t.embeddings), t.bytes_used,
                    )

    def _evict_from_task_until_under_cap(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            return
        with task.lock:
            while (
                task.bytes_used > self.max_task_bytes
                or len(task.embeddings) > self.max_frames_per_task
            ):
                victim = self._pick_farthest_frame(task)
                if victim is None:
                    break
                self._drop_frame(task, victim, reason="task_cap")

    def _evict_globally_until_under_cap(self, skip_task_id: Optional[str] = None) -> None:
        with self._global_lock:
            while self._global_bytes() > self.max_global_bytes:
                task = self._pick_idlest_task(skip_task_id=skip_task_id)
                if task is None:
                    break
                with task.lock:
                    victim = self._pick_farthest_frame(task)
                    if victim is None:
                        break
                    self._drop_frame(task, victim, reason="global_cap")

    def _pick_farthest_frame(self, task: _TaskCache) -> Optional[int]:
        if not task.embeddings:
            return None
        cur = task.current_frame
        return max(task.embeddings.keys(), key=lambda idx: abs(idx - cur))

    def _pick_idlest_task(self, skip_task_id: Optional[str]) -> Optional[_TaskCache]:
        candidates = [t for t in self._tasks.values() if t.task_id != skip_task_id and t.embeddings]
        if not candidates:
            # fall back to the skipped one if nothing else is available
            return self._tasks.get(skip_task_id) if skip_task_id else None
        return min(candidates, key=lambda t: t.last_access)

    def _drop_frame(self, task: _TaskCache, frame_idx: int, reason: str) -> None:
        embedding = task.embeddings.pop(frame_idx, None)
        if embedding is None:
            return
        size = _sizeof_embedding(embedding)
        task.bytes_used = max(0, task.bytes_used - size)
        self._evictions_total += 1
        logger.info(
            "evict frame task_id=%s frame=%s bytes=%s reason=%s",
            task.task_id, frame_idx, size, reason,
        )

    def _global_bytes(self) -> int:
        return sum(t.bytes_used for t in self._tasks.values())
