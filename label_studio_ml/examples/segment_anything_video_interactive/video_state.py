"""Per-task video decoding state: local file or HTTP streaming.

Supports two modes:
  * Local: cv2.VideoCapture on a downloaded file (legacy fallback).
  * Streaming: ffmpeg/ffprobe over HTTP with auth headers. No full download —
    only the requested frames are fetched via range requests.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Streaming frames from LS issues one HTTP request per ffmpeg/ffprobe call.
# Under a prewarm burst that can trip LS's rate limit (HTTP 429). ffmpeg
# doesn't surface `Retry-After`, so we detect the 429 in its stderr and retry
# with exponential backoff + jitter.
_FETCH_RETRY_ATTEMPTS = int(os.getenv("LS_FETCH_RETRY_ATTEMPTS", "4"))
_FETCH_RETRY_BASE_DELAY = float(os.getenv("LS_FETCH_RETRY_BASE_DELAY", "1.0"))
_FETCH_RETRY_MAX_DELAY = float(os.getenv("LS_FETCH_RETRY_MAX_DELAY", "30.0"))


def _is_rate_limited(stderr) -> bool:
    """True if ffmpeg/ffprobe stderr indicates an HTTP 429 from the server."""
    if stderr is None:
        return False
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
    text = text.lower()
    return "429" in text or "too many requests" in text


def _run_with_429_backoff(cmd: List[str], *, timeout: int, text: bool = False):
    """Run an ffmpeg/ffprobe command, retrying on HTTP 429 with exponential
    backoff + jitter. Returns the final CompletedProcess (success, a
    non-retryable failure, or the last 429 after exhausting attempts) — the
    caller inspects returncode as before."""
    result = None
    for attempt in range(_FETCH_RETRY_ATTEMPTS):
        result = subprocess.run(cmd, capture_output=True, text=text, timeout=timeout)
        if result.returncode == 0 or not _is_rate_limited(result.stderr):
            return result
        if attempt == _FETCH_RETRY_ATTEMPTS - 1:
            break
        delay = min(
            _FETCH_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5),
            _FETCH_RETRY_MAX_DELAY,
        )
        logger.warning(
            "LS rate-limited (HTTP 429); backing off %.1fs (attempt %d/%d)",
            delay, attempt + 1, _FETCH_RETRY_ATTEMPTS,
        )
        time.sleep(delay)
    return result


def _validate_probe_source(source: str) -> str:
    """Validate and normalize ffprobe input source from task data."""
    if not isinstance(source, str):
        raise ValueError("video source must be a string")
    source = source.strip()
    if not source:
        raise ValueError("video source is empty")
    if source.startswith("-"):
        raise ValueError("video source cannot start with '-'")

    if source.startswith("http://") or source.startswith("https://"):
        parsed = urlparse(source)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("invalid video URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("video URL must not include credentials")
        if any(ch in source for ch in ("\r", "\n", "\x00")):
            raise ValueError("video source contains invalid control characters")
        return source

    return os.path.abspath(source)


def _probe_video(source: str, headers: Optional[Dict[str, str]] = None) -> dict:
    """Run ffprobe on a local path or URL, return parsed JSON."""
    source = _validate_probe_source(source)
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format"]
    if headers:
        hdr_str = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
        cmd.extend(["-headers", hdr_str])
    cmd.extend(["--", source])
    source_kind = "url" if source.startswith("http://") or source.startswith("https://") else "local"
    logger.info("ffprobe started (source_kind=%s)", source_kind)
    result = _run_with_429_backoff(cmd, timeout=30, text=True)
    if result.returncode != 0:
        logger.error("ffprobe failed: returncode=%s stderr=%s stdout=%s",
                      result.returncode, result.stderr[:500], result.stdout[:200])
        raise RuntimeError(f"ffprobe failed (rc={result.returncode}): {result.stderr[:500]}")
    if not result.stdout.strip():
        logger.error("ffprobe returned empty output (source_kind=%s)", source_kind)
        raise RuntimeError(f"ffprobe returned empty output (source_kind={source_kind})")
    return json.loads(result.stdout)


def video_is_readable(path: str) -> Tuple[bool, str]:
    """Can a downloaded video file actually be decoded? Returns (ok, reason).

    ffprobe rather than cv2 because its error says *why*: a download truncated
    by LS's storage proxy fails with "moov atom not found", which cv2 reports
    exactly the same way as a codec its build doesn't support. Falls back to
    cv2 when ffprobe isn't installed.
    """
    if not os.path.exists(path):
        return False, "file does not exist"
    size = os.path.getsize(path)
    try:
        _parse_probe(_probe_video(path))
        return True, ""
    except FileNotFoundError:  # no ffprobe on PATH
        cap = cv2.VideoCapture(path)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        return (True, "") if ok else (False, f"cv2 could not decode it ({size} bytes)")
    except Exception as e:
        return False, f"{e} ({size} bytes)"


def _parse_probe(info: dict) -> Tuple[int, int, int, float]:
    """Extract (width, height, frame_count, fps) from ffprobe output."""
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            w = int(stream.get("width", 0))
            h = int(stream.get("height", 0))
            nb = int(stream.get("nb_frames", 0))
            r_frame_rate = stream.get("r_frame_rate", "30/1")
            num, den = (r_frame_rate.split("/") + ["1"])[:2]
            fps = float(num) / float(den) if float(den) else 30.0
            if nb == 0:
                duration = float(info.get("format", {}).get("duration", 0))
                nb = int(duration * fps) if duration else 0
            return w, h, nb, fps
    raise RuntimeError("no video stream found in ffprobe output")


@dataclass
class VideoHandle:
    """Unified video handle — works with local files or HTTP URLs."""

    task_id: str
    source: str
    width: int
    height: int
    frame_count: int
    fps: float
    is_streaming: bool = False
    headers: Optional[Dict[str, str]] = None
    raw_url: Optional[str] = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    _reader: Optional[cv2.VideoCapture] = None
    _last_frame_idx: int = -1

    def read_frame(self, frame_idx: int) -> np.ndarray:
        if self.is_streaming:
            return self._read_frame_ffmpeg(frame_idx)
        return self._read_frame_cv2(frame_idx)

    def read_frame_range(self, start: int, count: int) -> List[np.ndarray]:
        """Read a contiguous range of frames. More efficient than per-frame
        calls for streaming (single ffmpeg invocation)."""
        if self.is_streaming:
            return self._read_range_ffmpeg(start, count)
        return [self.read_frame(start + i) for i in range(count)]

    def close(self):
        with self.lock:
            if self._reader is not None:
                self._reader.release()
                self._reader = None

    # --- cv2 (local file) ------------------------------------------------

    def _read_frame_cv2(self, frame_idx: int) -> np.ndarray:
        with self.lock:
            if self._reader is None:
                self._reader = cv2.VideoCapture(self.source)
                if not self._reader.isOpened():
                    raise RuntimeError(f"failed to open video: {self.source}")
            if frame_idx != self._last_frame_idx + 1:
                self._reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = self._reader.read()
            if not success:
                raise RuntimeError(f"failed to read frame {frame_idx}")
            self._last_frame_idx = frame_idx
            return frame

    # --- ffmpeg (streaming) ----------------------------------------------

    def _ffmpeg_input_args(self) -> List[str]:
        args: List[str] = []
        if self.headers:
            hdr_str = "".join(f"{k}: {v}\r\n" for k, v in self.headers.items())
            args.extend(["-headers", hdr_str])
        return args

    def _read_frame_ffmpeg(self, frame_idx: int) -> np.ndarray:
        timestamp = frame_idx / self.fps if self.fps else 0
        cmd = (
            ["ffmpeg", "-hide_banner", "-loglevel", "error"]
            + self._ffmpeg_input_args()
            + ["-ss", f"{timestamp:.4f}", "-i", self.source,
               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "pipe:1"]
        )
        result = _run_with_429_backoff(cmd, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg frame read failed: {result.stderr[:300]}")
        expected = self.width * self.height * 3
        if len(result.stdout) < expected:
            raise RuntimeError(
                f"ffmpeg returned {len(result.stdout)} bytes, expected {expected}"
            )
        return np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape(
            self.height, self.width, 3
        )

    def _read_range_ffmpeg(self, start: int, count: int) -> List[np.ndarray]:
        timestamp = start / self.fps if self.fps else 0
        cmd = (
            ["ffmpeg", "-hide_banner", "-loglevel", "error"]
            + self._ffmpeg_input_args()
            + ["-ss", f"{timestamp:.4f}", "-i", self.source,
               "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "bgr24",
               "pipe:1"]
        )
        result = _run_with_429_backoff(cmd, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg range read failed: {result.stderr[:300]}")
        frame_bytes = self.width * self.height * 3
        raw = result.stdout
        frames: List[np.ndarray] = []
        for i in range(count):
            offset = i * frame_bytes
            if offset + frame_bytes > len(raw):
                break
            frame = np.frombuffer(
                raw[offset: offset + frame_bytes], dtype=np.uint8
            ).reshape(self.height, self.width, 3)
            frames.append(frame)
        return frames

    def write_frame_range_as_jpegs(self, start: int, count: int, output_dir: str) -> int:
        """Extract a frame range and write as numbered JPEGs to output_dir.
        Returns the number of frames written. Used by SAM2 video predictor
        which requires a directory of JPEG files.

        Prefers ffmpeg for both streaming and local sources — one hardware-
        decoded pass beats cv2's per-frame seek + imwrite loop.
        """
        try:
            return self._write_range_jpegs_ffmpeg(start, count, output_dir)
        except (FileNotFoundError, RuntimeError) as e:
            if self.is_streaming:
                raise
            logger.warning("ffmpeg extract failed (%s), falling back to cv2", e)
            frames = [self.read_frame(start + i) for i in range(count)]
            for i, frame in enumerate(frames):
                cv2.imwrite(os.path.join(output_dir, f"{i:05d}.jpg"), frame)
            return len(frames)

    def _write_range_jpegs_ffmpeg(self, start: int, count: int, output_dir: str) -> int:
        """Use ffmpeg to extract frames directly as JPEGs — no Python
        decode/re-encode round-trip."""
        timestamp = start / self.fps if self.fps else 0
        pattern = os.path.join(output_dir, "%05d.jpg")
        cmd = (
            ["ffmpeg", "-hide_banner", "-loglevel", "error"]
            + self._ffmpeg_input_args()
            + ["-ss", f"{timestamp:.4f}", "-i", self.source,
               "-frames:v", str(count), "-q:v", "2",
               "-start_number", "0", pattern]
        )
        result = _run_with_429_backoff(cmd, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg JPEG extraction failed: {result.stderr[:300]}")
        written = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
        return written


@dataclass
class _VideoEntry:
    task_id: str
    raw_url: str
    future: Future
    handle: Optional[VideoHandle] = None
    leases: int = 0
    drop_requested: bool = False
    last_access: float = field(default_factory=time.time)


class VideoRegistry:
    """Lease-safe per-task video registry with single-flight resolution."""

    def __init__(self, ttl_seconds: float = 1800):
        self.ttl_seconds = ttl_seconds
        self._entries: Dict[Tuple[str, str], _VideoEntry] = {}
        self._lock = threading.RLock()

    @contextmanager
    def acquire(
        self,
        task_id: str,
        raw_url: str,
        resolve_fn: Callable[[], VideoHandle],
    ) -> Iterator[VideoHandle]:
        key = (task_id, raw_url)
        entry, creator = self._entry_for_acquire(key, resolve_fn)
        try:
            handle = entry.future.result()
            with self._lock:
                entry.handle = handle
                entry.last_access = time.time()
            yield handle
        finally:
            self._release_entry(key, entry)

    def open_handle(
        self,
        task_id: str,
        source: str,
        headers: Optional[Dict[str, str]] = None,
        raw_url: Optional[str] = None,
    ) -> VideoHandle:
        is_streaming = source.startswith("http://") or source.startswith("https://")

        if is_streaming:
            info = _probe_video(source, headers)
            w, h, frame_count, fps = _parse_probe(info)
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"failed to open video: {source}")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            cap.release()

        handle = VideoHandle(
            task_id=task_id,
            source=source,
            width=w,
            height=h,
            frame_count=frame_count,
            fps=fps,
            is_streaming=is_streaming,
            headers=headers if is_streaming else None,
            raw_url=raw_url,
        )
        mode = "streaming" if is_streaming else "local"
        source_kind = "url" if is_streaming else "local"
        logger.info(
            "video registered [%s] task_id=%s source_kind=%s w=%s h=%s frames=%s fps=%.2f",
            mode, task_id, source_kind, w, h, frame_count, fps,
        )
        return handle

    def retain(self, handle: VideoHandle) -> Callable[[], None]:
        """Hold a lease for background work that outlives request handling."""
        retained_key: Optional[Tuple[str, str]] = None
        retained_entry: Optional[_VideoEntry] = None
        with self._lock:
            for key, entry in self._entries.items():
                resolved = self._resolved_handle(entry)
                if resolved is handle:
                    entry.leases += 1
                    entry.last_access = time.time()
                    retained_key = key
                    retained_entry = entry
                    break
        if retained_key is None or retained_entry is None:
            return lambda: None

        released = False
        release_lock = threading.Lock()

        def release() -> None:
            nonlocal released
            with release_lock:
                if released:
                    return
                released = True
            self._release_entry(retained_key, retained_entry)

        return release

    def drop(self, task_id: str) -> None:
        to_close: List[VideoHandle] = []
        with self._lock:
            for key, entry in list(self._entries.items()):
                if key[0] != task_id:
                    continue
                entry.drop_requested = True
                if entry.leases == 0:
                    self._entries.pop(key, None)
                    handle = self._resolved_handle(entry)
                    if handle is not None:
                        to_close.append(handle)
        for handle in to_close:
            handle.close()

    def expire_idle(self, ttl_seconds: Optional[float] = None) -> None:
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        now = time.time()
        to_close: List[VideoHandle] = []
        with self._lock:
            for key, entry in list(self._entries.items()):
                if entry.leases > 0 or now - entry.last_access <= ttl:
                    continue
                self._entries.pop(key, None)
                handle = self._resolved_handle(entry)
                if handle is not None:
                    to_close.append(handle)
        for handle in to_close:
            handle.close()

    def _entry_for_acquire(
        self,
        key: Tuple[str, str],
        resolve_fn: Callable[[], VideoHandle],
    ) -> Tuple[_VideoEntry, bool]:
        with self._lock:
            entry = self._entries.get(key)
            stale = (
                entry is not None
                and entry.leases == 0
                and (entry.drop_requested or self._entry_is_missing(entry))
            )
            if entry is None or stale:
                if entry is not None:
                    handle = self._resolved_handle(entry)
                    if handle is not None:
                        handle.close()
                future: Future = Future()
                entry = _VideoEntry(task_id=key[0], raw_url=key[1], future=future)
                self._entries[key] = entry
                creator = True
            else:
                creator = False
            entry.leases += 1
            entry.last_access = time.time()

        if creator:
            try:
                handle = resolve_fn()
            except Exception as exc:
                with self._lock:
                    self._entries.pop(key, None)
                entry.future.set_exception(exc)
                raise
            else:
                entry.handle = handle
                entry.future.set_result(handle)
        return entry, creator

    def _release_entry(self, key: Tuple[str, str], entry: _VideoEntry) -> None:
        handle_to_close: Optional[VideoHandle] = None
        with self._lock:
            entry.leases = max(0, entry.leases - 1)
            entry.last_access = time.time()
            if entry.leases == 0 and entry.drop_requested:
                self._entries.pop(key, None)
                handle_to_close = self._resolved_handle(entry)
        if handle_to_close is not None:
            handle_to_close.close()

    def _entry_is_missing(self, entry: _VideoEntry) -> bool:
        handle = self._resolved_handle(entry)
        return handle is not None and self._handle_missing(handle)

    def _resolved_handle(self, entry: _VideoEntry) -> Optional[VideoHandle]:
        if not entry.future.done() or entry.future.cancelled():
            return entry.handle
        try:
            return entry.future.result()
        except Exception:
            return entry.handle

    def _handle_missing(self, handle: VideoHandle) -> bool:
        return not handle.is_streaming and not os.path.exists(handle.source)
