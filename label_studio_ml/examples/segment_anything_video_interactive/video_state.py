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
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        logger.error("ffprobe failed: returncode=%s stderr=%s stdout=%s",
                      result.returncode, result.stderr[:500], result.stdout[:200])
        raise RuntimeError(f"ffprobe failed (rc={result.returncode}): {result.stderr[:500]}")
    if not result.stdout.strip():
        logger.error("ffprobe returned empty output for %s", source[:120])
        raise RuntimeError(f"ffprobe returned empty output for {source[:120]}")
    return json.loads(result.stdout)


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
        result = subprocess.run(cmd, capture_output=True, timeout=30)
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
        result = subprocess.run(cmd, capture_output=True, timeout=120)
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
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg JPEG extraction failed: {result.stderr[:300]}")
        written = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
        return written


class VideoRegistry:
    """Maps task_id to a VideoHandle. Supports local files and HTTP streaming."""

    def __init__(self):
        self._handles: Dict[str, VideoHandle] = {}
        self._lock = threading.RLock()

    def get_or_create(
        self,
        task_id: str,
        source: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> VideoHandle:
        with self._lock:
            handle = self._handles.get(task_id)
            if handle is not None and handle.source == source:
                return handle
            if handle is not None:
                handle.close()

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
            )
            self._handles[task_id] = handle
            mode = "streaming" if is_streaming else "local"
            source_kind = "url" if is_streaming else "local"
            logger.info(
                "video registered [%s] task_id=%s source_kind=%s w=%s h=%s frames=%s fps=%.2f",
                mode, task_id, source_kind, w, h, frame_count, fps,
            )
            return handle

    def drop(self, task_id: str) -> None:
        with self._lock:
            handle = self._handles.pop(task_id, None)
            if handle is not None:
                handle.close()
