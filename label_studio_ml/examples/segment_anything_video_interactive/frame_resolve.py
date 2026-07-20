"""Pure frame-index resolution for the SAM2 video interactive backend.

Kept free of heavy deps (torch/cv2/SAM2) so the conversion logic can be
unit-tested in isolation — see ``test_frame_resolve.py``.
"""

from __future__ import annotations

import math
from typing import Optional


def resolve_frame_index(
    time_ms: Optional[float],
    frame: Optional[int],
    fps: Optional[float],
    frame_count: Optional[int],
) -> int:
    """Translate an FE-supplied timestamp (ms) or 1-indexed frame into the
    BE's 0-indexed frame space.

    ``time_ms`` wins over ``frame``: timestamps are fps-agnostic, while
    ``frame`` carries the FE's config-fps assumption. If only ``frame`` is
    provided we fall back to the legacy N-1 mapping (FE is 1-indexed).

    The browser renders the frame whose start timestamp is <= currentTime,
    i.e. ``floor(currentTime * fps)``. When seeking, the FE nudges currentTime
    a couple of milliseconds *past* the frame boundary (BROWSER_TIME_PRECISION)
    so that frame is the one on screen, then sends ``time_ms = currentTime*1000``.
    Using ``math.floor`` here therefore picks exactly the on-screen frame.

    ``round()`` would tip to the *next* frame whenever the timestamp crosses
    the half-frame mark — common when the FE config framerate differs from the
    real video fps — sending SAM2 a frame ahead of what the user sees
    (BROS-1416: "SAM2 returns results for the wrong frame; move a frame forward
    and it matches").
    """
    if time_ms is not None:
        try:
            ms = float(time_ms)
        except (TypeError, ValueError):
            ms = None
        if ms is not None and fps:
            idx = math.floor((ms / 1000.0) * fps)
            if frame_count is not None:
                idx = min(idx, max(0, frame_count - 1))
            return max(0, idx)
    return max(0, int(frame if frame is not None else 1) - 1)
