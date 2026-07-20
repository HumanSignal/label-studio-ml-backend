"""Unit tests for resolve_frame_index (BROS-1416).

These run without torch/cv2/SAM2: the frame-resolution math lives in the
dependency-free ``frame_resolve`` module so it can be exercised in isolation.

Run with: pytest test_frame_resolve.py -v
"""

from frame_resolve import resolve_frame_index


def test_time_ms_resolves_to_displayed_frame_not_next():
    """Regression for BROS-1416.

    The FE seeks ~2ms past a frame boundary (BROWSER_TIME_PRECISION) so the
    browser renders the intended frame, then sends ``time_ms = currentTime*1000``.
    The browser displays ``floor(currentTime * fps)``. When the FE config
    framerate differs from the real video fps the timestamp can land in the
    *upper* half of a real-fps frame interval — e.g. config 25fps frame 4
    (0-based 3 -> 3/25 + 0.002 = 0.122s) against a real 30fps video:
    0.122 * 30 = 3.66, so the on-screen frame is floor(3.66) == 3.

    ``round(3.66) == 4`` would send SAM2 the *next* frame, which is the bug.
    """
    idx = resolve_frame_index(time_ms=122.0, frame=4, fps=30.0, frame_count=1000)
    assert idx == 3


def test_time_ms_on_boundary_nudge_stays_on_frame():
    """A timestamp nudged just past a frame's start resolves to that frame."""
    # 0-based frame 160 at 30fps starts at 160/30 = 5.33333s; FE nudges +2ms.
    time_ms = (160 / 30.0) * 1000 + 2
    assert resolve_frame_index(time_ms=time_ms, frame=161, fps=30.0, frame_count=1000) == 160


def test_time_ms_zero_is_first_frame():
    assert resolve_frame_index(time_ms=0.0, frame=1, fps=30.0, frame_count=1000) == 0


def test_time_ms_clamped_to_last_frame():
    # Way past the end -> clamped to frame_count - 1.
    assert resolve_frame_index(time_ms=10_000_000.0, frame=1, fps=30.0, frame_count=300) == 299


def test_falls_back_to_frame_when_no_time_ms():
    # Legacy 1-indexed FE frame -> 0-indexed BE frame.
    assert resolve_frame_index(time_ms=None, frame=161, fps=30.0, frame_count=1000) == 160


def test_falls_back_to_frame_when_fps_missing():
    assert resolve_frame_index(time_ms=5000.0, frame=42, fps=0, frame_count=1000) == 41
    assert resolve_frame_index(time_ms=5000.0, frame=42, fps=None, frame_count=1000) == 41


def test_falls_back_to_frame_when_time_ms_invalid():
    assert resolve_frame_index(time_ms="not-a-number", frame=42, fps=30.0, frame_count=1000) == 41


def test_frame_defaults_to_one_when_absent():
    assert resolve_frame_index(time_ms=None, frame=None, fps=30.0, frame_count=1000) == 0


def test_floor_matches_browser_across_a_sweep():
    """For any currentTime, the resolved frame equals the frame the browser
    shows: floor(currentTime * fps). round() diverges on upper-half timestamps.
    """
    fps = 30.0
    for ms in range(0, 5000, 7):
        expected = int((ms / 1000.0) * fps)  # floor for non-negative
        assert resolve_frame_index(time_ms=float(ms), frame=1, fps=fps, frame_count=100000) == expected
