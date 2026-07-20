"""Unit tests for HTTP-429 detection and ffmpeg backoff in video_state.

`video_state` imports cv2/numpy at module load but only uses them inside
methods, so we stub them in sys.modules to keep this test dependency-free
(same spirit as test_control_detect / test_ls_auth). subprocess + time.sleep
are mocked so no real process runs and no real waiting happens.

Run with: pytest test_video_fetch_retry.py -v
"""

import sys
import types
from unittest import mock

# --- stub heavy deps before importing the module under test ---------------
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

import video_state  # noqa: E402


def _proc(returncode, stderr=b""):
    return types.SimpleNamespace(returncode=returncode, stderr=stderr, stdout=b"")


# --- 429 detection --------------------------------------------------------

def test_detects_numeric_429():
    assert video_state._is_rate_limited(b"Server returned 429 Too Many Requests") is True


def test_detects_text_too_many_requests():
    assert video_state._is_rate_limited("HTTP error: too many requests") is True


def test_non_429_error_not_flagged():
    assert video_state._is_rate_limited(b"Server returned 404 Not Found") is False


def test_none_stderr_not_flagged():
    assert video_state._is_rate_limited(None) is False


# --- backoff behavior -----------------------------------------------------

def test_success_runs_once():
    with mock.patch.object(video_state.subprocess, "run", return_value=_proc(0)) as run, \
         mock.patch.object(video_state.time, "sleep") as sleep:
        result = video_state._run_with_429_backoff(["ffmpeg"], timeout=10)
    assert result.returncode == 0
    run.assert_called_once()
    sleep.assert_not_called()


def test_non_retryable_failure_runs_once():
    fail = _proc(1, b"Server returned 404 Not Found")
    with mock.patch.object(video_state.subprocess, "run", return_value=fail) as run, \
         mock.patch.object(video_state.time, "sleep") as sleep:
        result = video_state._run_with_429_backoff(["ffmpeg"], timeout=10)
    assert result.returncode == 1
    run.assert_called_once()  # 404 is not retried
    sleep.assert_not_called()


def test_retries_then_succeeds_on_429():
    seq = [_proc(1, b"429 Too Many Requests"), _proc(1, b"429 Too Many Requests"), _proc(0)]
    with mock.patch.object(video_state.subprocess, "run", side_effect=seq) as run, \
         mock.patch.object(video_state.time, "sleep") as sleep:
        result = video_state._run_with_429_backoff(["ffmpeg"], timeout=10)
    assert result.returncode == 0
    assert run.call_count == 3
    assert sleep.call_count == 2  # backoff before each retry, not after success


def test_gives_up_after_max_attempts():
    fail = _proc(1, b"429 Too Many Requests")
    with mock.patch.object(video_state, "_FETCH_RETRY_ATTEMPTS", 3), \
         mock.patch.object(video_state.subprocess, "run", return_value=fail) as run, \
         mock.patch.object(video_state.time, "sleep") as sleep:
        result = video_state._run_with_429_backoff(["ffmpeg"], timeout=10)
    assert result.returncode == 1  # last 429 returned; caller raises with stderr
    assert run.call_count == 3
    assert sleep.call_count == 2  # no sleep after the final failed attempt


def test_backoff_delay_is_capped():
    fail = _proc(1, b"429 Too Many Requests")
    delays = []
    with mock.patch.object(video_state, "_FETCH_RETRY_ATTEMPTS", 8), \
         mock.patch.object(video_state, "_FETCH_RETRY_MAX_DELAY", 5.0), \
         mock.patch.object(video_state.subprocess, "run", return_value=fail), \
         mock.patch.object(video_state.time, "sleep", side_effect=delays.append):
        video_state._run_with_429_backoff(["ffmpeg"], timeout=10)
    assert delays  # backed off at least once
    assert max(delays) <= 5.0  # never exceeds the configured cap
