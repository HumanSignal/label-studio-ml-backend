import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest


def _drop_dependency_stub(module_name):
    module = sys.modules.get(module_name)
    if module is not None and getattr(module, "__file__", None) is None:
        sys.modules.pop(module_name, None)


def test_cancel_cancels_queued_tracking_futures():
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    from model import TrackingSession

    session = TrackingSession("session", total_frames=2)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def running_job():
        calls.append("running")
        started.set()
        assert release.wait(timeout=2)

    def queued_job():
        calls.append("queued")

    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(running_job)
        assert started.wait(timeout=1)
        queued = executor.submit(queued_job)

        session.add_future(running)
        session.add_future(queued)
        session.cancel()
        release.set()

    assert queued.cancelled()
    assert calls == ["running"]
    assert session.cancelled
    assert session.done


def test_tracking_session_drain_respects_limit_and_done_after_empty():
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    from model import TrackingSession

    session = TrackingSession("session", total_frames=5)
    for frame in range(5):
        session.append_frame({"frame": frame})
    session.finish()

    first, produced, done = session.drain_new(limit=2)
    second, produced2, done2 = session.drain_new(limit=10)

    assert [f["frame"] for f in first] == [0, 1]
    assert [f["frame"] for f in second] == [2, 3, 4]
    assert produced == 5
    assert produced2 == 5
    assert done is False
    assert done2 is True


def test_client_work_sizes_are_clamped_and_invalid_values_rejected():
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    import model

    assert model._parse_bounded_int({"window": 999}, "window", 20, 20) == 20
    assert model._parse_tracking_frame_limit({"max_frames": 999}, fps=30) == model.MAX_FRAMES_TO_TRACK
    assert model._parse_tracking_frame_limit({"max_duration_ms": 999999}, fps=30) == model.MAX_FRAMES_TO_TRACK
    with pytest.raises(ValueError):
        model._parse_bounded_int({"window": -1}, "window", 20, 20)
    with pytest.raises(ValueError):
        model._parse_tracking_frame_limit({"max_duration_ms": -1}, fps=30)


def test_encode_frame_batch_limits_decode_chunk_size(monkeypatch):
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    import model

    class FakeVideo:
        def __init__(self):
            self.calls = []

        def read_frame_range(self, start, count):
            self.calls.append((start, count))
            return [f"frame-{idx}" for idx in range(start, start + count)]

    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    backend._encode_bgr = lambda frame: f"encoded-{frame}"
    video = FakeVideo()
    monkeypatch.setattr(model, "MAX_BATCH_DECODE_FRAMES", 2)

    encoded = backend._encode_frame_batch(video, [0, 1, 2, 3, 4])

    assert video.calls == [(0, 2), (2, 2), (4, 1)]
    assert encoded[4] == "encoded-frame-4"


def test_track_returns_busy_when_session_capacity_is_exhausted(monkeypatch):
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    import model

    class FakeVideo:
        width = 640
        height = 480
        frame_count = 100
        fps = 30.0

    existing = model.TrackingSession("existing", total_frames=1, task_id="task")
    with model._tracking_lock:
        model._tracking_sessions.clear()
        model._tracking_sessions["existing"] = existing
    monkeypatch.setattr(model, "MAX_TRACKING_SESSIONS", 1)

    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    response = backend._handle_track(
        "task", {}, FakeVideo(), 0, "labels", "video", "VideoRectangle"
    )

    assert response.result[0]["type"] == "track_busy"
    assert response.result[0]["value"]["max_sessions"] == 1

    with model._tracking_lock:
        model._tracking_sessions.clear()


def test_completed_session_with_queued_results_still_consumes_capacity(monkeypatch):
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    import model

    class FakeVideo:
        width = 640
        height = 480
        frame_count = 100
        fps = 30.0

    existing = model.TrackingSession("existing", total_frames=1, task_id="task")
    existing.append_frame({"frame": 0})
    existing.finish()
    with model._tracking_lock:
        model._tracking_sessions.clear()
        model._tracking_sessions["existing"] = existing
    monkeypatch.setattr(model, "MAX_TRACKING_SESSIONS", 1)
    monkeypatch.setattr(
        model._tracking_executor,
        "submit",
        lambda *_args, **_kwargs: pytest.fail("result-bearing completed session must consume capacity"),
    )

    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    response = backend._handle_track(
        "task", {}, FakeVideo(), 0, "labels", "video", "VideoRectangle", reservation=None
    )

    assert response.result[0]["type"] == "track_busy"

    with model._tracking_lock:
        model._tracking_sessions.clear()


def test_busy_track_is_rejected_before_video_resolution(monkeypatch):
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    import model

    existing = model.TrackingSession("existing", total_frames=1, task_id="task")
    with model._tracking_lock:
        model._tracking_sessions.clear()
        model._tracking_sessions["existing"] = existing
    monkeypatch.setattr(model, "MAX_TRACKING_SESSIONS", 1)

    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    backend._image_url_from_task = lambda *_args: pytest.fail("busy track must not resolve its video")

    response = backend._handle_video(
        {"id": 2, "data": {"video": "video.mp4"}},
        "task-2",
        {"event": "track"},
        "track",
        "labels",
        "video",
        "VideoRectangle",
    )

    assert response.result[0]["type"] == "track_busy"

    with model._tracking_lock:
        model._tracking_sessions.clear()


def test_tracking_session_is_bound_to_its_task():
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    import model

    session = model.TrackingSession("session", total_frames=1, task_id="task-a")
    session.append_frame({"frame": 0})
    with model._tracking_lock:
        model._tracking_sessions.clear()
        model._tracking_sessions["session"] = session

    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    progress = backend._handle_track_progress(
        "task-b", {"session_id": "session"}, "labels", "video"
    )
    cancel = backend._handle_track_cancel(
        "task-b", {"session_id": "session"}, "labels", "video"
    )

    assert progress.result[0]["value"]["error"] == "session not found"
    assert cancel.result[0]["value"]["status"] == "not_found"
    assert session.cancelled is False
    with model._tracking_lock:
        assert model._tracking_sessions["session"] is session
        model._tracking_sessions.clear()


def test_bidirectional_ranges_emit_prompt_frame_once():
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    import model

    ranges, total = model._build_tracking_ranges(
        prompt_frame=10,
        max_frames=5,
        frame_count=100,
        direction="both",
    )

    assert ranges == [
        ("forward", 10, 16, None),
        ("backward", 5, 11, 10),
    ]
    assert total == 11


def test_completed_session_ttl_starts_at_completion(monkeypatch):
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    import model

    session = model.TrackingSession("session", total_frames=0, task_id="task")
    session.created_at = 1_000.0
    session.last_access = 1_000.0
    session.done = True
    session.completed_at = 2_000.0
    with model._tracking_lock:
        model._tracking_sessions.clear()
        model._tracking_sessions["session"] = session

    monkeypatch.setattr(model.time, "time", lambda: 2_001.0)
    monkeypatch.setattr(model.FRAME_CACHE, "expire_idle", lambda: None)
    monkeypatch.setattr(model.VIDEOS, "expire_idle", lambda: None)
    model._cleanup_tracking_sessions()

    with model._tracking_lock:
        assert model._tracking_sessions["session"] is session
        model._tracking_sessions.clear()


def test_track_cancel_retains_task_runtime_until_release_or_ttl(monkeypatch):
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    import model

    session = model.TrackingSession("session", total_frames=1, task_id="task")
    with model._tracking_lock:
        model._tracking_sessions.clear()
        model._tracking_sessions["session"] = session
    releases = []
    monkeypatch.setattr(
        model,
        "_release_task_runtime_state",
        lambda task_id, reason: releases.append((task_id, reason)),
    )

    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    response = backend._handle_track_cancel(
        "task", {"session_id": "session"}, "labels", "video"
    )

    assert response.result[0]["value"]["status"] == "cancelled"
    assert releases == []


def test_accelerator_slot_enters_cuda_autocast_in_calling_thread(monkeypatch):
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    import model

    entered_threads = []

    class FakeAutocast:
        def __enter__(self):
            entered_threads.append(threading.get_ident())

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(model, "DEVICE", "cuda")
    monkeypatch.setattr(model.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model.torch, "autocast", lambda **_kwargs: FakeAutocast())

    worker_thread = []

    def run():
        worker_thread.append(threading.get_ident())
        with model._accelerator_slot():
            pass

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=2)

    assert entered_threads == worker_thread


def test_mps_initialization_does_not_enable_upstream_async_loader(monkeypatch):
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    import model

    calls = []

    class StopAfterInit(RuntimeError):
        pass

    class FakePredictor:
        def init_state(self, **kwargs):
            calls.append(kwargs)
            raise StopAfterInit

    monkeypatch.setattr(model, "DEVICE", "mps")

    with pytest.raises(StopAfterInit):
        model._init_tracking_inference_state(
            FakePredictor(),
            "/unused",
            use_on_demand_frames=True,
        )

    assert calls[0]["async_loading_frames"] is False


def test_release_sam2_inference_state_closes_loader_and_clears_state():
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    from model import _release_sam2_inference_state

    class FakeLoader:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakePredictor:
        def __init__(self):
            self.reset_called = False

        def reset_state(self, inference_state):
            self.reset_called = True

    loader = FakeLoader()
    predictor = FakePredictor()
    inference_state = {
        "images": loader,
        "cached_features": {0: object()},
        "constants": {"maskmem_pos_enc": object()},
        "point_inputs_per_obj": {0: {}},
        "mask_inputs_per_obj": {0: {}},
        "output_dict_per_obj": {0: {"non_cond_frame_outputs": {1: object()}}},
        "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {0: object()}}},
        "frames_tracked_per_obj": {0: {1: {}}},
    }

    _release_sam2_inference_state(predictor, inference_state)

    assert predictor.reset_called
    assert loader.closed
    assert inference_state == {}


def test_release_sam2_inference_state_clears_older_async_loader():
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    from model import _release_sam2_inference_state

    class OlderLoader:
        def __init__(self):
            self.images = [object(), object()]
            self.exception = None

    loader = OlderLoader()
    inference_state = {"images": loader, "cached_features": {0: object()}}

    _release_sam2_inference_state(None, inference_state)

    assert loader.images == []
    assert isinstance(loader.exception, RuntimeError)
    assert inference_state == {}
