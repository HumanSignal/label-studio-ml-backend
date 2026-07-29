import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor


def _drop_dependency_stub(module_name):
    module = sys.modules.get(module_name)
    if module is not None and getattr(module, "__file__", None) is None:
        sys.modules.pop(module_name, None)


def test_cancel_cancels_queued_tracking_futures():
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    from model import TrackingSession

    session = TrackingSession("session", total_frames=2, producers=2)
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
