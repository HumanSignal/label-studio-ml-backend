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


def test_prune_sam2_tracking_state_bounds_memory_bank():
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    from model import _prune_sam2_tracking_state

    class FakePredictor:
        num_maskmem = 3
        memory_temporal_stride_for_eval = 1
        use_obj_ptrs_in_encoder = False

    def frame_out(idx):
        return {
            "maskmem_features": f"mem-{idx}",
            "maskmem_pos_enc": f"pos-{idx}",
            "pred_masks": f"mask-{idx}",
            "obj_ptr": f"ptr-{idx}",
            "object_score_logits": idx,
        }

    inference_state = {
        "output_dict_per_obj": {
            0: {"non_cond_frame_outputs": {idx: frame_out(idx) for idx in range(6)}}
        },
        "frames_tracked_per_obj": {0: {idx: {} for idx in range(6)}},
    }

    _prune_sam2_tracking_state(FakePredictor(), inference_state, current_frame_idx=5, reverse=False)

    non_cond = inference_state["output_dict_per_obj"][0]["non_cond_frame_outputs"]
    assert set(non_cond) == {3, 4, 5}
    assert set(inference_state["frames_tracked_per_obj"][0]) == {3, 4, 5}
    assert non_cond[5]["maskmem_features"] == "mem-5"
    assert "pred_masks" not in non_cond[5]
    assert "object_score_logits" not in non_cond[5]
    assert "obj_ptr" not in non_cond[5]
