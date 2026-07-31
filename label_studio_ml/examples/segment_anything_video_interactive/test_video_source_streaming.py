import os
import sys
import tempfile


def _drop_dependency_stub(module_name):
    module = sys.modules.get(module_name)
    if module is not None and getattr(module, "__file__", None) is None:
        sys.modules.pop(module_name, None)


def _import_model():
    os.environ.setdefault("MODEL_DIR", tempfile.mkdtemp(prefix="sam2-test-cache-"))
    _drop_dependency_stub("cv2")
    _drop_dependency_stub("numpy")
    import model

    return model


def test_ls_relative_upload_downloads_instead_of_auth_streaming():
    model = _import_model()
    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    backend._ls_host_token = lambda: ("http://ls", "token")
    backend._download_valid_video = lambda raw_url, task_id: f"/cached/{task_id}.mp4"

    source, headers = backend._resolve_video_source("/data/upload/7/video.mp4", "task")

    assert source == "/cached/task.mp4"
    assert headers is None


def test_absolute_ls_upload_downloads_instead_of_auth_streaming():
    model = _import_model()
    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    backend._ls_host_token = lambda: ("http://ls", "token")
    backend._download_valid_video = lambda raw_url, task_id: f"/cached/{task_id}.mp4"

    source, headers = backend._resolve_video_source(
        "https://ls/data/upload/7/video.mp4", "task"
    )

    assert source == "/cached/task.mp4"
    assert headers is None


def test_external_url_streams_without_ls_auth():
    model = _import_model()
    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    backend._ls_host_token = lambda: ("http://ls", "token")

    source, headers = backend._resolve_video_source(
        "https://video.example/data/upload/7/video.mp4", "task"
    )

    assert source == "https://video.example/data/upload/7/video.mp4"
    assert headers is None
