import os
import sys
import tempfile
from urllib.parse import parse_qs, urlparse


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


def test_legacy_upload_path_rewrites_to_storage_proxy():
    model = _import_model()

    url = model._ls_uploaded_stream_url("/data/upload/7/my video.mp4", "http://ls")

    parsed = urlparse(url)
    assert (
        f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        == "http://ls/storage-data/uploaded/"
    )
    assert parse_qs(parsed.query) == {"filepath": ["upload/7/my video.mp4"]}


def test_raw_upload_key_rewrites_to_storage_proxy():
    model = _import_model()

    url = model._ls_uploaded_stream_url("upload/7/video.mp4", "http://ls/")

    assert url == "http://ls/storage-data/uploaded/?filepath=upload/7/video.mp4"


def test_existing_storage_proxy_path_is_preserved():
    model = _import_model()

    url = model._ls_uploaded_stream_url(
        "/storage-data/uploaded/?filepath=upload/7/video.mp4", "http://ls"
    )

    assert url == "http://ls/storage-data/uploaded/?filepath=upload/7/video.mp4"


def test_resolve_video_source_streams_uploaded_files_via_proxy(monkeypatch):
    model = _import_model()
    monkeypatch.setattr(model, "STREAM_LS_UPLOADED_FILES", True)
    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    backend._ls_host_token = lambda: ("http://ls", "token")

    source, headers = backend._resolve_video_source("/data/upload/7/video.mp4", "task")

    assert source == "http://ls/storage-data/uploaded/?filepath=upload/7/video.mp4"
    assert headers["Authorization"] == "Token token"


def test_absolute_ls_upload_streams_via_proxy(monkeypatch):
    model = _import_model()
    monkeypatch.setattr(model, "STREAM_LS_UPLOADED_FILES", True)
    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    backend._ls_host_token = lambda: ("http://ls", "token")

    source, headers = backend._resolve_video_source(
        "https://ls/data/upload/7/video.mp4", "task"
    )

    assert source == "http://ls/storage-data/uploaded/?filepath=upload/7/video.mp4"
    assert headers["Authorization"] == "Token token"


def test_uploaded_file_streaming_can_be_disabled(monkeypatch):
    model = _import_model()
    monkeypatch.setattr(model, "STREAM_LS_UPLOADED_FILES", False)
    monkeypatch.setattr(model, "STREAM_LS_UPLOADS", False)
    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    backend._ls_host_token = lambda: ("http://ls", "token")
    backend._download_valid_video = (
        lambda raw_url, task_id: f"/cached/{task_id}.mp4"
    )

    source, headers = backend._resolve_video_source("/data/upload/7/video.mp4", "task")

    assert source == "/cached/task.mp4"
    assert headers is None


def test_stream_probe_failure_falls_back_to_download(monkeypatch):
    model = _import_model()
    monkeypatch.setattr(model, "STREAM_LS_UPLOADED_FILES", True)
    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    backend._ls_host_token = lambda: ("http://ls", "token")
    backend._download_valid_video = lambda raw_url, task_id: "/cached/task.mp4"

    class FakeVideo:
        fps = 30.0
        frame_count = 10
        width = 640
        height = 480

    class FakeRegistry:
        def __init__(self):
            self.sources = []

        def get(self, task_id, raw_url=None):
            return None

        def get_or_create(self, task_id, source, headers=None, raw_url=None):
            self.sources.append((source, headers))
            if len(self.sources) == 1:
                raise RuntimeError("ffprobe failed")
            return FakeVideo()

    registry = FakeRegistry()
    monkeypatch.setattr(model, "VIDEOS", registry)

    result = backend._handle_video(
        {"data": {"video": "/data/upload/7/video.mp4"}},
        "task",
        {},
        "predict",
        "labels",
        "video",
        "VideoRectangle",
    )

    assert result.result == []
    assert registry.sources[0][0] == (
        "http://ls/storage-data/uploaded/?filepath=upload/7/video.mp4"
    )
    assert registry.sources[0][1]["Authorization"] == "Token token"
    assert registry.sources[1] == ("/cached/task.mp4", None)


def test_external_upload_looking_url_does_not_get_ls_auth(monkeypatch):
    model = _import_model()
    monkeypatch.setattr(model, "STREAM_LS_UPLOADED_FILES", True)
    backend = model.SamVideoInteractive.__new__(model.SamVideoInteractive)
    backend._ls_host_token = lambda: ("http://ls", "token")

    source, headers = backend._resolve_video_source(
        "https://evil.example/data/upload/7/video.mp4", "task"
    )

    assert source == "https://evil.example/data/upload/7/video.mp4"
    assert headers is None
