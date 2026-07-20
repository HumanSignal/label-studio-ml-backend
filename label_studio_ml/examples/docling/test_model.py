"""Tests for the Docling model wiring: task file resolution and prediction assembly."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest
from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import ContentLayer
from docling_core.types.doc.labels import DocItemLabel
from label_studio_ml.utils import DATA_UNDEFINED_NAME

import model as model_mod
from model import Docling


def _url(data, **env):
    m = Docling.__new__(Docling)  # skip __init__: it builds SDK/label-config state we do not need
    m._data_key = env.get("data_key", "image")
    return Docling._task_file_url(m, {"id": 1, "data": data})


def test_prefers_the_configured_data_key() -> None:
    assert _url({"image": "https://host/a.png", "pdf": "https://host/b.pdf"}) == "https://host/a.png"
    assert (
        _url({"image": "https://host/a.png", "pdf": "https://host/b.pdf"}, data_key="pdf")
        == "https://host/b.pdf"
    )


def test_falls_back_through_the_chain() -> None:
    assert _url({"ocr": "https://host/c.tif"}) == "https://host/c.tif"
    assert _url({DATA_UNDEFINED_NAME: "https://host/d.pdf"}) == "https://host/d.pdf"
    assert _url({"undefined": "https://host/e.pdf"}) == "https://host/e.pdf"


def test_unwraps_a_dict_with_a_url_field() -> None:
    assert _url({"image": {"url": "https://host/f.png"}}) == "https://host/f.png"
    assert _url({"image": {"URL": "https://host/g.png"}}) == "https://host/g.png"


def test_dict_without_a_url_is_skipped_rather_than_stringified() -> None:
    """Regression: the $undefined$ key is also DATA_UNDEFINED_NAME, and a second
    lookup used to bypass the dict handling and return str(dict) as the URL."""
    assert (
        _url({DATA_UNDEFINED_NAME: {"path": "no-url-here"}, "asset": "https://host/good.pdf"})
        == "https://host/good.pdf"
    )
    # And with nothing else to fall back to, we report "no URL" instead of a bogus one.
    assert _url({DATA_UNDEFINED_NAME: {"path": "no-url-here"}}) is None


def test_last_resort_scan_finds_file_shaped_values() -> None:
    assert _url({"attachment": "/storage-data/uploaded/1/x.pdf"}) == "/storage-data/uploaded/1/x.pdf"
    assert _url({"attachment": "s3://bucket/key.pdf"}) == "s3://bucket/key.pdf"
    # Unrelated string fields are not mistaken for files.
    assert _url({"title": "some text", "n": 5}) is None


def test_no_data_returns_none() -> None:
    assert _url({}) is None


# --- prediction assembly -------------------------------------------------------------
#
# These drive predict_single with the Docling client stubbed out, in both of the modes
# that decide where a result's original_width/original_height come from: remote-URL-only
# (no file on disk) and the default download path (file on disk, which real PDF tasks take).


def _page(width: float, height: float) -> SimpleNamespace:
    return SimpleNamespace(size=Size(width=width, height=height), image=None)


def _doc(pages: Optional[Dict[int, Any]], item_page_no: int = 1) -> SimpleNamespace:
    item = SimpleNamespace(
        prov=[
            SimpleNamespace(
                page_no=item_page_no,
                bbox=BoundingBox(l=10, t=10, r=30, b=30, coord_origin=CoordOrigin.TOPLEFT),
            )
        ],
        label=DocItemLabel.TEXT,
        text="hi",
        content_layer=ContentLayer.BODY,
        meta=None,
    )
    return SimpleNamespace(
        pages=pages or {},
        iterate_items=lambda **_kw: iter([(item, 1)]),
    )


_DOCLING_ENV = (
    "DOCLING_CONVERT_REMOTE_URL_ONLY",
    "DOCLING_CONVERT_SOURCE_HEADERS_JSON",
    "DOCLING_PAGE_NO",
    "DOCLING_PREDICT_READING_ORDER",
    "DOCLING_READING_ORDER_LEVEL",
    "DOCLING_CONTENT_LAYERS",
)


def _stub_convert(monkeypatch, doc) -> None:
    monkeypatch.setattr(Docling, "_ensure_client", lambda self: object())
    monkeypatch.setattr(
        Docling,
        "_convert_with_service",
        lambda self, client, *, source, headers=None: SimpleNamespace(
            status=ConversionStatus.SUCCESS, document=doc
        ),
    )


def _model(monkeypatch, doc, **env) -> Docling:
    # Start from a known env: an ambient DOCLING_PAGE_NO would otherwise filter out the
    # test's item and make these tests pass or fail for reasons they are not about.
    for name in _DOCLING_ENV:
        monkeypatch.delenv(name, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _stub_convert(monkeypatch, doc)
    m = Docling.__new__(Docling)  # skip __init__: it builds SDK/label-config state we do not need
    m._data_key, m._from_name, m._to_name = "image", "docling", "docling"
    return m


def _predict(monkeypatch, doc, **env) -> Optional[Any]:
    """Remote-URL-only mode: no file is downloaded, so `path` stays None."""
    env.setdefault("DOCLING_CONVERT_REMOTE_URL_ONLY", "true")
    m = _model(monkeypatch, doc, **env)
    return Docling.predict_single(m, {"id": 1, "data": {"image": "https://host/doc.pdf"}})


def _predict_downloaded(monkeypatch, tmp_path, doc, **env) -> Optional[Any]:
    """Default mode: the task file is downloaded, so `path` is set and the old code would
    probe it with get_image_size. This is the path real PDF tasks take."""
    m = _model(monkeypatch, doc, **env)
    m.MODEL_DIR = str(tmp_path)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 not a real pdf")
    monkeypatch.setattr(Docling, "get_label_studio_access_token", lambda self: "token")
    monkeypatch.setattr(model_mod, "ls_sdk_get_local_path", lambda *a, **kw: str(pdf))
    return Docling.predict_single(m, {"id": 1, "data": {"image": "https://host/doc.pdf"}})


def test_original_size_comes_from_the_docling_page_raster(monkeypatch) -> None:
    """The task is a PDF, which get_image_size() cannot open. The dimensions must still be
    real, because LS-native consumers do x * original_width / 100 to recover pixels."""
    pred = _predict(monkeypatch, _doc({1: _page(612.0, 792.0)}))

    assert pred.result, "expected at least one region"
    for r in pred.result:
        assert (r["original_width"], r["original_height"]) == (612, 792)
        assert (r["original_width"], r["original_height"]) != (100, 100), "placeholder leaked"


def test_downloaded_pdf_uses_the_raster_and_never_probes_the_file(monkeypatch, tmp_path) -> None:
    """The regression this guards: a downloaded PDF used to be handed to get_image_size(),
    which cannot open a PDF, and the bare except turned that into original_width=100.
    The raster must win over probing the file, not merely be used when there is no file."""
    monkeypatch.setattr(
        model_mod,
        "get_image_size",
        lambda _p: pytest.fail("get_image_size must not be probed when a page raster exists"),
    )
    pred = _predict_downloaded(monkeypatch, tmp_path, _doc({1: _page(612.0, 792.0)}))

    assert pred.result, "expected at least one region"
    for r in pred.result:
        assert (r["original_width"], r["original_height"]) == (612, 792)


def test_downloaded_image_falls_back_to_probing_when_there_is_no_raster(monkeypatch, tmp_path) -> None:
    """Without a page raster the downloaded file is still worth probing — that path only
    works for real images, but it beats the placeholder."""
    monkeypatch.setattr(model_mod, "page_raster_size", lambda _doc, _page_no=None: None)
    monkeypatch.setattr(model_mod, "get_image_size", lambda _p: (1024, 768))
    pred = _predict_downloaded(monkeypatch, tmp_path, _doc({1: _page(612.0, 792.0)}))

    assert pred.result
    for r in pred.result:
        assert (r["original_width"], r["original_height"]) == (1024, 768)


def test_original_size_follows_docling_page_no(monkeypatch) -> None:
    """DOCLING_PAGE_NO must reach page_raster_size, or a page-2 prediction reports page 1's raster."""
    pages = {1: _page(100.0, 100.0), 2: _page(300.0, 400.0)}
    pred = _predict(monkeypatch, _doc(pages, item_page_no=2), DOCLING_PAGE_NO="2")

    assert pred.result
    for r in pred.result:
        assert (r["original_width"], r["original_height"]) == (300, 400)


def test_original_size_falls_back_to_placeholder_without_a_raster(monkeypatch) -> None:
    """No pages and no local file: emit the placeholder rather than crash. Percent
    coordinates stay correct; only pixel reconstruction is unavailable."""
    # Force page_raster_size to find nothing while the item still resolves.
    monkeypatch.setattr(model_mod, "page_raster_size", lambda _doc, _page_no=None: None)
    pred = _predict(monkeypatch, _doc({1: _page(100.0, 100.0)}))

    assert pred.result
    for r in pred.result:
        assert (r["original_width"], r["original_height"]) == (100, 100)
