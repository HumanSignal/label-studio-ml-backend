"""Unit-level tests for ``docling_to_ls_results``.

These tests use lightweight stubs in place of the real ``DoclingDocument`` so
they run without IBM Docling SaaS access. They assert the *result envelope
shape* only — that is exactly the contract that
``docling-ls-implementation/docling_interface.jsx`` ``parseResults`` reads.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from docling_core.types.doc.document import ContentLayer
from docling_core.types.doc.labels import DocItemLabel

from docling_to_ls_results import docling_document_to_ls_results


class _FakeBBox:
    def __init__(self, l: float, t: float, r: float, b: float):
        self.l = l
        self.t = t
        self.r = r
        self.b = b
        self.width = r - l
        self.height = b - t

    def to_top_left_origin(self, page_height: float) -> "_FakeBBox":
        # Pretend our bbox is already top-left; production code calls this regardless.
        return self

    def scale_to_size(self, *, old_size, new_size) -> "_FakeBBox":
        # Scale unchanged — keep coordinates simple for the test.
        sx = new_size.width / old_size.width
        sy = new_size.height / old_size.height
        return _FakeBBox(self.l * sx, self.t * sy, self.r * sx, self.b * sy)


def _fake_size(w: float, h: float) -> SimpleNamespace:
    return SimpleNamespace(width=w, height=h)


def _fake_page(size_w: float = 100.0, size_h: float = 100.0) -> SimpleNamespace:
    return SimpleNamespace(size=_fake_size(size_w, size_h), image=None)


def _fake_item(
    *,
    label: DocItemLabel,
    bbox: _FakeBBox,
    page_no: int = 1,
    text: str = "",
    content_layer: ContentLayer = ContentLayer.BODY,
) -> SimpleNamespace:
    prov = SimpleNamespace(page_no=page_no, bbox=bbox)
    return SimpleNamespace(
        prov=[prov],
        label=label,
        text=text,
        content_layer=content_layer,
        meta=None,
    )


class _FakeDoc:
    def __init__(self, items_with_levels: List[Tuple[Any, int]]):
        self._items = items_with_levels
        self.pages: Dict[int, Any] = {1: _fake_page(100.0, 100.0)}

    def iterate_items(self, **_kwargs):
        for item, level in self._items:
            yield item, level


def test_rectanglelabels_envelope_shape() -> None:
    item = _fake_item(
        label=DocItemLabel.SECTION_HEADER,
        bbox=_FakeBBox(10, 20, 30, 40),
        text="Hello",
    )
    doc = _FakeDoc([(item, 1)])

    out = docling_document_to_ls_results(doc, from_name="docling", to_name="docling")
    assert len(out) == 1
    r = out[0]
    assert r["type"] == "rectanglelabels"
    assert r["from_name"] == "docling"
    assert r["to_name"] == "docling"
    assert r["origin"] == "prediction"
    assert "id" in r and isinstance(r["id"], str)

    v = r["value"]
    # Coordinates are in percent of the page raster.
    assert v["x"] == 10.0
    assert v["y"] == 20.0
    assert v["width"] == 20.0
    assert v["height"] == 20.0
    assert v["rotation"] == 0
    assert v["rectanglelabels"] == ["section_header"]
    assert v["content_layer"] == "BODY"
    assert v["level"] == 1
    assert v["picture_type"] is None
    assert v["text"] == "Hello"
    assert v["parentId"] is None


def test_reading_order_polyline_envelope_shape() -> None:
    items = [
        (
            _fake_item(label=DocItemLabel.TEXT, bbox=_FakeBBox(0, 0, 10, 10), text="a"),
            1,
        ),
        (
            _fake_item(label=DocItemLabel.TEXT, bbox=_FakeBBox(40, 40, 60, 60), text="b"),
            1,
        ),
    ]
    doc = _FakeDoc(items)

    out = docling_document_to_ls_results(
        doc, include_reading_order=True, reading_order_level=1
    )
    # Two rectangles + one polyline.
    types = [r["type"] for r in out]
    assert types.count("rectanglelabels") == 2
    assert types.count("polygonlabels") == 1

    poly = next(r for r in out if r["type"] == "polygonlabels")
    v = poly["value"]
    assert v["polygonlabels"] == ["reading_order"]
    assert v["closed"] is False
    assert isinstance(v["points"], list) and len(v["points"]) == 2
    # connectedRegions references the rectangle ids.
    rect_ids = [r["id"] for r in out if r["type"] == "rectanglelabels"]
    assert v["connectedRegions"] == rect_ids
    assert v["level"] == 1
    assert v["validationErrors"] == []
    assert v["parentId"] is None


def test_no_underscore_prefixed_keys_in_value() -> None:
    """The interface's spatial-region serialization validator rejects
    ``value`` payloads that leak in-memory underscore-prefixed keys."""
    item = _fake_item(label=DocItemLabel.TEXT, bbox=_FakeBBox(0, 0, 10, 10))
    doc = _FakeDoc([(item, 1)])
    out = docling_document_to_ls_results(doc)
    assert out, "expected at least one result"
    for r in out:
        for k in (r.get("value") or {}).keys():
            assert not k.startswith("_"), f"underscore-prefixed key {k!r} leaked into value"
