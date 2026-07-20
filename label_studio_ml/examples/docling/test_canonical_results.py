"""Unit-level tests for ``docling_to_ls_results``.

These tests use lightweight stubs in place of the real ``DoclingDocument`` so
they run without IBM Docling SaaS access. They assert the *result envelope
shape* only — that is exactly the contract that
``docling-ls-implementation/docling_interface.jsx`` ``parseResults`` reads.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from docling_core.types.doc.document import ContentLayer
from docling_core.types.doc.labels import DocItemLabel, GraphLinkLabel

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
    self_ref: Optional[str] = None,
    data: Any = None,
    captions: Optional[List[Any]] = None,
    footnotes: Optional[List[Any]] = None,
    graph: Any = None,
) -> SimpleNamespace:
    prov = SimpleNamespace(page_no=page_no, bbox=bbox)
    return SimpleNamespace(
        prov=[prov],
        label=label,
        text=text,
        content_layer=content_layer,
        meta=None,
        self_ref=self_ref,
        data=data,
        captions=captions or [],
        footnotes=footnotes or [],
        graph=graph,
    )


def _fake_table_cell(
    *,
    bbox: Optional[_FakeBBox],
    text: str = "",
    column_header: bool = False,
    row_header: bool = False,
    row_section: bool = False,
    row_span: int = 1,
    col_span: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        bbox=bbox,
        text=text,
        column_header=column_header,
        row_header=row_header,
        row_section=row_section,
        row_span=row_span,
        col_span=col_span,
    )


class _FakeRef:
    """Minimal stand-in for ``docling_core.types.doc.document.RefItem``.

    Resolves through a ``cref -> item`` map wired up by the fake doc, so
    tests don't need to mimic ``/tables/0`` JSON-pointer indirection.
    """

    def __init__(self, cref: str):
        self.cref = cref
        self._doc: Optional["_FakeDoc"] = None

    def resolve(self, doc: "_FakeDoc") -> Any:
        return doc._refs[self.cref]


class _FakeDoc:
    def __init__(
        self,
        items_with_levels: List[Tuple[Any, int]],
        *,
        tables: Optional[List[Any]] = None,
        pictures: Optional[List[Any]] = None,
        key_value_items: Optional[List[Any]] = None,
        form_items: Optional[List[Any]] = None,
    ):
        self._items = items_with_levels
        self.pages: Dict[int, Any] = {1: _fake_page(100.0, 100.0)}
        self.tables = tables or []
        self.pictures = pictures or []
        self.key_value_items = key_value_items or []
        self.form_items = form_items or []
        # cref -> resolved item, populated from every stub that has a self_ref.
        self._refs: Dict[str, Any] = {}
        for it, _ in self._items:
            ref = getattr(it, "self_ref", None)
            if isinstance(ref, str) and ref:
                self._refs[ref] = it
        for coll in (self.tables, self.pictures, self.key_value_items, self.form_items):
            for it in coll:
                ref = getattr(it, "self_ref", None)
                if isinstance(ref, str) and ref:
                    self._refs[ref] = it

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

    # Reading order needs 2+ items to emit a polyline; with a single item, the
    # output stays as one rect regardless of the (now default-on) include_reading_order.
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

    out = docling_document_to_ls_results(doc, reading_order_level=1)
    # Two rectangles + one polyline. Reading order is on by default now, so no
    # need for the caller to opt in — matches what the interface expects to render.
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


def test_reading_order_opt_out_disables_polyline() -> None:
    """Callers that don't want a reading-order polyline should still be able to opt out."""
    items = [
        (_fake_item(label=DocItemLabel.TEXT, bbox=_FakeBBox(0, 0, 10, 10)), 1),
        (_fake_item(label=DocItemLabel.TEXT, bbox=_FakeBBox(40, 40, 60, 60)), 1),
    ]
    doc = _FakeDoc(items)
    out = docling_document_to_ls_results(doc, include_reading_order=False)
    types = [r["type"] for r in out]
    assert "polygonlabels" not in types


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


def test_table_structure_emits_child_cells_with_parent_id() -> None:
    """A TableItem with cells produces one rect per cell, parented to the table rect."""
    cells = [
        _fake_table_cell(bbox=_FakeBBox(10, 10, 30, 20), text="H1", column_header=True),
        _fake_table_cell(bbox=_FakeBBox(30, 10, 50, 20), text="H2", column_header=True),
        _fake_table_cell(bbox=_FakeBBox(10, 20, 30, 30), text="a"),
        _fake_table_cell(bbox=_FakeBBox(30, 20, 50, 30), text="b", row_span=1, col_span=2),
    ]
    table = _fake_item(
        label=DocItemLabel.TABLE,
        bbox=_FakeBBox(10, 10, 50, 30),
        self_ref="#/tables/0",
        data=SimpleNamespace(table_cells=cells),
    )
    doc = _FakeDoc([(table, 1)], tables=[table])

    out = docling_document_to_ls_results(doc)
    rects = [r for r in out if r["type"] == "rectanglelabels"]
    # 1 table + 4 cells.
    assert len(rects) == 5

    table_rect = next(r for r in rects if r["value"]["rectanglelabels"] == ["table"])
    cell_rects = [r for r in rects if r is not table_rect]
    assert all(cr["value"]["parentId"] == table_rect["id"] for cr in cell_rects)

    # Header vs plain-cell vs merged-cell classification comes from cell flags/spans.
    labels = [cr["value"]["rectanglelabels"][0] for cr in cell_rects]
    assert labels.count("column_header") == 2
    assert labels.count("table_cell") == 1
    assert labels.count("table_merged_cell") == 1

    # Cells sit one level deeper than the table in the sub-annotation tree.
    for cr in cell_rects:
        assert cr["value"]["level"] == 2

    # Table cells must NOT be swept into the reading-order polyline (which
    # sequences top-level flow). Only 2+ top-level items would spawn one; here
    # there's just the single table, so no polyline at all.
    assert not any(r["type"] == "polygonlabels" for r in out)


def test_table_structure_opt_out_skips_cells() -> None:
    cells = [_fake_table_cell(bbox=_FakeBBox(10, 10, 30, 20), text="only")]
    table = _fake_item(
        label=DocItemLabel.TABLE,
        bbox=_FakeBBox(10, 10, 50, 30),
        self_ref="#/tables/0",
        data=SimpleNamespace(table_cells=cells),
    )
    doc = _FakeDoc([(table, 1)], tables=[table])
    out = docling_document_to_ls_results(doc, include_table_structure=False)
    assert len(out) == 1
    assert out[0]["value"]["rectanglelabels"] == ["table"]


def test_to_caption_and_to_footnote_polylines_emitted_for_picture() -> None:
    """FloatingItem captions / footnotes should surface as linking polylines."""
    caption = _fake_item(
        label=DocItemLabel.CAPTION,
        bbox=_FakeBBox(10, 60, 40, 65),
        text="Fig. 1: Widget",
        self_ref="#/texts/0",
    )
    footnote = _fake_item(
        label=DocItemLabel.FOOTNOTE,
        bbox=_FakeBBox(10, 65, 40, 70),
        text="* see appendix",
        self_ref="#/texts/1",
    )
    picture = _fake_item(
        label=DocItemLabel.PICTURE,
        bbox=_FakeBBox(10, 10, 40, 50),
        self_ref="#/pictures/0",
        captions=[_FakeRef("#/texts/0")],
        footnotes=[_FakeRef("#/texts/1")],
    )
    doc = _FakeDoc(
        [(picture, 1), (caption, 1), (footnote, 1)],
        pictures=[picture],
    )
    out = docling_document_to_ls_results(doc)

    polys = [r for r in out if r["type"] == "polygonlabels"]
    labels = [p["value"]["polygonlabels"][0] for p in polys]
    assert labels.count("to_caption") == 1
    assert labels.count("to_footnote") == 1
    # Plus one reading_order (3 items >= 2).
    assert labels.count("reading_order") == 1

    picture_id = next(
        r["id"] for r in out
        if r["type"] == "rectanglelabels" and r["value"]["rectanglelabels"] == ["picture"]
    )
    caption_id = next(
        r["id"] for r in out
        if r["type"] == "rectanglelabels" and r["value"]["rectanglelabels"] == ["caption"]
    )
    footnote_id = next(
        r["id"] for r in out
        if r["type"] == "rectanglelabels" and r["value"]["rectanglelabels"] == ["footnote"]
    )
    to_cap = next(p for p in polys if p["value"]["polygonlabels"] == ["to_caption"])
    to_foot = next(p for p in polys if p["value"]["polygonlabels"] == ["to_footnote"])
    assert to_cap["value"]["connectedRegions"] == [picture_id, caption_id]
    assert to_foot["value"]["connectedRegions"] == [picture_id, footnote_id]
    assert len(to_cap["value"]["points"]) == 2
    assert to_cap["value"]["closed"] is False


def test_to_value_polyline_from_kv_graph_link() -> None:
    """KeyValueItem.graph.links[TO_VALUE] should produce a key -> value polyline."""
    key_item = _fake_item(
        label=DocItemLabel.FIELD_KEY,
        bbox=_FakeBBox(10, 10, 30, 20),
        text="Name",
        self_ref="#/texts/0",
    )
    value_item = _fake_item(
        label=DocItemLabel.FIELD_VALUE,
        bbox=_FakeBBox(35, 10, 60, 20),
        text="Alice",
        self_ref="#/texts/1",
    )
    key_cell = SimpleNamespace(cell_id=1, item_ref=_FakeRef("#/texts/0"))
    value_cell = SimpleNamespace(cell_id=2, item_ref=_FakeRef("#/texts/1"))
    link = SimpleNamespace(
        label=GraphLinkLabel.TO_VALUE, source_cell_id=1, target_cell_id=2
    )
    graph = SimpleNamespace(cells=[key_cell, value_cell], links=[link])
    kv = _fake_item(
        label=DocItemLabel.KEY_VALUE_REGION,
        bbox=_FakeBBox(10, 10, 60, 20),
        self_ref="#/key_value_items/0",
        graph=graph,
    )
    doc = _FakeDoc(
        [(kv, 1), (key_item, 2), (value_item, 2)],
        key_value_items=[kv],
    )
    out = docling_document_to_ls_results(doc)

    polys = [r for r in out if r["type"] == "polygonlabels"]
    to_val = [p for p in polys if p["value"]["polygonlabels"] == ["to_value"]]
    assert len(to_val) == 1
    key_id = next(
        r["id"] for r in out
        if r["type"] == "rectanglelabels" and r["value"]["rectanglelabels"] == ["key"]
    )
    value_id = next(
        r["id"] for r in out
        if r["type"] == "rectanglelabels" and r["value"]["rectanglelabels"] == ["value"]
    )
    assert to_val[0]["value"]["connectedRegions"] == [key_id, value_id]


def test_relations_dropped_when_endpoint_not_emitted() -> None:
    """A caption ref pointing at an item filtered out (e.g. content-layer) is silently dropped."""
    # Caption item is missing from the iteration list, so no rect exists for it — the
    # relation emitter should skip the dangling link instead of writing a polyline
    # that connects to nothing.
    picture = _fake_item(
        label=DocItemLabel.PICTURE,
        bbox=_FakeBBox(10, 10, 40, 50),
        self_ref="#/pictures/0",
        captions=[_FakeRef("#/texts/0")],
    )
    doc = _FakeDoc([(picture, 1)], pictures=[picture])
    # Deliberately do NOT register the caption item — resolve() would still
    # succeed on the ref if we added it to _refs, so leave the _refs map empty
    # for that cref.
    out = docling_document_to_ls_results(doc)
    polys = [r for r in out if r["type"] == "polygonlabels"]
    assert not any(p["value"]["polygonlabels"] == ["to_caption"] for p in polys)


def test_relations_opt_out_disables_all_relation_polylines() -> None:
    caption = _fake_item(
        label=DocItemLabel.CAPTION,
        bbox=_FakeBBox(10, 60, 40, 65),
        self_ref="#/texts/0",
    )
    picture = _fake_item(
        label=DocItemLabel.PICTURE,
        bbox=_FakeBBox(10, 10, 40, 50),
        self_ref="#/pictures/0",
        captions=[_FakeRef("#/texts/0")],
    )
    doc = _FakeDoc([(picture, 1), (caption, 1)], pictures=[picture])
    out = docling_document_to_ls_results(doc, include_relations=False)
    labels = [
        r["value"]["polygonlabels"][0]
        for r in out
        if r["type"] == "polygonlabels"
    ]
    assert "to_caption" not in labels
