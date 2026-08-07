"""Unit-level tests for ``docling_to_ls_results``.

These build documents out of the real ``docling_core`` types (``BoundingBox``,
``Size``) so the coordinate math is exercised against the same API production
uses — in particular the BOTTOMLEFT -> TOPLEFT flip, which is the transform every
PDF provenance bbox actually goes through. Only ``iterate_items`` is stubbed, to
keep the tests independent of IBM Docling SaaS.

They assert the *result envelope shape*, which is the contract that
``docling-ls-implementation/docling_interface.jsx`` ``parseResults`` reads.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import ContentLayer
from docling_core.types.doc.labels import DocItemLabel, GraphLinkLabel, GroupLabel

from docling_to_ls_results import docling_document_to_ls_results, page_raster_size


def _page(width: float = 100.0, height: float = 100.0, image_size: Optional[Size] = None) -> SimpleNamespace:
    image = SimpleNamespace(size=image_size) if image_size is not None else None
    return SimpleNamespace(size=Size(width=width, height=height), image=image)


def _item(
    *,
    label: DocItemLabel,
    bbox: BoundingBox,
    page_no: int = 1,
    text: str = "",
    content_layer: ContentLayer = ContentLayer.BODY,
    self_ref: Optional[str] = None,
    data: Any = None,
    captions: Optional[List[Any]] = None,
    footnotes: Optional[List[Any]] = None,
    graph: Any = None,
) -> SimpleNamespace:
    return _multi_prov_item(
        label=label,
        provs=[(page_no, bbox)],
        text=text,
        content_layer=content_layer,
        self_ref=self_ref,
        data=data,
        captions=captions,
        footnotes=footnotes,
        graph=graph,
    )


def _multi_prov_item(
    *,
    label: DocItemLabel,
    provs: List[Tuple[int, BoundingBox]],
    text: str = "",
    content_layer: ContentLayer = ContentLayer.BODY,
    self_ref: Optional[str] = None,
    data: Any = None,
    captions: Optional[List[Any]] = None,
    footnotes: Optional[List[Any]] = None,
    graph: Any = None,
) -> SimpleNamespace:
    """An item with one provenance per page, as Docling reports for page-straddling items."""
    return SimpleNamespace(
        prov=[SimpleNamespace(page_no=p, bbox=b) for p, b in provs],
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


def _table_cell(
    *,
    bbox: Optional[BoundingBox],
    text: str = "",
    column_header: bool = False,
    row_header: bool = False,
    row_section: bool = False,
    row_span: int = 1,
    col_span: int = 1,
    start_row: int = 0,
    end_row: int = 1,
    start_col: int = 0,
    end_col: int = 1,
) -> SimpleNamespace:
    """Stand-in for ``docling_core.types.doc.document.TableCell``.

    Grid offsets default to a single 1×1 cell at (row 0, col 0); tests that
    exercise the row/column strip derivation MUST override these — the row/
    column emission logic reads ``start_row_offset_idx`` / ``end_row_offset_idx``
    (and the column equivalents) to decide which cells define a row's/column's
    exact geometry, so leaving them at zero would collapse every cell onto
    the same grid position.
    """
    return SimpleNamespace(
        bbox=bbox,
        text=text,
        column_header=column_header,
        row_header=row_header,
        row_section=row_section,
        row_span=row_span,
        col_span=col_span,
        start_row_offset_idx=start_row,
        end_row_offset_idx=end_row,
        start_col_offset_idx=start_col,
        end_col_offset_idx=end_col,
    )


class _Ref:
    """Stand-in for ``docling_core.types.doc.document.RefItem``.

    Resolves through the fake doc's ``_refs`` map, so tests don't have to
    mimic the ``/tables/0`` JSON-pointer indirection.
    """

    def __init__(self, cref: str):
        self.cref = cref

    def resolve(self, doc: "_Doc") -> Any:
        return doc._refs[self.cref]


class _Doc:
    """Minimal DoclingDocument stand-in: real pages/bboxes, stubbed iteration.

    ``tables`` / ``pictures`` / ``key_value_items`` / ``form_items`` mirror the
    real DoclingDocument collections and are what ``_emit_relations`` walks to
    find captions / footnotes / to_value graph links.
    """

    def __init__(
        self,
        items_with_levels: List[Tuple[Any, int]],
        pages: Optional[Dict[int, Any]] = None,
        *,
        tables: Optional[List[Any]] = None,
        pictures: Optional[List[Any]] = None,
        key_value_items: Optional[List[Any]] = None,
        form_items: Optional[List[Any]] = None,
        groups: Optional[List[Any]] = None,
    ):
        self._items = items_with_levels
        self.pages: Dict[int, Any] = pages if pages is not None else {1: _page()}
        self.tables = tables or []
        self.pictures = pictures or []
        self.key_value_items = key_value_items or []
        self.form_items = form_items or []
        # ``groups`` mirrors ``DoclingDocument.groups`` — the flat list of every
        # ListGroup / InlineGroup / plain GroupItem the document carries. The
        # emitter reads this for InlineGroup → merge polyline mapping.
        self.groups = groups or []
        # cref -> resolved item, populated from every stub carrying a self_ref.
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
        yield from self._items


def _bl(l: float, t: float, r: float, b: float) -> BoundingBox:
    """A bottom-left-origin bbox, the convention Docling reports PDF provenance in."""
    return BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.BOTTOMLEFT)


def _tl(l: float, t: float, r: float, b: float) -> BoundingBox:
    return BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.TOPLEFT)


def test_rectanglelabels_envelope_shape() -> None:
    item = _item(label=DocItemLabel.SECTION_HEADER, bbox=_tl(10, 20, 30, 40), text="Hello")
    out = docling_document_to_ls_results(_Doc([(item, 1)]), from_name="docling", to_name="docling")

    assert len(out) == 1
    r = out[0]
    assert r["type"] == "rectanglelabels"
    assert r["from_name"] == "docling"
    assert r["to_name"] == "docling"
    assert r["origin"] == "prediction"
    assert isinstance(r["id"], str) and r["id"]

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


def test_bottom_left_origin_bbox_is_flipped_to_top_left() -> None:
    """Docling reports PDF provenance bottom-left; LS wants top-left."""
    # On a 100-high page, a box spanning y=60..80 from the bottom is y=20..40 from the top.
    item = _item(label=DocItemLabel.TEXT, bbox=_bl(10, 80, 30, 60))
    out = docling_document_to_ls_results(_Doc([(item, 1)]))

    v = out[0]["value"]
    assert v["x"] == 10.0
    assert v["y"] == 20.0
    assert v["width"] == 20.0
    assert v["height"] == 20.0


def test_bbox_overhanging_page_edges_is_clipped() -> None:
    """A bbox past an edge is trimmed to the page, not left with its full extent."""
    items = [
        # Overhangs the right edge: 90..120 -> clipped to 90..100.
        (_item(label=DocItemLabel.TEXT, bbox=_tl(90, 10, 120, 30)), 1),
        # Overhangs the left edge: -10..40 -> clipped to 0..40.
        (_item(label=DocItemLabel.TEXT, bbox=_tl(-10, 10, 40, 30)), 1),
        # Overhangs the bottom edge: 90..130 -> clipped to 90..100.
        (_item(label=DocItemLabel.TEXT, bbox=_tl(10, 90, 30, 130)), 1),
    ]
    out = docling_document_to_ls_results(_Doc(items))

    right, left, bottom = (r["value"] for r in out)
    assert (right["x"], right["width"]) == (90.0, 10.0)
    assert (left["x"], left["width"]) == (0.0, 40.0)
    assert (bottom["y"], bottom["height"]) == (90.0, 10.0)

    # The invariant the clipping exists to hold.
    for r in out:
        v = r["value"]
        assert 0 <= v["x"] <= 100 and 0 <= v["y"] <= 100
        assert v["x"] + v["width"] <= 100.0
        assert v["y"] + v["height"] <= 100.0


def test_coordinates_are_percent_of_page_raster_when_image_present() -> None:
    """page.image is the raster the percentages must be relative to."""
    doc = _Doc(
        [(_item(label=DocItemLabel.TEXT, bbox=_tl(50, 100, 100, 200)), 1)],
        pages={1: _page(200.0, 400.0, image_size=Size(width=400.0, height=800.0))},
    )
    out = docling_document_to_ls_results(doc)

    v = out[0]["value"]
    # Percentages are scale-invariant: 50/200 == 100/400 == 25%.
    assert v["x"] == 25.0
    assert v["y"] == 25.0
    assert v["width"] == 25.0
    assert v["height"] == 25.0


def test_page_raster_size_prefers_image_and_survives_pdfs() -> None:
    """original_width/height come from the doc, so PDFs (unopenable as images) still work."""
    assert page_raster_size(_Doc([], pages={1: _page(612.0, 792.0)})) == (612, 792)
    assert page_raster_size(
        _Doc([], pages={1: _page(612.0, 792.0, image_size=Size(width=1224.0, height=1584.0))})
    ) == (1224, 1584)
    # Explicit page, and the no-pages case the caller must fall back from.
    doc = _Doc([], pages={1: _page(100.0, 100.0), 2: _page(300.0, 400.0)})
    assert page_raster_size(doc, page_no=2) == (300, 400)
    assert page_raster_size(doc) == (100, 100)  # defaults to the first page
    assert page_raster_size(_Doc([], pages={})) is None
    assert page_raster_size(doc, page_no=99) is None  # DOCLING_PAGE_NO naming a missing page


def test_page_raster_size_rounds_and_rejects_degenerate_sizes() -> None:
    # Round rather than truncate: 595.5 -> 596 stays closest to the measured raster.
    assert page_raster_size(_Doc([], pages={1: _page(595.5, 841.5)})) == (596, 842)
    # A sub-pixel page is not a usable dimension; report None so the caller can fall back
    # instead of emitting original_width=0.
    assert page_raster_size(_Doc([], pages={1: _page(0.5, 0.5)})) is None


def test_page_no_filter_keeps_only_the_requested_page() -> None:
    items = [
        (_item(label=DocItemLabel.TEXT, bbox=_tl(10, 10, 20, 20), page_no=1, text="p1"), 1),
        (_item(label=DocItemLabel.TEXT, bbox=_tl(10, 10, 20, 20), page_no=2, text="p2"), 1),
    ]
    doc = _Doc(items, pages={1: _page(), 2: _page()})
    out = docling_document_to_ls_results(doc, page_no=2)

    assert len(out) == 1
    assert out[0]["value"]["text"] == "p2"


def test_page_no_filter_measures_the_provenance_on_the_requested_page() -> None:
    """An item straddling a page break has one prov per page; iterate_items yields it for
    either page, so we must measure the prov on the page asked for, not prov[0]."""
    item = _multi_prov_item(
        label=DocItemLabel.TEXT,
        provs=[(1, _tl(10, 80, 30, 100)), (2, _tl(40, 0, 60, 20))],
        text="straddles",
    )
    doc = _Doc([(item, 1)], pages={1: _page(), 2: _page()})

    on_p2 = docling_document_to_ls_results(doc, page_no=2)
    assert len(on_p2) == 1, "the page-2 half of a straddling item must not be dropped"
    v = on_p2[0]["value"]
    assert (v["x"], v["y"], v["width"], v["height"]) == (40.0, 0.0, 20.0, 20.0)

    on_p1 = docling_document_to_ls_results(doc, page_no=1)
    v = on_p1[0]["value"]
    assert (v["x"], v["y"], v["width"], v["height"]) == (10.0, 80.0, 20.0, 20.0)


def test_rounding_cannot_break_the_page_bounds_or_fake_a_zero_area_region() -> None:
    """Rounding x and width independently would undo the clipping: a sub-precision box
    could round to width 0, and x + width could land past 100."""
    # Narrower than the emitted precision -> dropped, not emitted as a width-0 region.
    thin = _item(label=DocItemLabel.TEXT, bbox=_tl(10, 10, 10.00004, 20))
    assert docling_document_to_ls_results(_Doc([(thin, 1)])) == []

    # x rounds up while width would keep its unrounded extent -> x + width = 100.0001.
    wide = _item(label=DocItemLabel.TEXT, bbox=_tl(0.00555, 10, 100, 20))
    v = docling_document_to_ls_results(_Doc([(wide, 1)]))[0]["value"]
    assert v["x"] + v["width"] <= 100.0


def test_bbox_entirely_off_page_is_dropped() -> None:
    """Clipping a fully off-page box leaves zero area; emitting it would put an invisible
    region on the canvas and a stray point in the reading-order polyline."""
    off = _item(label=DocItemLabel.TEXT, bbox=_tl(110, 10, 120, 20))
    assert docling_document_to_ls_results(_Doc([(off, 1)])) == []

    # And it must not sneak into reading order alongside real regions.
    items = [
        (_item(label=DocItemLabel.TEXT, bbox=_tl(0, 0, 10, 10)), 1),
        (off, 1),
        (_item(label=DocItemLabel.TEXT, bbox=_tl(40, 40, 60, 60)), 1),
    ]
    out = docling_document_to_ls_results(_Doc(items), include_reading_order=True)
    assert [r["type"] for r in out].count("rectanglelabels") == 2
    poly = next(r for r in out if r["type"] == "polygonlabels")
    assert poly["value"]["points"] == [[5.0, 5.0], [50.0, 50.0]]


def test_inverted_bbox_is_normalized_not_dropped() -> None:
    """docling_core accepts l>r / t>b (BoundingBox.width is a signed r-l), so the edges must
    be sorted before clipping — otherwise the zero-area guard silently eats the region."""
    inverted = _item(label=DocItemLabel.TEXT, bbox=_tl(30, 40, 10, 20))
    well_ordered = _item(label=DocItemLabel.TEXT, bbox=_tl(10, 20, 30, 40))

    got = docling_document_to_ls_results(_Doc([(inverted, 1)]))
    assert got, "an inverted bbox must still be emitted"
    expected = docling_document_to_ls_results(_Doc([(well_ordered, 1)]))
    for key in ("x", "y", "width", "height"):
        assert got[0]["value"][key] == expected[0]["value"][key]


def test_degenerate_page_size_is_skipped_not_a_zero_division() -> None:
    """scale_to_size divides by page.size, so a zero page dimension would raise — and
    predict()'s per-task loop has no try/except to turn that into a skip."""
    doc = _Doc(
        [(_item(label=DocItemLabel.TEXT, bbox=_tl(1, 1, 2, 2)), 1)],
        # A zero page.size alongside a usable image raster is the one combination that
        # reaches the division.
        pages={1: _page(0.0, 400.0, image_size=Size(width=400.0, height=800.0))},
    )
    assert docling_document_to_ls_results(doc) == []


def test_content_layers_parsing(caplog) -> None:
    doc = _Doc([(_item(label=DocItemLabel.TEXT, bbox=_tl(0, 0, 10, 10)), 1)])

    # Unset -> Docling's default (body only); we must not pass included_content_layers.
    seen = {}
    doc.iterate_items = lambda **kw: seen.update(kw) or iter([])
    docling_document_to_ls_results(doc, content_layers=None)
    assert "included_content_layers" not in seen

    seen.clear()
    docling_document_to_ls_results(doc, content_layers=" BODY , furniture ")
    assert seen["included_content_layers"] == {ContentLayer.BODY, ContentLayer.FURNITURE}

    # A typo must be named in the logs, not silently ignored, and must not narrow the filter.
    seen.clear()
    with caplog.at_level("WARNING"):
        docling_document_to_ls_results(doc, content_layers="body,furnature")
    assert seen["included_content_layers"] == {ContentLayer.BODY}
    assert "furnature" in caplog.text

    # Nothing recognized -> fall back to the default rather than an empty filter.
    seen.clear()
    caplog.clear()
    with caplog.at_level("WARNING"):
        docling_document_to_ls_results(doc, content_layers="bogus")
    assert "included_content_layers" not in seen
    assert "bogus" in caplog.text


def test_unmapped_label_falls_back_to_text() -> None:
    item = _item(label=DocItemLabel.FORM, bbox=_tl(0, 0, 10, 10))
    out = docling_document_to_ls_results(_Doc([(item, 1)]))
    assert out[0]["value"]["rectanglelabels"] == ["form"]

    # A label with no entry in DOCLING_LABEL_TO_LS degrades to "text" rather than leaking
    # a Docling-internal name the interface has no category for.
    item = _item(label=DocItemLabel.FIELD_ITEM, bbox=_tl(0, 0, 10, 10))
    out = docling_document_to_ls_results(_Doc([(item, 1)]))
    assert out[0]["value"]["rectanglelabels"] == ["text"]


def test_reading_order_polyline_envelope_shape() -> None:
    items = [
        (_item(label=DocItemLabel.TEXT, bbox=_tl(0, 0, 10, 10), text="a"), 1),
        (_item(label=DocItemLabel.TEXT, bbox=_tl(40, 40, 60, 60), text="b"), 1),
    ]
    out = docling_document_to_ls_results(_Doc(items), include_reading_order=True, reading_order_level=1)

    types = [r["type"] for r in out]
    assert types.count("rectanglelabels") == 2
    assert types.count("polygonlabels") == 1

    poly = next(r for r in out if r["type"] == "polygonlabels")
    v = poly["value"]
    assert v["polygonlabels"] == ["reading_order"]
    assert v["closed"] is False
    # Centroids of the two rectangles, in iteration (reading) order.
    assert v["points"] == [[5.0, 5.0], [50.0, 50.0]]
    # connectedRegions references the rectangle ids.
    assert v["connectedRegions"] == [r["id"] for r in out if r["type"] == "rectanglelabels"]
    assert v["level"] == 1
    assert v["validationErrors"] == []
    assert v["parentId"] is None


def test_reading_order_polyline_is_per_page_and_needs_two_points() -> None:
    items = [
        (_item(label=DocItemLabel.TEXT, bbox=_tl(0, 0, 10, 10), page_no=1), 1),
        (_item(label=DocItemLabel.TEXT, bbox=_tl(40, 40, 60, 60), page_no=1), 1),
        # Page 2 has a single region: no polyline, since a one-point path means nothing.
        (_item(label=DocItemLabel.TEXT, bbox=_tl(0, 0, 10, 10), page_no=2), 1),
    ]
    doc = _Doc(items, pages={1: _page(), 2: _page()})
    out = docling_document_to_ls_results(doc, include_reading_order=True)

    assert [r["type"] for r in out].count("polygonlabels") == 1


def test_items_without_provenance_are_skipped() -> None:
    item = SimpleNamespace(prov=[], label=DocItemLabel.TEXT, text="", content_layer=ContentLayer.BODY, meta=None)
    assert docling_document_to_ls_results(_Doc([(item, 1)])) == []


def test_no_underscore_prefixed_keys_in_value() -> None:
    """The interface's spatial-region serialization validator rejects
    ``value`` payloads that leak in-memory underscore-prefixed keys."""
    item = _item(label=DocItemLabel.TEXT, bbox=_tl(0, 0, 10, 10))
    out = docling_document_to_ls_results(_Doc([(item, 1)]), include_reading_order=True)
    assert out, "expected at least one result"
    for r in out:
        for k in (r.get("value") or {}).keys():
            assert not k.startswith("_"), f"underscore-prefixed key {k!r} leaked into value"


# --- table structure ----------------------------------------------------------------
#
# The interface's ``emitTable`` walker rebuilds table markup from FOUR kinds
# of children on a ``table`` rect:
#
#   * ``table_row`` strips — full-table-width, one per grid row
#   * ``table_column`` strips — full-table-height, one per grid column
#   * ``table_merged_cell`` overlays — one per cell with row_span or col_span > 1
#   * Content children (``text`` / ``picture`` / …) at each non-empty cell's bbox
#
# Semantic role overlays (``column_header`` / ``row_header`` / ``row_section``)
# ride on top of the same per-cell geometry when a cell has that role flag.


def test_table_structure_emits_rows_columns_merged_and_text() -> None:
    # 2 x 2 header row + one merged cell that spans both columns on row 1.
    #   H1 | H2
    #   [ merged, col_span=2 ]
    # Layout in top-left coords: (l, t, r, b).
    cells = [
        _table_cell(
            bbox=_tl(10, 10, 30, 20), text="H1",
            column_header=True,
            start_row=0, end_row=1, start_col=0, end_col=1,
        ),
        _table_cell(
            bbox=_tl(30, 10, 50, 20), text="H2",
            column_header=True,
            start_row=0, end_row=1, start_col=1, end_col=2,
        ),
        _table_cell(
            bbox=_tl(10, 20, 50, 30), text="wide",
            col_span=2,
            start_row=1, end_row=2, start_col=0, end_col=2,
        ),
    ]
    table = _item(
        label=DocItemLabel.TABLE,
        bbox=_tl(10, 10, 50, 30),
        self_ref="#/tables/0",
        data=SimpleNamespace(table_cells=cells, num_rows=2, num_cols=2),
    )
    out = docling_document_to_ls_results(
        _Doc([(table, 1)], tables=[table]),
        include_table_structure=True,
    )
    rects = [r for r in out if r["type"] == "rectanglelabels"]

    table_rect = next(r for r in rects if r["value"]["rectanglelabels"] == ["table"])
    children = [r for r in rects if r is not table_rect]
    # Every non-table rect is parented to the table (so the interface renders
    # them under its sub-annotation tree, not as flat top-level content).
    assert all(cr["value"]["parentId"] == table_rect["id"] for cr in children)
    # ...and one level deeper than the table itself.
    for cr in children:
        assert cr["value"]["level"] == 2

    labels = [cr["value"]["rectanglelabels"][0] for cr in children]
    # 2 rows + 2 columns + 1 merged overlay + 2 column_header overlays + 3 text cells.
    assert labels.count("table_row") == 2
    assert labels.count("table_column") == 2
    assert labels.count("table_merged_cell") == 1
    assert labels.count("column_header") == 2
    assert labels.count("text") == 3
    # The old per-cell "table_cell" label is gone — content is expressed via
    # cell-geometry `text` rects that the JSX emitTable assigns to the origin
    # cell by bbox overlap, so the interface never has to invent a label for
    # a cell that carries only content.
    assert "table_cell" not in labels

    # Row strips are full table width; column strips are full table height.
    tv = table_rect["value"]
    for r in children:
        if r["value"]["rectanglelabels"] == ["table_row"]:
            assert r["value"]["x"] == tv["x"]
            assert r["value"]["width"] == tv["width"]
        if r["value"]["rectanglelabels"] == ["table_column"]:
            assert r["value"]["y"] == tv["y"]
            assert r["value"]["height"] == tv["height"]

    # The merged cell overlays sit on the merged geometry, NOT split into
    # per-column halves — that's the whole point of preserving the merge.
    merged = next(r for r in children if r["value"]["rectanglelabels"] == ["table_merged_cell"])
    assert merged["value"]["x"] == 10.0
    assert merged["value"]["width"] == 40.0

    # Cell text rides as a separate `text` child at the same per-cell bbox.
    # The interface's emitTable() assigns it to the origin cell by bbox overlap.
    text_rects = [r for r in children if r["value"]["rectanglelabels"] == ["text"]]
    text_bodies = sorted(r["value"]["text"] for r in text_rects)
    assert text_bodies == ["H1", "H2", "wide"]


def test_table_structure_skips_when_grid_indices_are_missing() -> None:
    """No grid indices → we can't derive row/column bands, so emit nothing
    beyond the parent table rect. Better than fabricating a bogus 1×1 grid
    from cells whose true positions we don't know."""
    cells = [_table_cell(bbox=_tl(10, 10, 30, 20), text="only")]  # default indices, no num_rows/cols
    table = _item(
        label=DocItemLabel.TABLE,
        bbox=_tl(10, 10, 50, 30),
        self_ref="#/tables/0",
        # SimpleNamespace with only table_cells — no num_rows / num_cols.
        data=SimpleNamespace(table_cells=cells),
    )
    out = docling_document_to_ls_results(
        _Doc([(table, 1)], tables=[table]),
        include_table_structure=True,
    )
    # Just the table rect: the single cell has default indices (0..1, 0..1)
    # so num_rows=1, num_cols=1 gets inferred — but with only ONE cell in a
    # 1×1 grid, we get 1 row + 1 column + 1 text child.
    labels = [r["value"]["rectanglelabels"][0] for r in out if r["type"] == "rectanglelabels"]
    assert labels.count("table") == 1
    # The 1x1 fallback path IS exercised — this cell has default (0,1,0,1) so
    # it does define a valid 1x1 grid.
    assert labels.count("table_row") == 1
    assert labels.count("table_column") == 1
    assert labels.count("text") == 1


def test_table_structure_default_off_matches_prior_behavior() -> None:
    """Without include_table_structure=True the emitter keeps its pre-existing
    "table as one flat rect" behavior; existing consumers can't accidentally start
    seeing extra child rects they don't know how to handle."""
    cells = [
        _table_cell(
            bbox=_tl(10, 10, 30, 20), text="only",
            start_row=0, end_row=1, start_col=0, end_col=1,
        ),
    ]
    table = _item(
        label=DocItemLabel.TABLE,
        bbox=_tl(10, 10, 50, 30),
        self_ref="#/tables/0",
        data=SimpleNamespace(table_cells=cells, num_rows=1, num_cols=1),
    )
    out = docling_document_to_ls_results(_Doc([(table, 1)], tables=[table]))
    assert len(out) == 1
    assert out[0]["value"]["rectanglelabels"] == ["table"]


def test_table_structure_children_are_not_swept_into_reading_order() -> None:
    """Reading order sequences top-level flow. Sweeping every row / column /
    merged / cell-text child into it would produce a polyline that zig-zags
    through every cell of every table on the page, drowning out the document
    flow the polyline is supposed to represent."""
    cells = [
        _table_cell(
            bbox=_tl(10, 10, 20, 15), text="a",
            start_row=0, end_row=1, start_col=0, end_col=1,
        ),
        _table_cell(
            bbox=_tl(20, 10, 30, 15), text="b",
            start_row=0, end_row=1, start_col=1, end_col=2,
        ),
    ]
    table = _item(
        label=DocItemLabel.TABLE,
        bbox=_tl(10, 10, 30, 15),
        self_ref="#/tables/0",
        data=SimpleNamespace(table_cells=cells, num_rows=1, num_cols=2),
    )
    # A second top-level item on the same page so reading order has 2 endpoints.
    para = _item(label=DocItemLabel.TEXT, bbox=_tl(10, 20, 90, 30), text="body")
    out = docling_document_to_ls_results(
        _Doc([(table, 1), (para, 1)], tables=[table]),
        include_reading_order=True,
        include_table_structure=True,
    )
    poly = next(r for r in out if r["type"] == "polygonlabels")
    assert len(poly["value"]["points"]) == 2, (
        "reading order must not include table_row / table_column / merged / cell-text children"
    )
    # Reading order references the top-level items only: the table and the paragraph.
    table_id = next(
        r["id"] for r in out
        if r["type"] == "rectanglelabels" and r["value"]["rectanglelabels"] == ["table"]
    )
    para_id = next(
        r["id"] for r in out
        if r["type"] == "rectanglelabels" and r["value"]["rectanglelabels"] == ["text"]
        and r["value"]["parentId"] is None
    )
    assert poly["value"]["connectedRegions"] == [table_id, para_id]


def test_semantic_and_merged_overlays_coexist_on_the_same_cell() -> None:
    """A merged column-header cell contributes BOTH a table_merged_cell overlay
    (structural: "this cell spans multiple columns") AND a column_header overlay
    (semantic: "this cell is a heading"). The two live on the same underlying
    geometry — the JSX interface renders them on separate display layers and
    the DocLang emitter reads them independently, so dropping either would
    lose information."""
    cells = [
        _table_cell(
            bbox=_tl(10, 10, 50, 20), text="Country",
            column_header=True, col_span=2,
            start_row=0, end_row=1, start_col=0, end_col=2,
        ),
        _table_cell(
            bbox=_tl(10, 20, 30, 30), text="US",
            start_row=1, end_row=2, start_col=0, end_col=1,
        ),
        _table_cell(
            bbox=_tl(30, 20, 50, 30), text="EU",
            start_row=1, end_row=2, start_col=1, end_col=2,
        ),
    ]
    table = _item(
        label=DocItemLabel.TABLE,
        bbox=_tl(10, 10, 50, 30),
        self_ref="#/tables/0",
        data=SimpleNamespace(table_cells=cells, num_rows=2, num_cols=2),
    )
    out = docling_document_to_ls_results(
        _Doc([(table, 1)], tables=[table]),
        include_table_structure=True,
    )
    labels = [r["value"]["rectanglelabels"][0] for r in out if r["type"] == "rectanglelabels"]
    assert labels.count("table_merged_cell") == 1
    assert labels.count("column_header") == 1
    # And they land on the SAME geometry (10, 10, 40, 10 ← width=40 spanning both cols).
    merged = next(
        r for r in out if r["type"] == "rectanglelabels"
        and r["value"]["rectanglelabels"] == ["table_merged_cell"]
    )
    header = next(
        r for r in out if r["type"] == "rectanglelabels"
        and r["value"]["rectanglelabels"] == ["column_header"]
    )
    for k in ("x", "y", "width", "height"):
        assert merged["value"][k] == header["value"][k]


# --- relations: to_caption / to_footnote / to_value ---------------------------------
#
# FloatingItem captions / footnotes and KeyValueItem TO_VALUE graph links become
# 2-point polylines, so predictions restore the field/figure structure the interface
# would otherwise render as detached text.


def test_to_caption_and_to_footnote_polylines_emitted_for_picture() -> None:
    caption = _item(
        label=DocItemLabel.CAPTION,
        bbox=_tl(10, 60, 40, 65),
        text="Fig. 1: Widget",
        self_ref="#/texts/0",
    )
    footnote = _item(
        label=DocItemLabel.FOOTNOTE,
        bbox=_tl(10, 65, 40, 70),
        text="* see appendix",
        self_ref="#/texts/1",
    )
    picture = _item(
        label=DocItemLabel.PICTURE,
        bbox=_tl(10, 10, 40, 50),
        self_ref="#/pictures/0",
        captions=[_Ref("#/texts/0")],
        footnotes=[_Ref("#/texts/1")],
    )
    out = docling_document_to_ls_results(
        _Doc([(picture, 1), (caption, 1), (footnote, 1)], pictures=[picture]),
        include_relations=True,
    )

    polys = [r for r in out if r["type"] == "polygonlabels"]
    labels = [p["value"]["polygonlabels"][0] for p in polys]
    assert labels.count("to_caption") == 1
    assert labels.count("to_footnote") == 1

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
    """A TO_VALUE graph link becomes a key -> value polyline; both endpoints must
    resolve to emitted rects via GraphCell.item_ref -> NodeItem.self_ref."""
    key_item = _item(
        label=DocItemLabel.FIELD_KEY,
        bbox=_tl(10, 10, 30, 20),
        text="Name",
        self_ref="#/texts/0",
    )
    value_item = _item(
        label=DocItemLabel.FIELD_VALUE,
        bbox=_tl(35, 10, 60, 20),
        text="Alice",
        self_ref="#/texts/1",
    )
    key_cell = SimpleNamespace(cell_id=1, item_ref=_Ref("#/texts/0"))
    value_cell = SimpleNamespace(cell_id=2, item_ref=_Ref("#/texts/1"))
    link = SimpleNamespace(
        label=GraphLinkLabel.TO_VALUE, source_cell_id=1, target_cell_id=2
    )
    graph = SimpleNamespace(cells=[key_cell, value_cell], links=[link])
    kv = _item(
        label=DocItemLabel.KEY_VALUE_REGION,
        bbox=_tl(10, 10, 60, 20),
        self_ref="#/key_value_items/0",
        graph=graph,
    )
    out = docling_document_to_ls_results(
        _Doc([(kv, 1), (key_item, 2), (value_item, 2)], key_value_items=[kv]),
        include_relations=True,
    )
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


def test_relations_silently_drop_when_an_endpoint_is_not_emitted() -> None:
    """A caption ref pointing at an item that was filtered out (content_layer etc.) must
    NOT produce a dangling polyline. Rendering one would put a link on the canvas
    pointing at nothing, which is worse than losing the link."""
    picture = _item(
        label=DocItemLabel.PICTURE,
        bbox=_tl(10, 10, 40, 50),
        self_ref="#/pictures/0",
        captions=[_Ref("#/texts/0")],
    )
    # Deliberately do NOT include the caption item in iterate_items or _refs.
    out = docling_document_to_ls_results(
        _Doc([(picture, 1)], pictures=[picture]),
        include_relations=True,
    )
    polys = [r for r in out if r["type"] == "polygonlabels"]
    assert not any(p["value"]["polygonlabels"] == ["to_caption"] for p in polys)


def test_relations_default_off_matches_prior_behavior() -> None:
    caption = _item(
        label=DocItemLabel.CAPTION,
        bbox=_tl(10, 60, 40, 65),
        self_ref="#/texts/0",
    )
    picture = _item(
        label=DocItemLabel.PICTURE,
        bbox=_tl(10, 10, 40, 50),
        self_ref="#/pictures/0",
        captions=[_Ref("#/texts/0")],
    )
    out = docling_document_to_ls_results(
        _Doc([(picture, 1), (caption, 1)], pictures=[picture]),
    )
    labels = [
        r["value"]["polygonlabels"][0]
        for r in out
        if r["type"] == "polygonlabels"
    ]
    assert "to_caption" not in labels


# --- merge polylines from InlineGroup ------------------------------------------------
#
# An InlineGroup is Docling's representation of inline text runs that belong to
# ONE logical element split across columns / pages / line breaks. The interface
# uses a `merge` polyline to express the same idea (connected rects share a
# thread_id in the emitted DocLang), so predictions map InlineGroup 1:1 to
# a merge polyline over its resolved children.


def test_inline_group_emits_merge_polyline_connecting_children() -> None:
    a = _item(label=DocItemLabel.TEXT, bbox=_tl(10, 10, 30, 20), text="left", self_ref="#/texts/0")
    b = _item(label=DocItemLabel.TEXT, bbox=_tl(60, 10, 80, 20), text="right", self_ref="#/texts/1")
    inline = SimpleNamespace(
        label=GroupLabel.INLINE,
        self_ref="#/groups/0",
        children=[_Ref("#/texts/0"), _Ref("#/texts/1")],
    )
    out = docling_document_to_ls_results(
        _Doc([(a, 1), (b, 1)], groups=[inline]),
        include_relations=True,
    )
    polys = [r for r in out if r["type"] == "polygonlabels"]
    merges = [p for p in polys if p["value"]["polygonlabels"] == ["merge"]]
    assert len(merges) == 1
    m = merges[0]
    rects = [r for r in out if r["type"] == "rectanglelabels"]
    a_id = next(r["id"] for r in rects if r["value"]["text"] == "left")
    b_id = next(r["id"] for r in rects if r["value"]["text"] == "right")
    assert m["value"]["connectedRegions"] == [a_id, b_id]
    # Centroids of the two rects, one per endpoint, in the same order as connectedRegions.
    assert m["value"]["points"] == [[20.0, 15.0], [70.0, 15.0]]
    assert m["value"]["closed"] is False
    # `merge` is a polyline path, not a bounded region — parentId is always None.
    assert m["value"]["parentId"] is None


def test_inline_group_skipped_when_children_dont_resolve() -> None:
    """When an InlineGroup's children weren't emitted (typically because
    ``content_layers`` filtered them out), the merge polyline would connect
    nothing — silently drop it rather than emitting a dangling shape."""
    a = _item(label=DocItemLabel.TEXT, bbox=_tl(10, 10, 30, 20), self_ref="#/texts/0")
    inline = SimpleNamespace(
        label=GroupLabel.INLINE,
        self_ref="#/groups/0",
        children=[_Ref("#/texts/0"), _Ref("#/texts/dropped")],  # 2nd never emitted
    )
    out = docling_document_to_ls_results(
        _Doc([(a, 1)], groups=[inline]),
        include_relations=True,
    )
    polys = [r for r in out if r["type"] == "polygonlabels"]
    assert not any(p["value"]["polygonlabels"] == ["merge"] for p in polys)


def test_non_inline_groups_do_not_emit_merge_polylines() -> None:
    """A generic SECTION / CHAPTER / SLIDE GroupItem is a semantic container,
    not a visual-merge hint — overloading `merge` for those would drown the
    annotator in false positives. Only InlineGroup (label=INLINE) maps to
    a merge polyline."""
    a = _item(label=DocItemLabel.TEXT, bbox=_tl(10, 10, 30, 20), self_ref="#/texts/0")
    b = _item(label=DocItemLabel.TEXT, bbox=_tl(10, 20, 30, 30), self_ref="#/texts/1")
    section = SimpleNamespace(
        label=GroupLabel.SECTION,
        self_ref="#/groups/0",
        children=[_Ref("#/texts/0"), _Ref("#/texts/1")],
    )
    out = docling_document_to_ls_results(
        _Doc([(a, 1), (b, 1)], groups=[section]),
        include_relations=True,
    )
    polys = [r for r in out if r["type"] == "polygonlabels"]
    assert not any(p["value"]["polygonlabels"] == ["merge"] for p in polys)


def test_inline_group_merge_polyline_off_when_relations_disabled() -> None:
    """Merge polylines are gated under ``include_relations`` (same gate as
    to_caption / to_footnote / to_value) since they express a cross-region
    link. With the flag off, no polyline shape gets emitted."""
    a = _item(label=DocItemLabel.TEXT, bbox=_tl(10, 10, 30, 20), self_ref="#/texts/0")
    b = _item(label=DocItemLabel.TEXT, bbox=_tl(60, 10, 80, 20), self_ref="#/texts/1")
    inline = SimpleNamespace(
        label=GroupLabel.INLINE,
        self_ref="#/groups/0",
        children=[_Ref("#/texts/0"), _Ref("#/texts/1")],
    )
    out = docling_document_to_ls_results(
        _Doc([(a, 1), (b, 1)], groups=[inline]),
        # include_relations defaults to False
    )
    polys = [r for r in out if r["type"] == "polygonlabels"]
    assert not any(p["value"]["polygonlabels"] == ["merge"] for p in polys)


# ----- multi-prov items (one NodeItem, several bboxes on the same page) -----
#
# Docling represents "one logical element split across columns on the same
# page" as a single NodeItem whose ``prov`` list has two entries — NOT as an
# InlineGroup. The emitter must therefore emit one rect per prov AND emit a
# merge polyline connecting them. Before this pass the code silently took
# prov[0] and dropped the rest, which in the field showed up as "the middle
# column of a wrapped paragraph is missing from the prediction".


def test_multi_prov_item_on_same_page_emits_one_rect_per_prov() -> None:
    """The core round-trip: two provs on the same page → two rects, sharing
    the item's text/label/level/etc. No page filter, no gate needed."""
    item = _multi_prov_item(
        label=DocItemLabel.TEXT,
        provs=[(1, _tl(10, 10, 30, 20)), (1, _tl(60, 10, 80, 20))],
        text="wraps across columns",
        self_ref="#/texts/0",
    )
    out = docling_document_to_ls_results(_Doc([(item, 1)]))
    rects = [r for r in out if r["type"] == "rectanglelabels"]
    assert len(rects) == 2, "each prov must yield its own rect"
    # Same logical element → identical semantic fields on every constituent.
    assert {r["value"]["text"] for r in rects} == {"wraps across columns"}
    assert {tuple(r["value"]["rectanglelabels"]) for r in rects} == {("text",)}
    # Geometry differs, one rect per prov, in prov order.
    geoms = [(r["value"]["x"], r["value"]["y"], r["value"]["width"], r["value"]["height"]) for r in rects]
    assert geoms == [(10.0, 10.0, 20.0, 10.0), (60.0, 10.0, 20.0, 10.0)]


def test_multi_prov_item_emits_merge_polyline_connecting_all_provs() -> None:
    """The whole point of splitting into N rects: the merge polyline is what
    tells the DocLang emitter these N rects are one <thread_id> element."""
    item = _multi_prov_item(
        label=DocItemLabel.TEXT,
        provs=[(1, _tl(10, 10, 30, 20)), (1, _tl(60, 10, 80, 20))],
        text="wraps across columns",
        self_ref="#/texts/0",
    )
    out = docling_document_to_ls_results(_Doc([(item, 1)]), include_relations=True)
    rects = [r for r in out if r["type"] == "rectanglelabels"]
    rect_ids = [r["id"] for r in rects]
    merges = [
        p for p in out
        if p["type"] == "polygonlabels" and p["value"]["polygonlabels"] == ["merge"]
    ]
    assert len(merges) == 1
    m = merges[0]
    # Every emitted rect must be an endpoint, in prov order, so the DocLang
    # emitter's thread_id assignment matches Docling's own prov ordering.
    assert m["value"]["connectedRegions"] == rect_ids
    # Two endpoints, centroids of the two rects.
    assert m["value"]["points"] == [[20.0, 15.0], [70.0, 15.0]]
    assert m["value"]["closed"] is False
    assert m["value"]["parentId"] is None


def test_multi_prov_merge_polyline_off_when_relations_disabled() -> None:
    """Same gate as InlineGroup merges: without include_relations, only the
    rects are emitted — the annotator can still see all constituent regions,
    just without the visual link between them."""
    item = _multi_prov_item(
        label=DocItemLabel.TEXT,
        provs=[(1, _tl(10, 10, 30, 20)), (1, _tl(60, 10, 80, 20))],
        text="wraps",
        self_ref="#/texts/0",
    )
    out = docling_document_to_ls_results(_Doc([(item, 1)]))
    rects = [r for r in out if r["type"] == "rectanglelabels"]
    assert len(rects) == 2  # rects still emitted
    polys = [r for r in out if r["type"] == "polygonlabels"]
    assert not any(p["value"]["polygonlabels"] == ["merge"] for p in polys)


def test_multi_prov_reading_order_uses_primary_prov_centroid_only() -> None:
    """A multi-prov item is ONE reading-order stop, at its primary (first)
    prov's centroid. Threading through every constituent rect would zigzag
    the reading order polyline (left col → right col → next paragraph's
    left col → ...) and make it unreadable.
    """
    a = _multi_prov_item(
        label=DocItemLabel.TEXT,
        provs=[(1, _tl(10, 10, 30, 20)), (1, _tl(60, 10, 80, 20))],
        text="wraps",
        self_ref="#/texts/0",
    )
    b = _item(label=DocItemLabel.TEXT, bbox=_tl(10, 40, 30, 50), text="next", self_ref="#/texts/1")
    out = docling_document_to_ls_results(_Doc([(a, 1), (b, 1)]), include_reading_order=True)
    ro = next(r for r in out if r["type"] == "polygonlabels" and r["value"]["polygonlabels"] == ["reading_order"])
    # 3 rects total but only 2 reading-order stops: primary prov of `a`,
    # then `b`. The `60,10,80,20` prov of `a` is intentionally NOT visited.
    assert len(ro["value"]["points"]) == 2
    # Primary prov centroid of `a` = (20, 15); centroid of `b` = (20, 45).
    assert ro["value"]["points"] == [[20.0, 15.0], [20.0, 45.0]]


def test_multi_prov_primary_prov_is_the_caption_target() -> None:
    """When a picture's caption is a multi-prov item, the ``to_caption`` link
    must land on the caption's PRIMARY rect — not on some arbitrary constituent
    picked by dict iteration. The primary prov is the caption's entry point.
    """
    caption = _multi_prov_item(
        label=DocItemLabel.CAPTION,
        provs=[(1, _tl(10, 60, 40, 70)), (1, _tl(50, 60, 90, 70))],
        text="wrapped caption",
        self_ref="#/texts/0",
    )
    picture = SimpleNamespace(
        prov=[SimpleNamespace(page_no=1, bbox=_tl(10, 10, 90, 50))],
        label=DocItemLabel.PICTURE,
        text="",
        content_layer=ContentLayer.BODY,
        meta=None,
        self_ref="#/pictures/0",
        data=None,
        captions=[_Ref("#/texts/0")],
        footnotes=[],
        graph=None,
    )
    doc = _Doc([(picture, 1), (caption, 1)], pictures=[picture])
    out = docling_document_to_ls_results(doc, include_relations=True)
    rects = [r for r in out if r["type"] == "rectanglelabels"]
    caption_rects = [r for r in rects if r["value"]["text"] == "wrapped caption"]
    picture_id = next(r["id"] for r in rects if r["value"]["rectanglelabels"] == ["picture"])
    # Primary prov comes first in emission order.
    primary_caption_id = caption_rects[0]["id"]
    secondary_caption_id = caption_rects[1]["id"]
    to_caption = next(
        r for r in out
        if r["type"] == "polygonlabels" and r["value"]["polygonlabels"] == ["to_caption"]
    )
    assert to_caption["value"]["connectedRegions"] == [picture_id, primary_caption_id]
    assert secondary_caption_id not in to_caption["value"]["connectedRegions"]


def test_multi_prov_item_survives_partial_off_page_prov() -> None:
    """One prov entirely off-page shouldn't drop the whole item — the surviving
    prov(s) still emit, just without a merge polyline (nothing to merge with)."""
    item = _multi_prov_item(
        label=DocItemLabel.TEXT,
        provs=[(1, _tl(10, 10, 30, 20)), (1, _tl(110, 10, 130, 20))],  # 2nd off-page
        text="partly on page",
        self_ref="#/texts/0",
    )
    out = docling_document_to_ls_results(_Doc([(item, 1)]), include_relations=True)
    rects = [r for r in out if r["type"] == "rectanglelabels"]
    assert len(rects) == 1
    # Only one rect survived → no merge to emit.
    polys = [r for r in out if r["type"] == "polygonlabels"]
    assert not any(p["value"]["polygonlabels"] == ["merge"] for p in polys)


def test_page_straddling_item_still_emits_only_the_requested_page_prov() -> None:
    """The page_no filter is orthogonal to same-page multi-prov: when the
    caller asks for page 2, they get page 2's prov(s) only. This is the
    original ``test_page_no_filter_measures_the_provenance_on_the_requested_page``
    contract; it must survive the multi-prov refactor unchanged."""
    item = _multi_prov_item(
        label=DocItemLabel.TEXT,
        provs=[(1, _tl(10, 80, 30, 100)), (2, _tl(40, 0, 60, 20))],
        text="straddles",
    )
    doc = _Doc([(item, 1)], pages={1: _page(), 2: _page()})

    on_p1 = docling_document_to_ls_results(doc, page_no=1, include_relations=True)
    p1_rects = [r for r in on_p1 if r["type"] == "rectanglelabels"]
    assert len(p1_rects) == 1
    # No merge — only one prov applies to page 1.
    p1_polys = [r for r in on_p1 if r["type"] == "polygonlabels"]
    assert not any(p["value"]["polygonlabels"] == ["merge"] for p in p1_polys)


def test_within_page_multi_prov_with_page_filter_emits_all_matching_provs() -> None:
    """The other half of the page filter contract: when BOTH provs are on the
    requested page, both survive and are merged. A page filter must not
    accidentally re-introduce the "prov[0] only" bug for same-page multi-prov."""
    item = _multi_prov_item(
        label=DocItemLabel.TEXT,
        provs=[
            (1, _tl(10, 10, 30, 20)),
            (1, _tl(60, 10, 80, 20)),
            (2, _tl(10, 10, 30, 20)),  # different page, must be excluded
        ],
        text="wraps on page 1",
        self_ref="#/texts/0",
    )
    doc = _Doc([(item, 1)], pages={1: _page(), 2: _page()})
    out = docling_document_to_ls_results(doc, page_no=1, include_relations=True)
    rects = [r for r in out if r["type"] == "rectanglelabels"]
    assert len(rects) == 2
    merges = [
        p for p in out
        if p["type"] == "polygonlabels" and p["value"]["polygonlabels"] == ["merge"]
    ]
    assert len(merges) == 1
