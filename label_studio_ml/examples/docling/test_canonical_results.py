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
from docling_core.types.doc.labels import DocItemLabel

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
) -> SimpleNamespace:
    return _multi_prov_item(
        label=label, provs=[(page_no, bbox)], text=text, content_layer=content_layer
    )


def _multi_prov_item(
    *,
    label: DocItemLabel,
    provs: List[Tuple[int, BoundingBox]],
    text: str = "",
    content_layer: ContentLayer = ContentLayer.BODY,
) -> SimpleNamespace:
    """An item with one provenance per page, as Docling reports for page-straddling items."""
    return SimpleNamespace(
        prov=[SimpleNamespace(page_no=p, bbox=b) for p, b in provs],
        label=label,
        text=text,
        content_layer=content_layer,
        meta=None,
    )


class _Doc:
    """Minimal DoclingDocument stand-in: real pages/bboxes, stubbed iteration."""

    def __init__(
        self,
        items_with_levels: List[Tuple[Any, int]],
        pages: Optional[Dict[int, Any]] = None,
    ):
        self._items = items_with_levels
        self.pages: Dict[int, Any] = pages if pages is not None else {1: _page()}

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
