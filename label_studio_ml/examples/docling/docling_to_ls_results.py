"""Map DoclingDocument items to canonical Label Studio result entries.

The Docling Interface (``docling-ls-implementation/docling_interface.jsx``,
a HumanSignal Interfaces project) reads predictions through its
``parseResults`` function and expects canonical Label Studio result shapes.
This module emits the shapes Docling can populate from a converted document:

  * ``rectanglelabels`` for layout regions, including per-cell rects for
    ``TableItem`` structure (``table_cell`` / ``column_header`` /
    ``row_header`` / ``row_section`` / ``table_merged_cell``) parented to
    their enclosing table.
  * ``polygonlabels`` for the reading-order polyline and for the
    ``to_caption`` / ``to_footnote`` / ``to_value`` linking polylines that
    connect a container to its caption / footnote and a key to its value.

The interface understands two further shapes — ``textarea`` for the doclang XML
snapshot and ``relation`` for region-to-region links — which only ever come from
manual annotation, so nothing here produces them.

The interface's ``getResults`` uses the same shapes when serializing manual
annotations, so predictions and human edits round-trip through the same code
paths without any shape gymnastics on either side.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from docling_core.types.doc.document import ContentLayer, DoclingDocument, NodeItem
from docling_core.types.doc.labels import DocItemLabel, GraphLinkLabel

logger = logging.getLogger(__name__)

# DoclingDocument labels -> canonical labels used in docling_interface.jsx LABEL_CATEGORIES.
DOCLING_LABEL_TO_LS: Dict[DocItemLabel, str] = {
    DocItemLabel.TITLE: "section_header",
    DocItemLabel.SECTION_HEADER: "section_header",
    DocItemLabel.PARAGRAPH: "text",
    DocItemLabel.TEXT: "text",
    DocItemLabel.LIST_ITEM: "list_item",
    DocItemLabel.TABLE: "table",
    DocItemLabel.PICTURE: "picture",
    DocItemLabel.CHART: "picture",
    DocItemLabel.FORMULA: "formula",
    DocItemLabel.CODE: "code",
    DocItemLabel.CAPTION: "caption",
    DocItemLabel.FOOTNOTE: "footnote",
    DocItemLabel.PAGE_HEADER: "page_header",
    DocItemLabel.PAGE_FOOTER: "page_footer",
    DocItemLabel.DOCUMENT_INDEX: "document_index",
    DocItemLabel.FORM: "form",
    DocItemLabel.KEY_VALUE_REGION: "text",
    DocItemLabel.CHECKBOX_SELECTED: "checkbox_selected",
    DocItemLabel.CHECKBOX_UNSELECTED: "checkbox_unselected",
    DocItemLabel.GRADING_SCALE: "grading_scale",
    DocItemLabel.HANDWRITTEN_TEXT: "handwritten_text",
    DocItemLabel.FIELD_KEY: "key",
    DocItemLabel.FIELD_VALUE: "value",
    DocItemLabel.FIELD_HEADING: "section_header",
    DocItemLabel.FIELD_HINT: "text",
    DocItemLabel.EMPTY_VALUE: "empty_value",
    DocItemLabel.REFERENCE: "footnote",
    DocItemLabel.MARKER: "text",
}

LS_CONTENT_LAYERS = {"BODY", "FURNITURE", "BACKGROUND"}

PICTURE_TYPES = {
    "CHART",
    "INFOGRAPHIC",
    "SCREENSHOT",
    "UI_ELEMENT",
    "BARCODE",
    "LOGO",
    "PICTOGRAM",
    "OTHER",
    "PERSON",
    "DECORATION",
    "ILLUSTRATION",
}


def _content_layer_to_ls(layer: ContentLayer) -> str:
    if layer == ContentLayer.FURNITURE:
        return "FURNITURE"
    if layer == ContentLayer.BACKGROUND:
        return "BACKGROUND"
    return "BODY"


# Percentages are emitted at this precision; every coordinate goes through _clip_pct so
# the rounded values themselves satisfy the 0-100 bounds, not just their inputs.
_PCT_DIGITS = 4


def _clip_pct(value_px: float, extent_px: float) -> float:
    """Convert a pixel edge to a page percentage, clipped to [0, 100] and rounded."""
    return round(min(max(value_px / extent_px * 100.0, 0.0), 100.0), _PCT_DIGITS)


def _page_raster_size(page: Any) -> Optional[Any]:
    """Return the raster ``Size`` the emitted percentages are relative to."""
    size = page.image.size if getattr(page, "image", None) is not None else page.size
    if not size or not size.width or not size.height:
        return None
    return size


def _bbox_page_to_percent(
    doc: DoclingDocument,
    bbox: Any,
    page_no: int,
) -> Optional[Tuple[float, float, float, float, int]]:
    """Return ``(x%, y%, w%, h%, page_no)`` for a raw bbox on a specific page.

    Split out from ``_bbox_to_percent_rect`` so callers without a full
    ``NodeItem.prov`` — table cells (``TableCell.bbox``), graph cells, any
    future case — share the same top-left / raster-scale / clip / normalize
    pipeline. All the invariants exercised by the master test suite (edge
    clipping, edge sorting, sub-precision drop, degenerate-size guard) live
    here so table cells and node items behave identically.
    """
    if bbox is None:
        return None
    page = doc.pages.get(page_no)
    if page is None:
        return None
    # scale_to_size divides by old_size, so page.size must be non-degenerate too.
    if not page.size or not page.size.width or not page.size.height:
        return None
    target_size = _page_raster_size(page)
    if target_size is None:
        return None

    try:
        bbox_tl = bbox.to_top_left_origin(page_height=page.size.height)
    except Exception:
        return None
    scaled = bbox_tl.scale_to_size(old_size=page.size, new_size=target_size)
    w_px, h_px = target_size.width, target_size.height

    # Normalize the edges before clipping. BoundingBox.width is a signed r-l and .height an
    # unsigned abs(t-b), so neither tells us which edge is which; sort them instead.
    left, right = sorted((scaled.l, scaled.r))
    top, bottom = sorted((scaled.t, scaled.b))
    # Round the edges, then derive the size from the rounded edges. Rounding x and width
    # independently would let a sub-precision box round to width 0, and let x + width land
    # just past 100 — the two things the clip below is here to prevent.
    x0 = _clip_pct(left, w_px)
    x1 = _clip_pct(right, w_px)
    y0 = _clip_pct(top, h_px)
    y1 = _clip_pct(bottom, h_px)
    if x1 <= x0 or y1 <= y0:
        # Nothing of the box survives on the page (or it was degenerate to begin with).
        # Emitting it would put an invisible zero-area region on the canvas and a stray
        # point in the reading-order polyline.
        return None
    return (x0, y0, round(x1 - x0, _PCT_DIGITS), round(y1 - y0, _PCT_DIGITS), page_no)


def _bbox_to_percent_rect(
    doc: DoclingDocument,
    item: NodeItem,
    prov_index: int = 0,
) -> Optional[Tuple[float, float, float, float, int]]:
    """Return ``(x%, y%, width%, height%, page_no)`` in top-left page raster coordinates.

    Top-left / percentage coordinates match the interface's spatial-region
    format, so predictions and manual edits share the same coordinate
    convention and round-trip through the same code paths.

    Delegates the geometry to :func:`_bbox_page_to_percent` so a raw
    ``TableCell.bbox`` gets the same clipping / normalization treatment as
    a top-level NodeItem's provenance bbox.
    """
    if not item.prov or prov_index >= len(item.prov):
        return None
    prov = item.prov[prov_index]
    return _bbox_page_to_percent(doc, prov.bbox, prov.page_no)


def _ls_label_for_item(item: NodeItem) -> str:
    label = getattr(item, "label", None)
    if isinstance(label, DocItemLabel):
        return DOCLING_LABEL_TO_LS.get(label, "text")
    return "text"


def _picture_type(item: NodeItem, ls_label: str) -> Optional[str]:
    if ls_label != "picture":
        return None
    dl = getattr(item, "label", None)
    if dl == DocItemLabel.CHART:
        return "CHART"
    meta = getattr(item, "meta", None)
    if meta is None:
        return "OTHER"
    classification = getattr(meta, "classification", None)
    if not classification:
        return "OTHER"
    preds = getattr(classification, "predictions", None) or []
    if not preds:
        return "OTHER"
    name = getattr(preds[0], "class_name", None) or ""
    upper = name.upper().replace(" ", "_")
    if upper in PICTURE_TYPES:
        return upper
    return "OTHER"


def _item_text(item: NodeItem) -> str:
    t = getattr(item, "text", None)
    if t is None:
        return ""
    if isinstance(t, str):
        return t
    return str(t)


_CONTENT_LAYER_BY_NAME = {
    "body": ContentLayer.BODY,
    "furniture": ContentLayer.FURNITURE,
    "background": ContentLayer.BACKGROUND,
    "invisible": ContentLayer.INVISIBLE,
    "notes": ContentLayer.NOTES,
}


def _parse_content_layers(raw: Optional[str]) -> Optional[Set[ContentLayer]]:
    """Parse DOCLING_CONTENT_LAYERS; ``None`` means "use Docling's default (body only)"."""
    if not raw:
        return None
    out: Set[ContentLayer] = set()
    unknown: List[str] = []
    for part in raw.lower().split(","):
        part = part.strip()
        if not part:
            continue
        layer = _CONTENT_LAYER_BY_NAME.get(part)
        if layer is None:
            unknown.append(part)
        else:
            out.add(layer)
    if unknown:
        # Silently falling back to the default here reads as "my filter did nothing";
        # name the bad value so a typo is obvious from the logs.
        logger.warning(
            "Ignoring unknown DOCLING_CONTENT_LAYERS value(s) %s; supported layers are %s",
            ", ".join(sorted(unknown)),
            ", ".join(sorted(_CONTENT_LAYER_BY_NAME)),
        )
    if not out:
        logger.warning(
            "DOCLING_CONTENT_LAYERS=%r selected no known layer; using Docling's default (body only)",
            raw,
        )
    return out or None


def _table_cell_label(cell: Any) -> str:
    """Pick the interface label for a docling ``TableCell``.

    Header / row-section / merged / plain — matching what an annotator would
    draw manually. If a cell is both a header AND a merge, header wins (it's
    the more informative label; the merged geometry is still preserved as a
    single rectangle rather than N sub-cells).
    """
    if getattr(cell, "column_header", False):
        return "column_header"
    if getattr(cell, "row_header", False):
        return "row_header"
    if getattr(cell, "row_section", False):
        return "row_section"
    row_span = int(getattr(cell, "row_span", 1) or 1)
    col_span = int(getattr(cell, "col_span", 1) or 1)
    if row_span > 1 or col_span > 1:
        return "table_merged_cell"
    return "table_cell"


def _make_rect_result(
    *,
    ls_label: str,
    x_pct: float,
    y_pct: float,
    w_pct: float,
    h_pct: float,
    from_name: str,
    to_name: str,
    content_layer: str = "BODY",
    level: int = 1,
    picture_type: Optional[str] = None,
    text: str = "",
    parent_id: Optional[str] = None,
    score: Optional[float] = None,
    region_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a canonical ``rectanglelabels`` result envelope.

    Extracted so table cells share the exact same value-block shape as the
    top-level items and don't drift from the ``parseResults`` contract.
    Coordinates are trusted to already be clipped and rounded by
    :func:`_bbox_page_to_percent`; rounding again here would reintroduce the
    x + width > 100 drift the clipping exists to prevent.
    """
    rid = region_id or str(uuid.uuid4())
    value: Dict[str, Any] = {
        "x": x_pct,
        "y": y_pct,
        "width": w_pct,
        "height": h_pct,
        "rotation": 0,
        "rectanglelabels": [ls_label],
        "content_layer": content_layer,
        "level": max(1, min(100, int(level) if level else 1)),
        "picture_type": picture_type,
        "text": text or "",
        "parentId": parent_id,
    }
    out: Dict[str, Any] = {
        "id": rid,
        "from_name": from_name,
        "to_name": to_name,
        "type": "rectanglelabels",
        "origin": "prediction",
        "value": value,
    }
    if score is not None:
        out["score"] = score
    return out


def _rect_center(rect: Dict[str, Any]) -> Tuple[float, float]:
    """Return the (cx%, cy%) center of a ``rectanglelabels`` value block."""
    v = rect.get("value") or {}
    x = float(v.get("x", 0) or 0)
    y = float(v.get("y", 0) or 0)
    w = float(v.get("width", 0) or 0)
    h = float(v.get("height", 0) or 0)
    return x + w / 2.0, y + h / 2.0


def _make_link_polyline(
    *,
    label: str,
    src_rect: Dict[str, Any],
    dst_rect: Dict[str, Any],
    from_name: str,
    to_name: str,
    score: Optional[float] = None,
    level: int = 1,
) -> Dict[str, Any]:
    """Build a 2-point ``polygonlabels`` result linking two rectangles.

    Used for ``to_caption`` / ``to_footnote`` / ``to_value`` — the label
    values the interface's ``LINK_RESTRICTIONS`` in ``docling_interface.jsx``
    expects. Points are the geometric centers of each endpoint; the interface
    snaps them to their enclosing rects on next drag anyway, but drawing at
    the center gives a sensible initial visual.
    """
    sx, sy = _rect_center(src_rect)
    dx, dy = _rect_center(dst_rect)
    out: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "from_name": from_name,
        "to_name": to_name,
        "type": "polygonlabels",
        "origin": "prediction",
        "value": {
            "points": [[round(sx, 4), round(sy, 4)], [round(dx, 4), round(dy, 4)]],
            "polygonlabels": [label],
            "connectedRegions": [src_rect["id"], dst_rect["id"]],
            "level": max(1, min(100, int(level) if level else 1)),
            "validationErrors": [],
            "parentId": None,
            "closed": False,
        },
    }
    if score is not None:
        out["score"] = score
    return out


def _floating_items(doc: DoclingDocument) -> Iterable[Any]:
    """Yield the four FloatingItem collections that carry captions/footnotes.

    Guarded with ``getattr`` fallbacks so a minimal test fixture (only
    ``tables`` set, say) doesn't blow up.
    """
    for attr in ("tables", "pictures", "key_value_items", "form_items"):
        for it in getattr(doc, attr, None) or ():
            yield it


def _resolve_ref_to_rect(
    doc: DoclingDocument,
    ref: Any,
    ref_to_id: Dict[str, str],
    rect_by_id: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve a docling ``RefItem`` (or any ``.self_ref`` carrier) to an emitted rect.

    Returns ``None`` when the ref doesn't point at anything we emitted — the
    likely reason is content-layer filtering (e.g. the caption sat in
    ``FURNITURE`` and the caller only asked for ``BODY``), so silently
    dropping the relation is the right behavior. Callers should NOT treat
    None as an error.
    """
    if ref is None:
        return None
    resolved_ref: Optional[str] = None
    cref = getattr(ref, "cref", None)
    if isinstance(cref, str) and cref:
        try:
            resolved = ref.resolve(doc)
        except Exception:
            resolved = None
        if resolved is not None:
            resolved_ref = getattr(resolved, "self_ref", None)
    if resolved_ref is None:
        maybe = getattr(ref, "self_ref", None)
        if isinstance(maybe, str) and maybe:
            resolved_ref = maybe
    if not resolved_ref:
        return None
    rid = ref_to_id.get(resolved_ref)
    if rid is None:
        return None
    return rect_by_id.get(rid)


def _emit_relations(
    doc: DoclingDocument,
    *,
    ref_to_id: Dict[str, str],
    rect_by_id: Dict[str, Dict[str, Any]],
    from_name: str,
    to_name: str,
    score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Emit ``to_caption`` / ``to_footnote`` / ``to_value`` polylines.

    Walks the four FloatingItem collections (``tables`` / ``pictures`` /
    ``key_value_items`` / ``form_items``) that carry ``captions[]`` and
    ``footnotes[]`` refs, plus the ``graph.links`` on KV / form items for
    ``TO_VALUE`` pairs. Every link needs BOTH endpoints to have been emitted
    as rects in the main iteration pass — otherwise the interface would
    render a dangling polyline pointing at nothing. When an endpoint is
    missing we silently drop the link.
    """
    out: List[Dict[str, Any]] = []
    for floating in _floating_items(doc):
        src_ref = getattr(floating, "self_ref", None)
        src_rid = ref_to_id.get(src_ref) if isinstance(src_ref, str) else None
        src_rect = rect_by_id.get(src_rid) if src_rid else None
        if src_rect is None:
            continue

        for cap_ref in getattr(floating, "captions", None) or ():
            dst_rect = _resolve_ref_to_rect(doc, cap_ref, ref_to_id, rect_by_id)
            if dst_rect is None:
                continue
            out.append(
                _make_link_polyline(
                    label="to_caption",
                    src_rect=src_rect,
                    dst_rect=dst_rect,
                    from_name=from_name,
                    to_name=to_name,
                    score=score,
                )
            )
        for foot_ref in getattr(floating, "footnotes", None) or ():
            dst_rect = _resolve_ref_to_rect(doc, foot_ref, ref_to_id, rect_by_id)
            if dst_rect is None:
                continue
            out.append(
                _make_link_polyline(
                    label="to_footnote",
                    src_rect=src_rect,
                    dst_rect=dst_rect,
                    from_name=from_name,
                    to_name=to_name,
                    score=score,
                )
            )

        graph = getattr(floating, "graph", None)
        if graph is None:
            continue
        cells = getattr(graph, "cells", None) or ()
        cell_by_id = {int(getattr(c, "cell_id", -1)): c for c in cells}
        for link in getattr(graph, "links", None) or ():
            if getattr(link, "label", None) != GraphLinkLabel.TO_VALUE:
                continue
            src_cell = cell_by_id.get(int(getattr(link, "source_cell_id", -1)))
            dst_cell = cell_by_id.get(int(getattr(link, "target_cell_id", -1)))
            if src_cell is None or dst_cell is None:
                continue
            src_link_rect = _resolve_ref_to_rect(
                doc, getattr(src_cell, "item_ref", None), ref_to_id, rect_by_id
            )
            dst_link_rect = _resolve_ref_to_rect(
                doc, getattr(dst_cell, "item_ref", None), ref_to_id, rect_by_id
            )
            if src_link_rect is None or dst_link_rect is None:
                continue
            out.append(
                _make_link_polyline(
                    label="to_value",
                    src_rect=src_link_rect,
                    dst_rect=dst_link_rect,
                    from_name=from_name,
                    to_name=to_name,
                    score=score,
                )
            )
    return out


def docling_document_to_ls_results(
    doc: DoclingDocument,
    *,
    page_no: Optional[int] = None,
    include_reading_order: bool = False,
    reading_order_level: int = 1,
    include_table_structure: bool = False,
    include_relations: bool = False,
    content_layers: Optional[str] = None,
    from_name: str = "docling",
    to_name: str = "docling",
    score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build canonical Label Studio prediction results.

    Output is a flat list ready to drop into ``PredictionValue.result``. Each
    entry is a complete envelope (``id``, ``from_name``, ``to_name``, ``type``,
    ``value``) — the caller still needs to attach ``original_width`` /
    ``original_height`` / ``image_rotation`` to every entry, since Label Studio
    carries those per result rather than on the prediction as a whole. Use
    :func:`page_raster_size` for dimensions consistent with these percentages.

    What lands in the output:

      * ``rectanglelabels`` entries for every Docling layout item with a
        bounding box.
      * When ``include_table_structure`` is enabled: one extra
        ``rectanglelabels`` per structural cell of every ``TableItem`` —
        ``table_cell`` / ``column_header`` / ``row_header`` / ``row_section``
        / ``table_merged_cell`` picked per cell, with ``parentId`` set to
        the enclosing table's region id.
      * When ``include_reading_order`` is enabled: one ``polygonlabels`` per
        page tracing the centroids of that page's items in Docling's
        iteration order, labeled ``reading_order``.
      * When ``include_relations`` is enabled: ``to_caption`` / ``to_footnote``
        / ``to_value`` 2-point ``polygonlabels`` for every ``FloatingItem``
        caption/footnote ref and every ``KeyValueItem`` / ``FormItem``
        ``TO_VALUE`` graph link, provided both endpoints were emitted as
        rects (endpoints filtered out by ``content_layers`` are silently
        dropped — no dangling links).

    All three shape gates default OFF so callers pay only for what they ask
    for. ``model.py`` opts in explicitly (all three on by default at that
    layer) because the interface needs the reading-order polyline to render
    anything at all, and captions/values without their links reduce to
    disconnected text.
    """
    included = _parse_content_layers(content_layers)
    iter_kw: Dict[str, Any] = {
        "with_groups": False,
        "traverse_pictures": True,
    }
    if page_no is not None:
        iter_kw["page_no"] = page_no
    if included is not None:
        iter_kw["included_content_layers"] = included

    results: List[Dict[str, Any]] = []
    reading_centers: Dict[int, List[Tuple[str, float, float]]] = defaultdict(list)
    # NodeItem.self_ref -> region id, so the relations pass can link back to
    # rects we already emitted. Rects without a self_ref (raw fixtures in
    # tests, top-level items with the attr set to None) simply don't appear
    # in the map and any relation targeting them silently no-ops.
    ref_to_id: Dict[str, str] = {}
    rect_by_id: Dict[str, Dict[str, Any]] = {}

    for item, level in doc.iterate_items(**iter_kw):
        if not item.prov:
            continue
        # An item straddling a page break carries one provenance per page, and
        # iterate_items(page_no=N) yields it if *any* of them is on page N. Measure the
        # provenance actually on the requested page instead of assuming it is prov[0].
        prov_index = 0
        if page_no is not None:
            prov_index = next(
                (i for i, p in enumerate(item.prov) if p.page_no == page_no), -1
            )
            if prov_index < 0:
                continue
        rect = _bbox_to_percent_rect(doc, item, prov_index=prov_index)
        if rect is None:
            continue
        x_pct, y_pct, w_pct, h_pct, p_no = rect

        ls_label = _ls_label_for_item(item)
        layer = _content_layer_to_ls(getattr(item, "content_layer", ContentLayer.BODY))
        if layer not in LS_CONTENT_LAYERS:
            layer = "BODY"
        item_level = max(1, min(100, int(level) if level else 1))

        region_id = str(uuid.uuid4())
        result: Dict[str, Any] = {
            "id": region_id,
            "from_name": from_name,
            "to_name": to_name,
            "type": "rectanglelabels",
            "origin": "prediction",
            "value": {
                # Already clipped and rounded by _bbox_to_percent_rect; rounding again here
                # would reintroduce the x + width > 100 drift it exists to prevent.
                "x": x_pct,
                "y": y_pct,
                "width": w_pct,
                "height": h_pct,
                "rotation": 0,
                "rectanglelabels": [ls_label],
                "content_layer": layer,
                "level": item_level,
                "picture_type": _picture_type(item, ls_label),
                "text": _item_text(item),
                "parentId": None,
            },
        }
        if score is not None:
            result["score"] = score
        results.append(result)
        rect_by_id[region_id] = result
        self_ref = getattr(item, "self_ref", None)
        if isinstance(self_ref, str) and self_ref:
            ref_to_id[self_ref] = region_id

        if include_reading_order:
            cx = x_pct + w_pct / 2.0
            cy = y_pct + h_pct / 2.0
            reading_centers[p_no].append((region_id, cx, cy))

        # Table structure: emit one child rect per cell that has a bbox. Cells share
        # the parent table's page (docling always puts the whole table on a single page)
        # so we reuse p_no rather than trusting a potentially-missing prov on the cell.
        # The cell rect is intentionally NOT swept into the reading-order polyline: the
        # reading order sequences top-level flow, and a level-2 reading order inside a
        # cell is a human affordance that we don't fabricate.
        if (
            include_table_structure
            and ls_label == "table"
            and getattr(item, "data", None) is not None
        ):
            for cell in getattr(item.data, "table_cells", None) or ():
                cell_bbox = getattr(cell, "bbox", None)
                if cell_bbox is None:
                    continue
                cell_rect = _bbox_page_to_percent(doc, cell_bbox, p_no)
                if cell_rect is None:
                    continue
                cx_pct, cy_pct, cw_pct, ch_pct, _ = cell_rect
                cell_result = _make_rect_result(
                    ls_label=_table_cell_label(cell),
                    x_pct=cx_pct,
                    y_pct=cy_pct,
                    w_pct=cw_pct,
                    h_pct=ch_pct,
                    from_name=from_name,
                    to_name=to_name,
                    content_layer=layer,
                    level=item_level + 1,
                    text=getattr(cell, "text", "") or "",
                    parent_id=region_id,
                    score=score,
                )
                results.append(cell_result)
                rect_by_id[cell_result["id"]] = cell_result

    if include_reading_order:
        ro_level = max(1, min(100, int(reading_order_level)))
        for p_no, chain in reading_centers.items():
            if len(chain) < 2:
                continue
            ids = [c[0] for c in chain]
            points = [[round(c[1], 4), round(c[2], 4)] for c in chain]
            ro_result: Dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "from_name": from_name,
                "to_name": to_name,
                "type": "polygonlabels",
                "origin": "prediction",
                "value": {
                    "points": points,
                    "polygonlabels": ["reading_order"],
                    "connectedRegions": ids,
                    "level": ro_level,
                    "validationErrors": [],
                    "parentId": None,
                    "closed": False,
                },
            }
            if score is not None:
                ro_result["score"] = score
            results.append(ro_result)

    if include_relations:
        results.extend(
            _emit_relations(
                doc,
                ref_to_id=ref_to_id,
                rect_by_id=rect_by_id,
                from_name=from_name,
                to_name=to_name,
                score=score,
            )
        )

    return results


def page_raster_size(
    doc: DoclingDocument, page_no: Optional[int] = None
) -> Optional[Tuple[int, int]]:
    """Return ``(width, height)`` in px of the raster the percentages are relative to.

    This is the right source for a result's ``original_width`` / ``original_height``:
    it is the same raster :func:`docling_document_to_ls_results` measured against, and
    unlike probing the downloaded file it works for PDFs, which are not images.

    ``page_no`` defaults to the document's first page. Returns ``None`` rather than a
    degenerate size, so the caller can fall back instead of emitting a zero dimension.
    """
    pages = getattr(doc, "pages", None) or {}
    if not pages:
        return None
    page = pages.get(page_no) if page_no is not None else pages.get(min(pages))
    if page is None:
        return None
    size = _page_raster_size(page)
    if size is None:
        return None
    # Label Studio wants ints here; round rather than truncate so a fractional page size
    # (595.5 -> 596, not 595) stays as close as possible to the raster the percentages
    # were measured against.
    width, height = round(size.width), round(size.height)
    if width < 1 or height < 1:
        return None
    return width, height
