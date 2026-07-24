"""Map DoclingDocument items to canonical Label Studio result entries.

The Docling Interface (``docling-ls-implementation/docling_interface.jsx``,
a HumanSignal Interfaces project) reads predictions through its
``parseResults`` function and expects canonical Label Studio result shapes.
This module emits the shapes Docling can populate from a converted document:

  * ``rectanglelabels`` for layout regions.
  * ``rectanglelabels`` for table structure — ``table_row`` / ``table_column``
    strips derived from the docling grid, ``table_merged_cell`` at each cell
    with ``row_span`` or ``col_span`` > 1, semantic overlays
    (``column_header`` / ``row_header`` / ``row_section``) at their per-cell
    geometry, and ``text`` content children at each non-empty cell's bbox.
    The row and column strips OVERLAP by design: the JSX ``emitTable`` walker
    computes each cell as ``intersect(row_bbox, col_bbox)``, and then assigns
    every content child to its origin cell by bbox overlap — so the
    row/column geometry rebuilds the grid AND the cell text rides on the
    grid without knowing its (r, c) index up front.
  * ``polygonlabels`` for the reading-order polyline, for the
    ``to_caption`` / ``to_footnote`` / ``to_value`` linking polylines that
    connect a container to its caption / footnote and a key to its value,
    and for ``merge`` polylines emitted in both cases Docling uses to
    represent a single logical element split across columns / pages / line
    breaks: (1) an ``InlineGroup`` wrapping multiple items, and (2) a
    single item whose ``prov`` list has more than one entry on the same
    page (typical for a paragraph that wraps between two columns). Both
    become a shared ``merge`` polyline connecting the constituent rects.

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
from docling_core.types.doc.labels import DocItemLabel, GraphLinkLabel, GroupLabel

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


def _cell_grid_extent(cell: Any) -> Tuple[int, int, int, int]:
    """Return ``(start_row, end_row, start_col, end_col)`` for a ``TableCell``.

    Reads the four ``*_offset_idx`` fields the real ``docling_core.TableCell``
    exposes. Fixture cells that omit them (SimpleNamespace with only bbox +
    text) get zeros, which combined with the ``num_rows``/``num_cols``
    fallback in :func:`_emit_table_structure` means such fixtures produce no
    structural overlays — the row/column geometry is not derivable without
    grid indices, so silently skipping is safer than fabricating a 1×1 layout.
    """
    return (
        int(getattr(cell, "start_row_offset_idx", 0) or 0),
        int(getattr(cell, "end_row_offset_idx", 0) or 0),
        int(getattr(cell, "start_col_offset_idx", 0) or 0),
        int(getattr(cell, "end_col_offset_idx", 0) or 0),
    )


def _emit_table_structure(
    doc: DoclingDocument,
    table_item: NodeItem,
    *,
    table_rect: Dict[str, Any],
    table_page_no: int,
    from_name: str,
    to_name: str,
    content_layer: str,
    item_level: int,
    score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Rebuild a Docling table into the interface's expected structural overlays.

    Docling SaaS ships a flat list of ``TableCell`` rectangles with grid
    coordinates and header/section flags. The interface's ``emitTable``
    (``docling-ls-implementation/docling_interface.jsx``) does NOT read
    per-cell rects; it walks children of the table looking for THREE kinds
    of structural overlays plus content children:

      * ``table_row`` — one horizontal strip per grid row, full table width
      * ``table_column`` — one vertical strip per grid column, full table height
        (row × column intersect = individual cell geometry — this is why the
        two must OVERLAP, and why we anchor them to the parent table's bbox
        rather than clipping to the cell union)
      * ``table_merged_cell`` — a per-cell overlay for cells with ``row_span``
        or ``col_span`` > 1, at the cell's own (already-merged) bbox
      * Semantic role overlays (``column_header`` / ``row_header`` /
        ``row_section``) at the same per-cell geometry as the merged overlay,
        so a merged column-header cell contributes BOTH a ``table_merged_cell``
        AND a ``column_header`` rect. The interface renders these on separate
        display layers and DocLang XML build cares about both dimensions.
      * ``text`` — one per non-empty cell, at the cell's exact bbox with
        ``parentId`` set to the table. The interface's ``emitTable`` assigns
        every content child to the origin cell whose bbox overlaps it most;
        emitting cell text as an independent content child rather than as a
        ``table_cell`` label preserves the docling round-trip AND makes the
        cell's OCR string visible to downstream DocLang emitters.

    Row bands are derived from cells that do NOT span multiple rows (row_span
    == 1) — those cells define the row's exact vertical extent. Same idea
    for column bands. A row/column with no single-span cell falls back to
    the union of all cells covering that grid position, which is the widest
    band consistent with the geometry available.

    Skips emission (returns []) when ``num_rows``/``num_cols`` is not
    derivable, when no cell has a bbox, or when the table rect itself has
    no geometry — every failure mode leaves the table as a single flat rect
    the annotator can rebuild by hand, which is the safe default rather
    than fabricating an incorrect grid.
    """
    data = getattr(table_item, "data", None)
    if data is None:
        return []
    cells = getattr(data, "table_cells", None) or ()
    if not cells:
        return []

    # Prefer TableData's own grid dimensions; fall back to inferring from cell
    # end-offset indices so fixtures without an explicit num_rows/num_cols
    # setting still work when they DO set the offsets on each cell.
    num_rows = int(getattr(data, "num_rows", 0) or 0)
    num_cols = int(getattr(data, "num_cols", 0) or 0)
    if num_rows < 1 or num_cols < 1:
        derived_rows = 0
        derived_cols = 0
        for cell in cells:
            _, er, _, ec = _cell_grid_extent(cell)
            derived_rows = max(derived_rows, er)
            derived_cols = max(derived_cols, ec)
        num_rows = num_rows if num_rows >= 1 else derived_rows
        num_cols = num_cols if num_cols >= 1 else derived_cols
    if num_rows < 1 or num_cols < 1:
        return []

    # Convert every cell's bbox to page-percent once. Store the raw cell
    # alongside so semantic and text overlays can read text / header flags
    # without a second lookup.
    cell_infos: List[Dict[str, Any]] = []
    for cell in cells:
        bbox = getattr(cell, "bbox", None)
        if bbox is None:
            continue
        rect = _bbox_page_to_percent(doc, bbox, table_page_no)
        if rect is None:
            continue
        x, y, w, h, _ = rect
        sr, er, sc, ec = _cell_grid_extent(cell)
        cell_infos.append(
            {"cell": cell, "x": x, "y": y, "w": w, "h": h, "sr": sr, "er": er, "sc": sc, "ec": ec}
        )
    if not cell_infos:
        return []

    tv = table_rect["value"]
    tx = float(tv.get("x") or 0.0)
    ty = float(tv.get("y") or 0.0)
    tw = float(tv.get("width") or 0.0)
    th = float(tv.get("height") or 0.0)
    table_rect_id = table_rect["id"]

    out: List[Dict[str, Any]] = []

    # Row bands: full-width strips at each grid row's Y range.
    for r in range(num_rows):
        strict = [ci for ci in cell_infos if ci["sr"] == r and ci["er"] == r + 1]
        pool = strict or [ci for ci in cell_infos if ci["sr"] <= r < ci["er"]]
        if not pool:
            continue
        top = min(ci["y"] for ci in pool)
        bottom = max(ci["y"] + ci["h"] for ci in pool)
        height = max(bottom - top, 0.0)
        if height <= 0:
            continue
        out.append(
            _make_rect_result(
                ls_label="table_row",
                x_pct=round(tx, _PCT_DIGITS),
                y_pct=round(top, _PCT_DIGITS),
                w_pct=round(tw, _PCT_DIGITS),
                h_pct=round(height, _PCT_DIGITS),
                from_name=from_name,
                to_name=to_name,
                content_layer=content_layer,
                level=item_level + 1,
                parent_id=table_rect_id,
                score=score,
            )
        )

    # Column bands: full-height strips at each grid column's X range.
    for c in range(num_cols):
        strict = [ci for ci in cell_infos if ci["sc"] == c and ci["ec"] == c + 1]
        pool = strict or [ci for ci in cell_infos if ci["sc"] <= c < ci["ec"]]
        if not pool:
            continue
        left = min(ci["x"] for ci in pool)
        right = max(ci["x"] + ci["w"] for ci in pool)
        width = max(right - left, 0.0)
        if width <= 0:
            continue
        out.append(
            _make_rect_result(
                ls_label="table_column",
                x_pct=round(left, _PCT_DIGITS),
                y_pct=round(ty, _PCT_DIGITS),
                w_pct=round(width, _PCT_DIGITS),
                h_pct=round(th, _PCT_DIGITS),
                from_name=from_name,
                to_name=to_name,
                content_layer=content_layer,
                level=item_level + 1,
                parent_id=table_rect_id,
                score=score,
            )
        )

    # Per-cell overlays: merged geometry AND semantic role coexist on the
    # same cell (a merged column-header contributes two overlapping rects).
    for ci in cell_infos:
        cell = ci["cell"]
        row_span = int(getattr(cell, "row_span", 1) or 1)
        col_span = int(getattr(cell, "col_span", 1) or 1)

        if row_span > 1 or col_span > 1:
            out.append(
                _make_rect_result(
                    ls_label="table_merged_cell",
                    x_pct=ci["x"],
                    y_pct=ci["y"],
                    w_pct=ci["w"],
                    h_pct=ci["h"],
                    from_name=from_name,
                    to_name=to_name,
                    content_layer=content_layer,
                    level=item_level + 1,
                    text="",
                    parent_id=table_rect_id,
                    score=score,
                )
            )

        # Semantic role — priority column_header > row_header > row_section,
        # matching how real docling data flags cells (mutually exclusive in
        # practice on the tables the SaaS emits).
        role_label: Optional[str] = None
        if getattr(cell, "column_header", False):
            role_label = "column_header"
        elif getattr(cell, "row_header", False):
            role_label = "row_header"
        elif getattr(cell, "row_section", False):
            role_label = "row_section"
        if role_label is not None:
            out.append(
                _make_rect_result(
                    ls_label=role_label,
                    x_pct=ci["x"],
                    y_pct=ci["y"],
                    w_pct=ci["w"],
                    h_pct=ci["h"],
                    from_name=from_name,
                    to_name=to_name,
                    content_layer=content_layer,
                    level=item_level + 1,
                    text="",
                    parent_id=table_rect_id,
                    score=score,
                )
            )

        # Cell text — an independent content child, INDEPENDENT of row/column
        # placement. The interface's emitTable then assigns it to whichever
        # origin cell its bbox overlaps most, which recovers the (r, c) home
        # from the geometry alone. Empty-text cells are skipped so the
        # interface doesn't render a stack of invisible text rects.
        cell_text = getattr(cell, "text", "") or ""
        if cell_text.strip():
            out.append(
                _make_rect_result(
                    ls_label="text",
                    x_pct=ci["x"],
                    y_pct=ci["y"],
                    w_pct=ci["w"],
                    h_pct=ci["h"],
                    from_name=from_name,
                    to_name=to_name,
                    content_layer=content_layer,
                    level=item_level + 1,
                    text=cell_text,
                    parent_id=table_rect_id,
                    score=score,
                )
            )

    return out


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


def _make_merge_polyline(
    *,
    region_ids: List[str],
    rect_by_id: Dict[str, Dict[str, Any]],
    from_name: str,
    to_name: str,
    score: Optional[float] = None,
    level: int = 1,
) -> Optional[Dict[str, Any]]:
    """Build one ``merge`` polyline connecting the given region ids.

    Returns ``None`` when fewer than 2 anchored region ids remain (a merge
    with one endpoint carries no information the annotator can act on).
    Missing rects are filtered — same reasoning as the caption/footnote
    emitter: a dangling polyline pointing at nothing is worse than no
    polyline at all.
    """
    kept: List[str] = []
    for rid in region_ids:
        if rid in rect_by_id and rid not in kept:
            kept.append(rid)
    if len(kept) < 2:
        return None
    points: List[List[float]] = []
    for rid in kept:
        cx, cy = _rect_center(rect_by_id[rid])
        points.append([round(cx, 4), round(cy, 4)])
    out: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "from_name": from_name,
        "to_name": to_name,
        "type": "polygonlabels",
        "origin": "prediction",
        "value": {
            "points": points,
            "polygonlabels": ["merge"],
            "connectedRegions": kept,
            "level": max(1, min(100, int(level) if level else 1)),
            "validationErrors": [],
            "parentId": None,
            "closed": False,
        },
    }
    if score is not None:
        out["score"] = score
    return out


def _emit_multi_prov_merges(
    *,
    multi_prov_groups: List[List[str]],
    rect_by_id: Dict[str, Dict[str, Any]],
    from_name: str,
    to_name: str,
    score: Optional[float] = None,
    level: int = 1,
) -> List[Dict[str, Any]]:
    """Emit a ``merge`` polyline for every multi-prov Docling item.

    Docling's layout model uses TWO shapes to say "one logical element split
    across columns / pages": (1) an InlineGroup wrapping multiple items, and
    (2) a single item with multiple ``prov`` entries (typical for a paragraph
    that wraps between two columns of the same page). This helper handles
    case (2); ``_emit_inline_group_merges`` handles case (1). Both emit the
    same ``merge`` polyline shape, so the DocLang round-trip is identical
    regardless of which Docling representation triggered it.
    """
    out: List[Dict[str, Any]] = []
    for group in multi_prov_groups:
        poly = _make_merge_polyline(
            region_ids=group,
            rect_by_id=rect_by_id,
            from_name=from_name,
            to_name=to_name,
            score=score,
            level=level,
        )
        if poly is not None:
            out.append(poly)
    return out


def _emit_inline_group_merges(
    doc: DoclingDocument,
    *,
    ref_to_id: Dict[str, str],
    rect_by_id: Dict[str, Dict[str, Any]],
    from_name: str,
    to_name: str,
    score: Optional[float] = None,
    level: int = 1,
) -> List[Dict[str, Any]]:
    """Emit ``merge`` polylines for every Docling ``InlineGroup``.

    An ``InlineGroup`` is Docling's representation of inline text runs that
    the layout model determined belong to ONE logical element split across
    column / page / line breaks (e.g. a paragraph that wraps between two
    columns). The interface's DocLang emitter treats a ``merge`` polyline
    exactly the same way — connected rects form one logical element with a
    shared ``thread_id`` — so mapping InlineGroup → merge polyline round-trips
    the split-element structure that would otherwise be lost between predict
    and manual annotation.

    Skips groups whose children don't map to any emitted rect (typically
    because a content-layer filter dropped the children upstream), and skips
    groups that resolve to fewer than 2 anchored rects (a merge with one
    endpoint carries no information the annotator can act on).

    Non-inline ``GroupItem`` kinds (SECTION / CHAPTER / SLIDE / etc.) are
    NOT emitted here — those are semantic containers, not visual-merge
    hints, so overloading the ``merge`` label would drown the annotator in
    false positives. If we ever need to surface them, they belong on the
    separate ``group`` polyline path type (which is a different design
    conversation about what "group" means when the annotator sees it).
    """
    groups = getattr(doc, "groups", None) or ()
    out: List[Dict[str, Any]] = []
    for group in groups:
        # Duck-typed check: match on the GroupLabel value rather than the
        # concrete InlineGroup class so SimpleNamespace test fixtures don't
        # have to import the real docling_core type. The label is the
        # discriminant Docling uses in its own group.label field, so this
        # is not a workaround — it IS the semantic check.
        if getattr(group, "label", None) != GroupLabel.INLINE:
            continue
        children = getattr(group, "children", None) or ()
        child_ids: List[str] = []
        for ref in children:
            cref = getattr(ref, "cref", None)
            rid = ref_to_id.get(cref) if isinstance(cref, str) else None
            if rid and rid in rect_by_id and rid not in child_ids:
                child_ids.append(rid)
        poly = _make_merge_polyline(
            region_ids=child_ids,
            rect_by_id=rect_by_id,
            from_name=from_name,
            to_name=to_name,
            score=score,
            level=level,
        )
        if poly is not None:
            out.append(poly)
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
      * When ``include_table_structure`` is enabled: per-table children the
        interface's ``emitTable`` walker expects — ``table_row`` strips
        (full table width, at each grid row's Y range), ``table_column``
        strips (full table height, at each grid column's X range),
        ``table_merged_cell`` overlays at every cell whose ``row_span`` or
        ``col_span`` > 1, semantic overlays (``column_header`` /
        ``row_header`` / ``row_section``) at their per-cell geometry, and
        ``text`` content children at each non-empty cell's bbox. All child
        rects carry ``parentId`` set to the enclosing table's region id.
        Rows / columns / merged / semantic overlays / cell text CO-EXIST
        on the same underlying geometry — see ``_emit_table_structure`` for
        the full contract.
      * When ``include_reading_order`` is enabled: one ``polygonlabels`` per
        page tracing the centroids of that page's items in Docling's
        iteration order, labeled ``reading_order``.
      * When ``include_relations`` is enabled: ``to_caption`` / ``to_footnote``
        / ``to_value`` 2-point ``polygonlabels`` for every ``FloatingItem``
        caption/footnote ref and every ``KeyValueItem`` / ``FormItem``
        ``TO_VALUE`` graph link, provided both endpoints were emitted as
        rects (endpoints filtered out by ``content_layers`` are silently
        dropped — no dangling links). Also emits ``merge`` polylines in both
        cases Docling uses to represent one logical element split across
        columns / pages / line breaks: multi-prov items (one NodeItem with
        several bboxes on the same page — typical for a paragraph wrapping
        between two columns) AND ``InlineGroup``s (a wrapper around several
        items). The interface's DocLang emitter maps a ``merge`` polyline
        to a shared ``thread_id`` so the split-element structure round-trips.

    Multi-prov items ALWAYS emit one rect per prov (not just prov[0]) —
    dropping the extra provs used to silently lose the second column of
    every cross-column paragraph, showing up as "missing text box" in the
    UI. The primary (first) prov's rect is the reading-order anchor and
    the target for caption/footnote/to_value links pointed at the item.

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
    # NodeItem.self_ref -> region id of the PRIMARY (first) prov's rect, so the
    # relations pass can link back to a stable handle even when an item has
    # multiple provs (see multi_prov_groups below). Rects without a self_ref
    # (raw fixtures in tests, top-level items with the attr set to None) simply
    # don't appear in the map and any relation targeting them silently no-ops.
    ref_to_id: Dict[str, str] = {}
    rect_by_id: Dict[str, Dict[str, Any]] = {}
    # Items whose prov[] has >1 entry on the requested page(s) render as one
    # rect per prov PLUS a shared ``merge`` polyline connecting them. This is
    # Docling's own model for "one logical element split across columns /
    # pages" — the layout model emits a single NodeItem with multiple provs
    # rather than a group. The interface's DocLang emitter turns a merge
    # polyline into a shared ``thread_id`` on each member's ``<location>``,
    # which is exactly the round-trip we want. Without this pass, the
    # ``prov_index=0`` code path silently dropped the second half of every
    # cross-column paragraph — visible in the UI as "missing text box".
    multi_prov_groups: List[List[str]] = []

    for item, level in doc.iterate_items(**iter_kw):
        if not item.prov:
            continue
        # Which provenance entries apply? An item straddling a page break
        # carries one prov per page, and iterate_items(page_no=N) yields it if
        # *any* of them is on page N — so we must measure the prov(s) actually
        # on the requested page and emit ONE rect per prov, not just prov[0].
        # A single-column single-page item just falls out of this loop with
        # one rect, identical to the old single-prov behavior.
        if page_no is not None:
            prov_indices = [i for i, p in enumerate(item.prov) if p.page_no == page_no]
        else:
            prov_indices = list(range(len(item.prov)))
        if not prov_indices:
            continue

        ls_label = _ls_label_for_item(item)
        layer = _content_layer_to_ls(getattr(item, "content_layer", ContentLayer.BODY))
        if layer not in LS_CONTENT_LAYERS:
            layer = "BODY"
        item_level = max(1, min(100, int(level) if level else 1))
        item_text = _item_text(item)
        picture_type = _picture_type(item, ls_label)

        # Emit one rect per applicable prov. All rects share the item's label,
        # text, level, content layer, and picture_type — they're literally the
        # same logical element in different regions, so they must not disagree
        # about what they are. Skips (bbox off-page, degenerate size, etc.)
        # come from _bbox_to_percent_rect; a partial skip is fine as long as
        # at least one prov survived to anchor relations against.
        per_prov_rects: List[Dict[str, Any]] = []
        per_prov_pages: List[int] = []
        for pi in prov_indices:
            rect = _bbox_to_percent_rect(doc, item, prov_index=pi)
            if rect is None:
                continue
            x_pct, y_pct, w_pct, h_pct, p_no = rect
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
                    "picture_type": picture_type,
                    "text": item_text,
                    "parentId": None,
                },
            }
            if score is not None:
                result["score"] = score
            results.append(result)
            rect_by_id[region_id] = result
            per_prov_rects.append(result)
            per_prov_pages.append(p_no)

        if not per_prov_rects:
            continue

        # Primary rect = first successful prov. Docling emits provs in
        # reading order (top-to-bottom, left-to-right for LTR scripts), so
        # the first prov is where a reader would enter the item. That makes
        # it the natural anchor for the reading-order polyline and for
        # caption/footnote/to_value links that target this item.
        primary = per_prov_rects[0]
        primary_page = per_prov_pages[0]
        self_ref = getattr(item, "self_ref", None)
        if isinstance(self_ref, str) and self_ref:
            ref_to_id[self_ref] = primary["id"]

        if include_reading_order:
            pv = primary["value"]
            cx = float(pv["x"]) + float(pv["width"]) / 2.0
            cy = float(pv["y"]) + float(pv["height"]) / 2.0
            reading_centers[primary_page].append((primary["id"], cx, cy))

        # Multi-prov item → record for a ``merge`` polyline at the end. Emit
        # only when we actually kept ≥2 rects (a partial skip could leave a
        # multi-prov item with just one rect, which needs no merge).
        if len(per_prov_rects) > 1:
            multi_prov_groups.append([r["id"] for r in per_prov_rects])

        # Table structure: rebuild the interface's expected overlays from the
        # docling TableData grid. See :func:`_emit_table_structure` for the
        # full contract; TL;DR: emit table_row / table_column strips, plus
        # merged-cell + semantic-role overlays, plus one `text` content child
        # per non-empty cell. The child rects are intentionally NOT swept
        # into the reading-order polyline: the reading order sequences top-
        # level flow, and a level-2 reading order inside a cell is a human
        # affordance that we don't fabricate. Tables in practice never
        # straddle pages, so anchoring the structure to the primary rect
        # covers the shape without needing to duplicate cells across provs.
        if include_table_structure and ls_label == "table":
            children = _emit_table_structure(
                doc,
                item,
                table_rect=primary,
                table_page_no=primary_page,
                from_name=from_name,
                to_name=to_name,
                content_layer=layer,
                item_level=item_level,
                score=score,
            )
            for child in children:
                results.append(child)
                rect_by_id[child["id"]] = child

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
        # Multi-prov items (one NodeItem, several bboxes on the same page)
        # emit their merge polyline BEFORE InlineGroup merges. Order only
        # matters cosmetically — the interface renders polylines in emit
        # order — but keeping multi-prov first mirrors how they were
        # collected during the main iteration.
        results.extend(
            _emit_multi_prov_merges(
                multi_prov_groups=multi_prov_groups,
                rect_by_id=rect_by_id,
                from_name=from_name,
                to_name=to_name,
                score=score,
                level=reading_order_level,
            )
        )
        results.extend(
            _emit_inline_group_merges(
                doc,
                ref_to_id=ref_to_id,
                rect_by_id=rect_by_id,
                from_name=from_name,
                to_name=to_name,
                score=score,
                level=reading_order_level,
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
