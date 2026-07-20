"""Map DoclingDocument items to canonical Label Studio result entries.

The Docling Interface (``docling-ls-implementation/docling_interface.jsx``,
a HumanSignal Interfaces project) reads predictions through its
``parseResults`` function and expects canonical Label Studio result shapes:

  * ``rectanglelabels`` for layout regions,
  * ``polygonlabels`` for polylines (reading order / merge / group / link
    relations rendered as paths),
  * ``textarea`` for the doclang XML snapshot,
  * ``relation`` for region-to-region links.

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


def _bbox_page_to_percent(
    doc: DoclingDocument,
    bbox: Any,
    page_no: int,
) -> Optional[Tuple[float, float, float, float, int]]:
    """Convert a raw docling ``BoundingBox`` on a given page to percent coords.

    Split out from ``_bbox_to_percent_rect`` so callers that don't have a full
    ``NodeItem.prov`` — table cells (``TableCell.bbox``), graph cells
    (``GraphCell.prov``), any future case — can share the exact same
    top-left / raster-scale / clamp pipeline. Returns ``None`` when the page
    is missing or the target raster has zero extent.
    """
    if bbox is None:
        return None
    page = doc.pages.get(page_no)
    if page is None:
        return None
    try:
        bbox_tl = bbox.to_top_left_origin(page_height=page.size.height)
    except Exception:
        return None
    target_size = page.image.size if page.image is not None else page.size
    if not target_size.width or not target_size.height:
        return None
    scaled = bbox_tl.scale_to_size(old_size=page.size, new_size=target_size)
    w_px, h_px = target_size.width, target_size.height
    x_pct = max(0.0, min(100.0, scaled.l / w_px * 100.0))
    y_pct = max(0.0, min(100.0, scaled.t / h_px * 100.0))
    width_pct = max(0.0, min(100.0, scaled.width / w_px * 100.0))
    height_pct = max(0.0, min(100.0, scaled.height / h_px * 100.0))
    return (x_pct, y_pct, width_pct, height_pct, page_no)


def _bbox_to_percent_rect(
    doc: DoclingDocument,
    item: NodeItem,
    prov_index: int = 0,
) -> Optional[Tuple[float, float, float, float, int]]:
    """Return ``(x%, y%, width%, height%, page_no)`` in top-left page raster coordinates.

    Top-left / percentage coordinates match the interface's spatial-region
    format, so predictions and manual edits share the same coordinate
    convention and round-trip through the same code paths.
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


def _parse_content_layers(raw: Optional[str]) -> Optional[Set[ContentLayer]]:
    if not raw:
        return None
    out: Set[ContentLayer] = set()
    for part in raw.lower().split(","):
        part = part.strip()
        if part == "body":
            out.add(ContentLayer.BODY)
        elif part == "furniture":
            out.add(ContentLayer.FURNITURE)
        elif part == "background":
            out.add(ContentLayer.BACKGROUND)
        elif part == "invisible":
            out.add(ContentLayer.INVISIBLE)
        elif part == "notes":
            out.add(ContentLayer.NOTES)
    return out or None


def _table_cell_label(cell: Any) -> str:
    """Pick the interface label for a docling ``TableCell``.

    The interface distinguishes column_header / row_header / row_section /
    merged / plain cell — matching what an annotator would draw manually. If
    a cell is both a header AND a merge, header wins (it's the more
    informative label; the merged geometry is still preserved as a single
    rectangle rather than N sub-cells).
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


def _rect_center(rect: Dict[str, Any]) -> Tuple[float, float]:
    """Return the (cx%, cy%) center of a ``rectanglelabels`` value block."""
    v = rect.get("value") or {}
    x = float(v.get("x", 0) or 0)
    y = float(v.get("y", 0) or 0)
    w = float(v.get("width", 0) or 0)
    h = float(v.get("height", 0) or 0)
    return x + w / 2.0, y + h / 2.0


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
    marker: Optional[str] = None,
    list_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a canonical ``rectanglelabels`` result envelope.

    Extracted so table cells / caption / footnote / graph-cell rects all
    share the exact same schema as the top-level items and don't drift
    from the ``parseResults`` contract. Underscore-prefixed keys are
    deliberately omitted from ``value`` (the interface's serializer
    rejects them — see test_no_underscore_prefixed_keys_in_value).
    """
    rid = region_id or str(uuid.uuid4())
    value: Dict[str, Any] = {
        "x": round(x_pct, 4),
        "y": round(y_pct, 4),
        "width": round(w_pct, 4),
        "height": round(h_pct, 4),
        "rotation": 0,
        "rectanglelabels": [ls_label],
        "content_layer": content_layer,
        "level": max(1, min(100, int(level) if level else 1)),
        "picture_type": picture_type,
        "text": text or "",
        "parentId": parent_id,
        # v20.4-era list_item extras (interface parseResults reads these).
        "marker": marker,
        "list_type": list_type,
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

    Used for the ``to_caption`` / ``to_footnote`` / ``to_value`` labels the
    interface reads (see ``LINK_RESTRICTIONS`` in ``docling_interface.jsx``).
    Points are the geometric centers of each endpoint's rect; the interface
    snaps points to their enclosing rects on next drag anyway, but drawing
    them at the center gives a sensible initial visual.
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

    Guarded with ``getattr`` fallbacks so a partially-populated doc (e.g. a
    minimal test fixture that only sets ``tables``) doesn't blow up.
    """
    for attr in ("tables", "pictures", "key_value_items", "form_items"):
        for it in getattr(doc, attr, None) or ():
            yield it


def docling_document_to_ls_results(
    doc: DoclingDocument,
    *,
    page_no: Optional[int] = None,
    include_reading_order: bool = True,
    reading_order_level: int = 1,
    include_table_structure: bool = True,
    include_relations: bool = True,
    content_layers: Optional[str] = None,
    from_name: str = "docling",
    to_name: str = "docling",
    score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build canonical Label Studio prediction results.

    Output is a flat list ready to drop into ``PredictionValue.result``. Each
    entry is a complete envelope (``id``, ``from_name``, ``to_name``, ``type``,
    ``value``) — the caller still needs to attach ``original_width`` /
    ``original_height`` / ``image_rotation`` per Label Studio's prediction
    format requirements (the model envelope, not the per-result envelope).

    What lands in the output:

      * One ``rectanglelabels`` per Docling layout item with a bounding box.
      * One ``rectanglelabels`` per structural cell of every ``TableItem``
        (``table_cell`` / ``column_header`` / ``row_header`` / ``row_section``
        / ``table_merged_cell`` — pick per cell), with ``parentId`` set to
        the enclosing table's region id (``include_table_structure`` gate).
      * One ``reading_order`` ``polygonlabels`` per page tracing item
        centroids in Docling's iteration order. Required by the interface
        — without at least one ``reading_order`` path the DocLang preview
        pane renders nothing (``include_reading_order`` gate, default on).
      * ``to_caption`` / ``to_footnote`` 2-point ``polygonlabels`` for every
        ``FloatingItem`` (table / picture / kv / form) whose ``captions``
        or ``footnotes`` refs resolve to an emitted rect
        (``include_relations`` gate).
      * ``to_value`` 2-point ``polygonlabels`` for every ``TO_VALUE`` link
        in a ``KeyValueItem`` / ``FormItem`` graph whose source and target
        cells resolve to emitted rects via ``GraphCell.item_ref``.

    Everything above except the top-level ``rectanglelabels`` can be disabled
    by the caller (``include_reading_order`` / ``include_table_structure`` /
    ``include_relations``). Defaults are all on because that's what makes the
    interface render the prediction usefully out of the box; the previous
    default of "no reading order, no cells, no relations" produced a wall of
    disconnected rectangles that lost most of the docling structure.
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
    # NodeItem.self_ref -> region id, so second-pass relation emission can
    # link back to rects we already emitted (captions/footnotes/graph cells
    # all reference source items by cref).
    ref_to_id: Dict[str, str] = {}
    # Region id -> full rect envelope, so the link emitter can compute
    # centers for the polyline endpoints without a second search.
    rect_by_id: Dict[str, Dict[str, Any]] = {}
    reading_centers: Dict[int, List[Tuple[str, float, float]]] = defaultdict(list)

    def _emit_and_track_rect(item: NodeItem, ls_label: str, r: Dict[str, Any]) -> None:
        results.append(r)
        rect_by_id[r["id"]] = r
        ref = getattr(item, "self_ref", None)
        if isinstance(ref, str) and ref:
            ref_to_id[ref] = r["id"]

    for item, level in doc.iterate_items(**iter_kw):
        if not item.prov:
            continue
        rect = _bbox_to_percent_rect(doc, item, prov_index=0)
        if rect is None:
            continue
        x_pct, y_pct, w_pct, h_pct, p_no = rect
        if page_no is not None and p_no != page_no:
            continue

        ls_label = _ls_label_for_item(item)
        layer = _content_layer_to_ls(getattr(item, "content_layer", ContentLayer.BODY))
        if layer not in LS_CONTENT_LAYERS:
            layer = "BODY"

        result = _make_rect_result(
            ls_label=ls_label,
            x_pct=x_pct,
            y_pct=y_pct,
            w_pct=w_pct,
            h_pct=h_pct,
            from_name=from_name,
            to_name=to_name,
            content_layer=layer,
            level=int(level) if level else 1,
            picture_type=_picture_type(item, ls_label),
            text=_item_text(item),
            score=score,
        )
        _emit_and_track_rect(item, ls_label, result)
        parent_region_id = result["id"]

        if include_reading_order:
            cx = x_pct + w_pct / 2.0
            cy = y_pct + h_pct / 2.0
            reading_centers[p_no].append((parent_region_id, cx, cy))

        # Table structure: emit one child rect per cell that has a bbox.
        # Cells share the parent table's page (docling always puts the whole
        # table on a single page) so we reuse p_no rather than trusting a
        # potentially-missing prov on the cell itself.
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
                    # Cells sit one level deeper than their table in the
                    # interface's sub-annotation tree; +1 from the parent's
                    # effective iteration level.
                    level=(int(level) if level else 1) + 1,
                    text=getattr(cell, "text", "") or "",
                    parent_id=parent_region_id,
                    score=score,
                )
                # Table cells are structural — do NOT include them in the
                # reading-order polyline (which sequences the top-level
                # document flow). The interface allows level-2 reading-
                # orders inside a table cell for cell contents, but that's
                # a human affordance; we don't fabricate one.
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
    # RefItem shape (cref) — the common case.
    cref = getattr(ref, "cref", None)
    if isinstance(cref, str) and cref:
        try:
            resolved = ref.resolve(doc)
        except Exception:
            resolved = None
        if resolved is not None:
            resolved_ref = getattr(resolved, "self_ref", None)
    # Direct NodeItem-like object with a self_ref attribute.
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
    ``TO_VALUE`` pairs. Every link needs BOTH endpoints to have been
    emitted as rects (via the main iteration pass) — otherwise the
    interface would render a dangling polyline pointing at nothing. When
    an endpoint is missing we silently drop the link.
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

        # to_value links: only meaningful on items with a graph.
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


def parse_image_data_key(label_config: Optional[str]) -> str:
    """Return the task.data key the labeling iframe reads from.

    The Docling Interface's default ``params.imageField`` is ``image``. ML
    backends should default to ``image`` and fall back through the same chain
    (``image``, ``url``, ``ocr``, ``$undefined``, ``$undefined$``,
    ``undefined``) before giving up.
    """
    return "image"
