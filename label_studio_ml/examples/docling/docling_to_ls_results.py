"""Map DoclingDocument items to canonical Label Studio result entries.

Why this exists
---------------
The original ``docling_to_reactcode.py`` produces region payloads wrapped in
the legacy ReactCode envelope (``type: "reactcode"``,
``value: { "reactcode": <payload> }``). That format only renders inside the
old ``<ReactCode>`` XML labeling config.

The current Docling Interface (HumanSignal Interfaces format,
``docling-ls-implementation/docling_interface.jsx``) reads predictions through
its ``parseResults`` function and expects canonical Label Studio result
shapes — ``rectanglelabels`` for rectangles, ``polygonlabels`` for polylines
(reading order / merge / group / link relations rendered as paths),
``textarea`` for the doclang XML snapshot, and ``relation`` for region-to-
region links. The interface's ``getResults`` uses the same shapes when
serializing manual annotations, so emitting canonical results from the ML
backend means predictions and human edits round-trip through the same code
paths.

This module is additive: ``docling_to_reactcode.py`` is retained for any
deployment still pointed at the old labeling config.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from docling_core.types.doc.document import ContentLayer, DoclingDocument, NodeItem
from docling_core.types.doc.labels import DocItemLabel

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


def _bbox_to_percent_rect(
    doc: DoclingDocument,
    item: NodeItem,
    prov_index: int = 0,
) -> Optional[Tuple[float, float, float, float, int]]:
    """Return ``(x%, y%, width%, height%, page_no)`` in top-left page raster coordinates.

    Mirrors the existing ReactCode converter so manual edits and predictions
    share the same coordinate convention.
    """
    if not item.prov or prov_index >= len(item.prov):
        return None
    prov = item.prov[prov_index]
    page = doc.pages.get(prov.page_no)
    if page is None:
        return None

    bbox_tl = prov.bbox.to_top_left_origin(page_height=page.size.height)
    target_size = page.image.size if page.image is not None else page.size
    if not target_size.width or not target_size.height:
        return None

    scaled = bbox_tl.scale_to_size(old_size=page.size, new_size=target_size)
    w_px, h_px = target_size.width, target_size.height
    x_pct = max(0.0, min(100.0, scaled.l / w_px * 100.0))
    y_pct = max(0.0, min(100.0, scaled.t / h_px * 100.0))
    width_pct = max(0.0, min(100.0, scaled.width / w_px * 100.0))
    height_pct = max(0.0, min(100.0, scaled.height / h_px * 100.0))
    return (x_pct, y_pct, width_pct, height_pct, prov.page_no)


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


def docling_document_to_ls_results(
    doc: DoclingDocument,
    *,
    page_no: Optional[int] = None,
    include_reading_order: bool = False,
    reading_order_level: int = 1,
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

    Returns:
      * ``rectanglelabels`` entries for every Docling layout item with a
        bounding box.
      * ``polygonlabels`` entries (one per page) when ``include_reading_order``
        is enabled — one polyline tracing the centroids of the page's items in
        Docling's iteration order, labeled ``reading_order``.
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
        item_level = max(1, min(100, int(level) if level else 1))

        region_id = str(uuid.uuid4())
        result: Dict[str, Any] = {
            "id": region_id,
            "from_name": from_name,
            "to_name": to_name,
            "type": "rectanglelabels",
            "origin": "prediction",
            "value": {
                "x": round(x_pct, 4),
                "y": round(y_pct, 4),
                "width": round(w_pct, 4),
                "height": round(h_pct, 4),
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

        if include_reading_order:
            cx = x_pct + w_pct / 2.0
            cy = y_pct + h_pct / 2.0
            reading_centers[p_no].append((region_id, cx, cy))

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

    return results


def parse_image_data_key(label_config: Optional[str]) -> str:
    """Return the task.data key the labeling iframe reads from.

    The new Interface's default ``params.imageField`` is ``image``. ML backends
    talking to the new interface should default to ``image`` and fall back
    through the same chain (``image``, ``url``, ``ocr``, ``$undefined``,
    ``$undefined$``, ``undefined``) before giving up.

    The legacy ReactCode XML parser remains in
    ``docling_to_reactcode.parse_image_data_key`` for old-config compatibility.
    """
    return "image"
