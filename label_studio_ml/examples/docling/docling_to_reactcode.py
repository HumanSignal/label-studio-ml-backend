"""Map DoclingDocument items to Label Studio ReactCode region payloads (percent coordinates)."""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from docling_core.types.doc.document import ContentLayer, DoclingDocument, NodeItem
from docling_core.types.doc.labels import DocItemLabel

logger = logging.getLogger(__name__)

# DoclingDocument labels -> names used in docling_labeling_config.xml (LABEL_CATEGORIES)
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
    """Return (x%, y%, width%, height%, page_no) in top-left page raster coordinates."""
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


def docling_document_to_reactcode_regions(
    doc: DoclingDocument,
    *,
    page_no: Optional[int] = None,
    include_reading_order: bool = False,
    reading_order_level: int = 1,
    content_layers: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build rectangle (and optional reading_order polyline) payloads matching
    docling_labeling_config.xml: type rectangle/polyline, percent x/y/width/height, etc.
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

    rectangles: List[Dict[str, Any]] = []
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

        region_id = str(uuid.uuid4())
        payload: Dict[str, Any] = {
            "id": region_id,
            "type": "rectangle",
            "label": ls_label,
            "x": round(x_pct, 4),
            "y": round(y_pct, 4),
            "width": round(w_pct, 4),
            "height": round(h_pct, 4),
            "rotation": 0,
            "content_layer": layer,
            "level": max(1, min(100, int(level) if level else 1)),
            "picture_type": _picture_type(item, ls_label),
            "text": _item_text(item),
        }
        rectangles.append(payload)
        if include_reading_order:
            cx = x_pct + w_pct / 2.0
            cy = y_pct + h_pct / 2.0
            reading_centers[p_no].append((region_id, cx, cy))

    if include_reading_order:
        for p_no, chain in reading_centers.items():
            if len(chain) < 2:
                continue
            ids = [c[0] for c in chain]
            points = [[round(c[1], 4), round(c[2], 4)] for c in chain]
            rectangles.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "polyline",
                    "label": "reading_order",
                    "points": points,
                    "connectedRegions": ids,
                    "level": max(1, min(100, int(reading_order_level))),
                }
            )

    return rectangles


def parse_reactcode_tag_names(label_config: str) -> Tuple[str, str]:
    """Read ReactCode name= and toName= from raw XML."""
    if not label_config:
        return "docling", "docling"
    m = re.search(
        r"<ReactCode\b[^>]*\bname=[\"']([^\"']+)[\"'][^>]*\btoName=[\"']([^\"']+)[\"']",
        label_config,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1), m.group(2)
    m2 = re.search(r"<ReactCode\b[^>]*\bname=[\"']([^\"']+)[\"']", label_config, re.IGNORECASE | re.DOTALL)
    if m2:
        n = m2.group(1)
        return n, n
    return "docling", "docling"


def parse_image_data_key(label_config: str) -> str:
    """Read Image value=\"$key\" or value=\"$undefined\" -> task['data']['undefined']."""
    if not label_config:
        return "undefined"
    m = re.search(
        r"<Image\b[^>]*\bvalue=[\"']\$([^\"']+)[\"']",
        label_config,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1)
    return "undefined"
