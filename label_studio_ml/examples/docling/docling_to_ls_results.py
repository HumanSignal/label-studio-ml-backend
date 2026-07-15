"""Map DoclingDocument items to canonical Label Studio result entries.

The Docling Interface (``docling-ls-implementation/docling_interface.jsx``,
a HumanSignal Interfaces project) reads predictions through its
``parseResults`` function and expects canonical Label Studio result shapes.
This module emits the two that Docling can populate from a converted document:

  * ``rectanglelabels`` for layout regions,
  * ``polygonlabels`` for the reading-order polyline.

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


def _bbox_to_percent_rect(
    doc: DoclingDocument,
    item: NodeItem,
    prov_index: int = 0,
) -> Optional[Tuple[float, float, float, float, int]]:
    """Return ``(x%, y%, width%, height%, page_no)`` in top-left page raster coordinates.

    Top-left / percentage coordinates match the interface's spatial-region
    format, so predictions and manual edits share the same coordinate
    convention and round-trip through the same code paths.

    The rect is clipped to the page: a bbox that overhangs an edge is trimmed to
    the page boundary rather than keeping its full extent, so ``x + width`` and
    ``y + height`` always stay within 0–100. A bbox that lies entirely off the
    page clips to nothing and returns ``None``.
    """
    if not item.prov or prov_index >= len(item.prov):
        return None
    prov = item.prov[prov_index]
    page = doc.pages.get(prov.page_no)
    if page is None:
        return None

    # scale_to_size divides by old_size, so page.size must be non-degenerate too.
    if not page.size or not page.size.width or not page.size.height:
        return None
    target_size = _page_raster_size(page)
    if target_size is None:
        return None

    bbox_tl = prov.bbox.to_top_left_origin(page_height=page.size.height)
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
    return (x0, y0, round(x1 - x0, _PCT_DIGITS), round(y1 - y0, _PCT_DIGITS), prov.page_no)


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
    ``original_height`` / ``image_rotation`` to every entry, since Label Studio
    carries those per result rather than on the prediction as a whole. Use
    :func:`page_raster_size` for dimensions consistent with these percentages.

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
