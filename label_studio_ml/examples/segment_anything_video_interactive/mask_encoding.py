"""Mask-to-result conversion for Label Studio control tags.

Each helper takes a binary 2D numpy mask (H, W) and returns a Label Studio
`value` payload for the corresponding control tag type. Bitmask output is a
PNG data URL; geometry outputs (rectanglelabels, polygonlabels) are in percent
coordinates.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


def mask_to_bitmap_png_base64(mask: np.ndarray) -> str:
    """Encode a binary mask as a PNG data URL payload (no `data:` prefix)."""
    bool_mask = (mask > 0).astype(np.uint8) * 255
    img = Image.fromarray(bool_mask, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def mask_to_bbox_percent(mask: np.ndarray) -> Optional[Dict[str, float]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    h, w = mask.shape[:2]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return {
        "x": round(float(x_min) / w * 100.0, 4),
        "y": round(float(y_min) / h * 100.0, 4),
        "width": round(float(x_max - x_min + 1) / w * 100.0, 4),
        "height": round(float(y_max - y_min + 1) / h * 100.0, 4),
    }


def mask_to_polygons_percent(mask: np.ndarray, simplify_px: float = 1.5) -> List[List[List[float]]]:
    """Return a list of polygons (each a list of [x%, y%] vertices)."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h, w = mask.shape[:2]
    polygons: List[List[List[float]]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        approx = cv2.approxPolyDP(contour, simplify_px, True)
        pts = [
            [round(float(p[0][0]) / w * 100.0, 4), round(float(p[0][1]) / h * 100.0, 4)]
            for p in approx
        ]
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons
