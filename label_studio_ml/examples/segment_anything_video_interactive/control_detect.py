"""Control-tag detection for the SAM2 interactive backend.

Pure config-parsing logic, free of heavy deps (torch/cv2), so it can be
unit-tested in isolation against a ``label_studio_sdk`` LabelInterface — see
``test_control_detect.py``. ``model.py`` imports ``detect_control`` from here.
"""

from __future__ import annotations

from typing import Tuple

# Image control tags, in priority order.
_IMAGE_CONTROLS = ("BitmaskLabels", "RectangleLabels", "PolygonLabels", "VectorLabels")

# Video control tags, in priority order. The self-labeled combined tags come
# first; `VideoVector` is the standalone (separated) vector control paired with
# a sibling `<Labels>` tag and is matched last.
_VIDEO_CONTROLS = ("VideoVectorLabels", "VideoRectangle", "VideoVector")

_CONTROL_TO_TYPE = {
    "BitmaskLabels": "bitmap",
    "RectangleLabels": "rectanglelabels",
    "PolygonLabels": "polygonlabels",
    "VectorLabels": "vectorlabels",
    "VideoRectangle": "videorectangle",
    "VideoVectorLabels": "videovectorlabels",
    # Standalone `<VideoVector>` (paired with a sibling `<Labels>`) emits the
    # same type as the combined tag: the result pipeline is identical (a mask
    # the FE turns into vector geometry) and the FE binds it to the
    # `VideoVectorLabels` capability. Without this, a VideoVector+Labels config
    # raised ValueError → HTTP 500 (BROS-1418).
    "VideoVector": "videovectorlabels",
}


def control_to_type(control: str) -> str:
    return _CONTROL_TO_TYPE[control]


def detect_control(label_interface) -> Tuple[str, str, str, str]:
    """Return (from_name, to_name, object_type, control_type).

    `object_type` ∈ {'Image', 'Video'}, `control_type` is the xml-lowercased
    type we'll emit.
    """
    for candidate in _IMAGE_CONTROLS:
        try:
            from_name, to_name, _ = label_interface.get_first_tag_occurence(candidate, "Image")
            return from_name, to_name, "Image", control_to_type(candidate)
        except Exception:
            pass
    for candidate in _VIDEO_CONTROLS:
        try:
            from_name, to_name, _ = label_interface.get_first_tag_occurence(candidate, "Video")
            return from_name, to_name, "Video", control_to_type(candidate)
        except Exception:
            pass
    raise ValueError("no supported control tag found in label config")
