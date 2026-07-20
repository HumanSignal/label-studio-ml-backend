"""Unit tests for control-tag detection (BROS-1418).

These run without torch/cv2: detection lives in the dependency-free
``control_detect`` module and is exercised against a real
``label_studio_sdk`` LabelInterface.

Run with: pytest test_control_detect.py -v
"""

import pytest
from label_studio_sdk.label_interface import LabelInterface

from control_detect import detect_control

# Standalone <VideoVector> + separate <Labels> — the config from BROS-1418.
VIDEOVECTOR_PLUS_LABELS = """
<View>
  <VideoVector name="videovec" toName="video" closable="true"/>
  <Labels name="labels" toName="video">
    <Label value="Human" background="pink"/>
    <Label value="Reptile" background="#7CFC00"/>
  </Labels>
  <Video name="video" value="$video"/>
</View>
""".strip()

# Combined <VideoVectorLabels> — the config that already worked.
VIDEOVECTORLABELS = """
<View>
  <VideoVectorLabels name="videolabels" toName="video">
    <Label value="Human" background="pink"/>
  </VideoVectorLabels>
  <Video name="video" value="$video"/>
</View>
""".strip()

VIDEORECTANGLE_PLUS_LABELS = """
<View>
  <VideoRectangle name="box" toName="video"/>
  <Labels name="labels" toName="video">
    <Label value="Human"/>
  </Labels>
  <Video name="video" value="$video"/>
</View>
""".strip()

IMAGE_RECTANGLELABELS = """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="tag" toName="image">
    <Label value="object"/>
  </RectangleLabels>
</View>
""".strip()

NO_SUPPORTED_CONTROL = """
<View>
  <Text name="txt" value="$text"/>
  <Choices name="choice" toName="txt">
    <Choice value="a"/>
  </Choices>
</View>
""".strip()


def test_videovector_plus_labels_is_detected():
    """Regression for BROS-1418: a standalone <VideoVector> control (paired
    with a sibling <Labels>) must be detected as a Video control bound to the
    VideoVectorLabels capability — not raise (which surfaced as HTTP 500).
    """
    li = LabelInterface(VIDEOVECTOR_PLUS_LABELS)
    assert detect_control(li) == ("videovec", "video", "Video", "videovectorlabels")


def test_combined_videovectorlabels_still_detected():
    li = LabelInterface(VIDEOVECTORLABELS)
    assert detect_control(li) == ("videolabels", "video", "Video", "videovectorlabels")


def test_videorectangle_still_detected():
    li = LabelInterface(VIDEORECTANGLE_PLUS_LABELS)
    assert detect_control(li) == ("box", "video", "Video", "videorectangle")


def test_image_control_still_detected():
    li = LabelInterface(IMAGE_RECTANGLELABELS)
    assert detect_control(li) == ("tag", "image", "Image", "rectanglelabels")


def test_no_supported_control_raises():
    li = LabelInterface(NO_SUPPORTED_CONTROL)
    with pytest.raises(ValueError):
        detect_control(li)
