"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:
"""

import pytest
import json

from label_studio_ml.utils import compare_nested_structures
from model import YOLO
from .test_common import client, load_file, TEST_DIR
from unittest import mock


label_configs = [
    # test 1: one control tag with video timeline labels
    """
    <View>
       <TimelineLabels name="videoLabels" toName="video">
         <Label value="Car" predicted_values="racer, cab"/>
         <Label value="Airplane" background="red"/>
       </TimelineLabels>
       <Video name="video" value="$video" framerate="25.0" height="400"/>
    </View>
    """,
]

tasks = [
    # test 1
    {"data": {"video": "tests/opossum_snow_short.mp4"}},
]

yolo_results = [
    None # load_file(TEST_DIR + "/timeline_labels.pickle")
]

expected = [
    # test 1: one control tag with rectangle labels
    None # load_file(TEST_DIR + "/timeline_labels_1.json"),
]


@pytest.mark.skip(reason="Not yet implemented")
@pytest.mark.parametrize(
    "label_config, task, yolo_result, expect",
    zip(label_configs, tasks, yolo_results, expected),
)
def test_timelinelabels_predict(client, label_config, task, yolo_result, expect):
    data = {"schema": label_config, "project": "42"}
    response = client.post(
        "/setup", data=json.dumps(data), content_type="application/json"
    )
    assert response.status_code == 200, "Error while setup: " + str(response.content)

    data = {"tasks": [task], "label_config": label_config}

    # mock yolo model.track, because it takes too different results from run to run
    # also track is a heavy operation, and it might take too much time for tests
    if yolo_result:
        with mock.patch("ultralytics.YOLO.track") as mock_yolo:
            mock_yolo.return_value = yolo_result
            response = client.post(
                "/predict", data=json.dumps(data), content_type="application/json"
            )

    # don't mock if no yolo_result
    else:
        response = client.post(
            "/predict", data=json.dumps(data), content_type="application/json"
        )

    assert response.status_code == 200, "Error while predict"
    data = response.json
    compare_nested_structures(data["results"], expect, rel=1e-3)

