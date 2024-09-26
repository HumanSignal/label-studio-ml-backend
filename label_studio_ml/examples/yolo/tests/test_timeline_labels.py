"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:
"""

import os
import json
import pytest
import numpy as np

from label_studio_ml.utils import compare_nested_structures
from unittest import mock
from .test_common import client, load_file, TEST_DIR
from ..utils.converter import (
    convert_timelinelabels_to_probs,
    convert_probs_to_timelinelabels,
)
from ..control_models.timeline_labels import TimelineLabelsModel
from unittest.mock import MagicMock, patch
from label_studio_sdk.label_interface import LabelInterface
from label_studio_ml.model import LabelStudioMLBase

label_configs = [
    # test 1: one control tag with video timeline labels
    """
    <View>
       <TimelineLabels name="videoLabels" toName="video" model_score_threshold="0.01">
         <Label value="Car" predicted_values="snowmobile,racer,cab"/>
         <Label value="croquet_ball" background="red"/>
       </TimelineLabels>
       <Video name="video" value="$video" framerate="25.0" />
    </View>
    """,
]

tasks = [
    # test 1
    {"data": {"video": "tests/opossum_snow_short.mp4"}},
]

yolo_results = [None]  # load_file(TEST_DIR + "/timeline_labels.pickle")

expected = [
    # test 1: one control tag with rectangle labels
    [
        {
            "model_version": "yolo",
            "score": 0.10993397843318456,
            "result": [
                {
                    "from_name": "videoLabels",
                    "id": "0_1_15",
                    "score": 0.26830227896571157,
                    "to_name": "video",
                    "type": "timelinelabels",
                    "value": {
                        "ranges": [{"end": 15, "start": 1}],
                        "timelinelabels": ["Car"],
                    },
                },
                {
                    "from_name": "videoLabels",
                    "id": "2_23_25",
                    "score": 0.044929164151350655,
                    "to_name": "video",
                    "type": "timelinelabels",
                    "value": {
                        "ranges": [{"end": 25, "start": 23}],
                        "timelinelabels": ["Car"],
                    },
                },
                {
                    "from_name": "videoLabels",
                    "id": "1_19_29",
                    "score": 0.19226251965896649,
                    "to_name": "video",
                    "type": "timelinelabels",
                    "value": {
                        "ranges": [{"end": 29, "start": 19}],
                        "timelinelabels": ["croquet_ball"],
                    },
                },
                {
                    "from_name": "videoLabels",
                    "id": "3_30_31",
                    "score": 0.01777686830610037,
                    "to_name": "video",
                    "type": "timelinelabels",
                    "value": {
                        "ranges": [{"end": 31, "start": 30}],
                        "timelinelabels": ["Car"],
                    },
                },
                {
                    "from_name": "videoLabels",
                    "id": "4_34_34",
                    "score": 0.02639906108379364,
                    "to_name": "video",
                    "type": "timelinelabels",
                    "value": {
                        "ranges": [{"end": 34, "start": 34}],
                        "timelinelabels": ["Car"],
                    },
                },
            ],
        }
    ]
]


# @pytest.mark.skip(reason="Not yet implemented")
@pytest.mark.parametrize(
    "label_config, task, yolo_result, expect",
    zip(label_configs, tasks, yolo_results, expected),
)
def test_timelinelabels_simple(client, label_config, task, yolo_result, expect):
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


def test_convert_timelinelabels_to_probs():
    # Example usage:
    regions = [
        {
            "from_name": "videoLabels",
            "id": "0_9",
            "origin": "prediction",
            "to_name": "video",
            "type": "timelinelabels",
            "value": {"ranges": [{"end": 8, "start": 1}], "timelinelabels": ["Snow"]},
        },
        {
            "from_name": "videoLabels",
            "id": "10_15",
            "origin": "prediction",
            "to_name": "video",
            "type": "timelinelabels",
            "value": {"ranges": [{"end": 14, "start": 10}], "timelinelabels": ["Rain"]},
        },
        {
            "from_name": "videoLabels",
            "id": "0_9x",
            "origin": "prediction",
            "to_name": "video",
            "type": "timelinelabels",
            "value": {"ranges": [{"end": 14, "start": 14}], "timelinelabels": ["Snow"]},
        },
    ]
    label_map = {"Rain": 0, "Snow": 1}
    labels_array, used_labels = convert_timelinelabels_to_probs(regions, label_map)

    # Label Mapping
    expected_used_labels = {"Rain", "Snow"}

    # Labels Array
    expected_labels_array = np.array(
        [
            [0, 1],  # 1 frame
            [0, 1],  # 2 frame
            [0, 1],  # 3 frame
            [0, 1],  # 4 frame
            [0, 1],  # 5 frame
            [0, 1],  # 6 frame
            [0, 1],  # 7 frame
            [0, 1],  # 8 frame
            [0, 0],  # 9 frame
            [1, 0],  # 10 frame
            [1, 0],  # 11 frame
            [1, 0],  # 12 frame
            [1, 0],  # 13 frame
            [1, 1],  # 14 frame
        ]
    )

    print("Labels Array:\n", labels_array)
    print("Label Mapping:\n", used_labels)

    assert used_labels == expected_used_labels
    assert np.array_equal(labels_array, expected_labels_array)


def test_convert_probs_to_timelinelabels():
    # Example usage
    probs = [
        [0.8, 0.2],  # Frame 1: 'Rain' has a high probability
        [0.9, 0.1],  # Frame 2: 'Rain' is still active
        [0.6, 0.4],  # Frame 3: Both probabilities are near threshold
        [0.2, 0.7],  # Frame 4: 'Snow' becomes more active
        [0.1, 0.9],  # Frame 5: 'Snow' has a high probability
        [0.1, 0.1],  # Frame 6
        [0.9, 0.9],  # Frame 7
    ]
    # Regions => [1-3: Rain], [4-5: Snow], [7-7: Rain, Snow]

    label_map = {"Rain": 0, "Snow": 1}

    timeline_regions = convert_probs_to_timelinelabels(
        probs, label_map, "videoLabels", score_threshold=0.5
    )

    assert timeline_regions == [
        {
            "id": "0_1_3",
            "type": "timelinelabels",
            "value": {"ranges": [{"start": 1, "end": 3}], "timelinelabels": ["Rain"]},
            "to_name": "video",
            "from_name": "videoLabels",
            "score": 0.7666666666666667,
        },
        {
            "id": "1_4_5",
            "type": "timelinelabels",
            "value": {"ranges": [{"start": 4, "end": 5}], "timelinelabels": ["Snow"]},
            "to_name": "video",
            "from_name": "videoLabels",
            "score": 0.8,
        },
        {
            "id": "2_7_7",
            "type": "timelinelabels",
            "value": {"ranges": [{"start": 7, "end": 7}], "timelinelabels": ["Rain"]},
            "to_name": "video",
            "from_name": "videoLabels",
            "score": 0.9,
        },
        {
            "id": "3_7_7",
            "type": "timelinelabels",
            "value": {"ranges": [{"start": 7, "end": 7}], "timelinelabels": ["Snow"]},
            "to_name": "video",
            "from_name": "videoLabels",
            "score": 0.9,
        },
    ]

    # invert conversion
    labels_array, _ = convert_timelinelabels_to_probs(
        timeline_regions, label_map, max_frame=len(probs)
    )

    # convert probs to binary
    probs = np.array(probs) > 0.5

    # compare probs and labels_array
    assert np.array_equal(probs, labels_array)


def test_timelinelabels_trainable(client):
    # rootdir is label_studio_ml/examples/yolo
    path = "./models/timelinelabels-42-yolov8n-cls-videoLabels.pkl"
    if os.path.exists(path):
        os.remove(path)

    label_config = """
    <View>
         <TimelineLabels name="videoLabels" toName="video" 
            model_trainable="true"
            model_classifier_accuracy_threshold="1.0"
            model_classifier_f1_threshold="1.0"
            model_epoch="5000"
          >
            <Label value="Car"/>
            <Label value="croquet_ball" background="red"/>
        </TimelineLabels>
        <Video name="video" value="$video" framerate="25.0" />
    </View>
    """

    # setup
    data = {"schema": label_config, "project": "42"}
    response = client.post(
        "/setup", data=json.dumps(data), content_type="application/json"
    )
    assert response.status_code == 200, "Error while setup: " + str(response.content)

    # predict when model is not trained => 500 error is ok
    task = {"data": {"video": "tests/opossum_snow_short.mp4"}}
    data = {"tasks": [task], "label_config": label_config, "project": 42}
    response = client.post(
        "/predict", data=json.dumps(data), content_type="application/json"
    )
    assert (
        response.status_code == 500
    ), "It should be error because model is not yet trained"

    # train model
    data = load_file(TEST_DIR + "/test_timeline_labels_1.json")
    data["project"]["label_config"] = label_config
    data["action"] = "ANNOTATION_CREATED"
    response = client.post(
        "/webhook", data=json.dumps(data), content_type="application/json"
    )

    assert response.status_code == 201, "Error while fit: " + str(response.content)
    result = response.json["result"]
    assert (
        result["videoLabels"]["accuracy"] > 0.99
        or result["videoLabels"]["f1_score"] > 0.99
    )
    assert result["videoLabels"]["epoch"] > 5

    # predict again => 200
    task = {"data": {"video": "tests/opossum_snow_short.mp4"}}
    data = {"tasks": [task], "label_config": label_config, "project": 42}
    response = client.post(
        "/predict", data=json.dumps(data), content_type="application/json"
    )
    assert response.status_code == 200, "Error while predict: " + str(response.content)

    expected_result = {
        "results": [
            {
                "model_version": "yolo",
                "result": [
                    {
                        "from_name": "videoLabels",
                        "id": "0_1_10",
                        "score": 0.7321293950080872,
                        "to_name": "video",
                        "type": "timelinelabels",
                        "value": {
                            "ranges": [{"end": 10, "start": 1}],
                            "timelinelabels": ["Car"],
                        },
                    },
                    {
                        "from_name": "videoLabels",
                        "id": "1_5_15",
                        "score": 0.7476321404630487,
                        "to_name": "video",
                        "type": "timelinelabels",
                        "value": {
                            "ranges": [{"end": 15, "start": 5}],
                            "timelinelabels": ["croquet_ball"],
                        },
                    },
                ],
                "score": 0.7398807677355679,
            }
        ]
    }
    compare_nested_structures(response.json, expected_result, abs=0.2)


def test_timelinelabels_no_label_match():
    """
    Test that a ValueError is raised when the TimelineLabelsModel is in simple mode (model_trainable="false")
    and none of the labels from the labeling config match the YOLO model's labels.
    """

    # Create a label config with labels that do not match any YOLO labels
    label_config = """
    <View>
        <TimelineLabels name="videoLabels" toName="video">
            <Label value="NonexistentLabel1"/>
            <Label value="NonexistentLabel2"/>
        </TimelineLabels>
        <Video name="video" value="$video" />
    </View>
    """
    ml = LabelStudioMLBase(label_config=label_config)
    label_interface = ml.label_interface
    control = list(label_interface.controls)[0]

    # Attempt to create the TimelineLabelsModel
    # Expect a ValueError because no labels match and model_trainable is False by default
    with pytest.raises(ValueError) as excinfo:
        model = TimelineLabelsModel.create(ml, control=control)

    # Assert that the exception message contains the expected text
    assert (
        "TimelinesLabels model works in simple mode (without training), "
        "but no labels from YOLO model names are matched"
    ) in str(excinfo.value)
