"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:
"""

import pytest
import json
import numpy as np

from label_studio_ml.utils import compare_nested_structures
from unittest import mock
from .test_common import client, load_file, TEST_DIR
from ..utils.converter import convert_timelinelabels_to_probs, convert_probs_to_timelinelabels


label_configs = [
    # test 1: one control tag with video timeline labels
    """
    <View>
       <TimelineLabels name="videoLabels" toName="video">
         <Label value="Car" predicted_values="racer, cab"/>
         <Label value="Person" background="red"/>
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


# @pytest.mark.skip(reason="Not yet implemented")
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


def test_convert_timelinelabels_to_probs():
    # Example usage:
    regions = [
        {
            'from_name': 'videoLabels',
            'id': '0_9',
            'origin': 'prediction',
            'to_name': 'video',
            'type': 'timelinelabels',
            'value': {
                'ranges': [{'end': 8, 'start': 0}],
                'timelinelabels': ['Snow']
            }
        },
        {
            'from_name': 'videoLabels',
            'id': '10_15',
            'origin': 'prediction',
            'to_name': 'video',
            'type': 'timelinelabels',
            'value': {
                'ranges': [{'end': 14, 'start': 10}],
                'timelinelabels': ['Rain']
            }
        },
        {
            'from_name': 'videoLabels',
            'id': '0_9x',
            'origin': 'prediction',
            'to_name': 'video',
            'type': 'timelinelabels',
            'value': {
                'ranges': [{'end': 14, 'start': 14}],
                'timelinelabels': ['Snow']
            }
        },
    ]

    labels_array, label_mapping = convert_timelinelabels_to_probs(regions)

    # Label Mapping
    expected_label_mapping = {'Rain': 0, 'Snow': 1}

    # Labels Array
    expected_labels_array = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 1]
    ])

    print("Labels Array:\n", labels_array)
    print("Label Mapping:\n", label_mapping)

    assert label_mapping == expected_label_mapping
    assert np.array_equal(labels_array, expected_labels_array)


def test_convert_probs_to_timelinelabels():
    # Example usage
    probs = [
        [0.8, 0.2],  # Frame 0: 'Rain' has a high probability
        [0.9, 0.1],  # Frame 1: 'Rain' is still active
        [0.6, 0.4],  # Frame 2: Both probabilities are near threshold
        [0.2, 0.7],  # Frame 3: 'Snow' becomes more active
        [0.1, 0.9]  # Frame 4: 'Snow' has a high probability
    ]
    
    label_mapping = {'Rain': 0, 'Snow': 1}
    
    timeline_regions = convert_probs_to_timelinelabels(probs, label_mapping, score_threshold=0.5)
    
    for region in timeline_regions:
        print(region)