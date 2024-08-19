"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:
"""
import os.path
import pickle
import pytest
import json

from label_studio_ml.utils import compare_nested_structures
from model import YOLO
from test_common import client


label_configs = [
    # test 1: one control tag with video rectangle
    """
    <View>
       <Labels name="videoLabels" toName="video" allowEmpty="true">
         <Label value="person" background="blue"/>
       </Labels>
       
       <!-- Please specify FPS carefully, it will be used for all project videos -->
       <Video name="video" value="$video" framerate="25.0"/>
       <VideoRectangle name="box" toName="video" />
    </View>
    """,
]

tasks = [
    # test 1: one control tag with rectangle labels
    {
        "data": {
            "video": "opossum_snow_short.mp4"
        }
    },
]

expected = [
    # test 1: one control tag with rectangle labels
    [
        {
            "model_version": "yolo",
            "result": [
            ],
            "score": 0.5
        }
    ],
]


@pytest.mark.parametrize("label_config, task, expect", zip(label_configs, tasks, expected))
def test_rectanglelabels_predict(client, label_config, task, expect):
    data = {"schema": label_config, "project": "42"}
    response = client.post("/setup", data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200, "Error while setup: " + str(response.content)

    data = {"tasks": [task], "label_config": label_config}
    response = client.post("/predict", data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200, "Error while predict"
    data = response.json
    compare_nested_structures(data["results"], expect)


def test_create_video_rectangles():
    """ How to create pickle?
    1. Make a break point at the first line of create_video_rectangles()
    2. Run test_rectanglelabels_predict() test
    3. On the breakpoint:
        3.1 import pickle
        3.2 for r in results: r.orig_img = []
        3.2 with open('model_track_results.pickle', 'wb') as f: pickle.dump(results, f)
    """
    with open('model_track_results.pickle', 'rb') as f:
        results = pickle.load(f)

    ml = YOLO(project_id='42', label_config=label_configs[0])
    control_models = ml.detect_control_models()
    regions = control_models[0].create_video_rectangles(results, 'opossum_snow_short.mp4')

    with open(os.path.dirname(__file__) + '/test_videorectangle.json') as f:
        expected_regions = json.load(f)

    assert regions == expected_regions

