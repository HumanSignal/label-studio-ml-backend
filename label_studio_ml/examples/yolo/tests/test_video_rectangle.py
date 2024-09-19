"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:
"""

import os
import pytest
import json
import yaml

from label_studio_ml.utils import compare_nested_structures
from model import YOLO
from .test_common import client, load_file, TEST_DIR
from unittest import mock


label_configs = [
    # test 1: one control tag with video rectangle
    """
    <View>
       <Labels name="videoLabels" toName="video" allowEmpty="true">
         <Label value="person" background="blue"/>
       </Labels>
       
       <!-- Please specify FPS carefully, it will be used for all project videos -->
       <Video name="video" value="$video" framerate="25.0"/>
       <VideoRectangle name="box" toName="video" botsort_track_high_thresh="0.1" botsort_track_low_thresh="0.1" />
    </View>
    """,
    # test 2: video rectangle without botsort parameters
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
    {"data": {"video": "tests/opossum_snow_short.mp4"}},
    # test 2: one control tag with rectangle labels
    {"data": {"video": "tests/opossum_snow_short.mp4"}},
]

yolo_results = [load_file(TEST_DIR + "/opossum_snow_short.pickle"), None]

expected = [
    # test 1: one control tag with rectangle labels
    load_file(TEST_DIR + "/opossum_snow_short_1.json"),
    load_file(TEST_DIR + "/opossum_snow_short_2.json"),
]


@pytest.mark.parametrize(
    "label_config, task, yolo_result, expect",
    zip(label_configs, tasks, yolo_results, expected),
)
def test_rectanglelabels_predict(client, label_config, task, yolo_result, expect):
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


def test_create_video_rectangles():
    """How to create pickle?
    1. Make a break point at the first line of create_video_rectangles()
    2. Run test_rectanglelabels_predict() test
    3. On the breakpoint:
        3.1 import pickle
        3.2 for r in results: r.orig_img = []
        3.2 with open('model_track_results.pickle', 'wb') as f: pickle.dump(results, f)
    """
    ml = YOLO(project_id="42", label_config=label_configs[0])
    control_models = ml.detect_control_models()
    regions = control_models[0].create_video_rectangles(
        yolo_results[0], "tests/opossum_snow_short.mp4"
    )

    predictions = expected[0]
    assert regions == predictions[0]["result"]


def test_update_tracker_params_with_real_config():
    tmp_path = os.path.dirname(__file__)
    label_config = """
    <View>
       <Labels name="videoLabels" toName="video" allowEmpty="true">
         <Label value="person" background="blue"/>
       </Labels>

       <!-- Please specify FPS carefully, it will be used for all project videos -->
       <Video name="video" value="$video" framerate="25.0"/>
       <VideoRectangle name="box" toName="video" 
           botsort_track_high_thresh="0.6" 
           botsort_track_low_thresh="0.4" 
           botsort_new_track_thresh="0.3" 
           botsort_track_buffer="50" 
           botsort_match_thresh="0.85" 
           botsort_fuse_score="false" 
           botsort_gmc_method="none" />
    </View>
    """

    # Initialize the model with the label config
    ml = YOLO(project_id="42", label_config=label_config)
    control_models = ml.detect_control_models()
    video_rectangle_model = control_models[0]

    # Mock original botsort.yaml content
    original_yaml_content = """
        tracker_type: botsort
        track_high_thresh: 0.1
        track_low_thresh: 0.1
        new_track_thresh: 0.1
        track_buffer: 30
        match_thresh: 0.8
        fuse_score: true
        gmc_method: sparseOptFlow
        proximity_thresh: 0.5
        appearance_thresh: 0.25
        with_reid: false
    """

    # Create a temporary YAML file to simulate the original config
    original_yaml_path = f"{tmp_path}/botsort.yaml"
    with open(original_yaml_path, "w") as file:
        file.write(original_yaml_content)

    # Update tracker parameters based on the labeling config
    new_yaml_path = video_rectangle_model.update_tracker_params(
        original_yaml_path, "botsort_"
    )

    # Check that the new YAML file was created
    assert os.path.exists(new_yaml_path), "The new YAML file was not created."

    # Load the new YAML file
    with open(new_yaml_path, "r") as file:
        updated_config = yaml.safe_load(file)

    # Verify that the parameters were correctly updated
    assert updated_config["track_high_thresh"] == 0.6
    assert updated_config["track_low_thresh"] == 0.4
    assert updated_config["new_track_thresh"] == 0.3
    assert updated_config["track_buffer"] == 50
    assert updated_config["match_thresh"] == 0.85
    assert updated_config["fuse_score"] == False  # Boolean comparison
    assert updated_config["gmc_method"] == "none"

    # Clean up: remove the temporary YAML file
    os.remove(new_yaml_path)
