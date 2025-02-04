"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

```bash
pip install -r requirements-test.txt
```
"""

import pytest
import json

from label_studio_ml.utils import compare_nested_structures
from .test_common import client, load_file, TEST_DIR


label_configs = [
    # test 1
    """
    <View>
      <KeyPointLabels name="keypoints" toName="image" 
        model_score_threshold="0.85" model_point_threshold="0.95" model_add_bboxes="true" model_point_size="1"  
      >
        <Label value="nose" predicted_values="person" model_index="0" background="red" />

        <Label value="left_eye" predicted_values="person" model_index="1" background="yellow" />
        <Label value="right_eye" predicted_values="person" model_index="2" background="yellow" />

        <Label value="left_ear" predicted_values="person" model_index="3" background="purple" />
        <Label value="right_ear" predicted_values="person" model_index="4" background="purple" />

        <View>
        <Label value="left_shoulder" predicted_values="person" model_index="5" background="green" />
        <Label value="left_elbow" predicted_values="person" model_index="7" background="green" />
        <Label value="left_wrist" predicted_values="person" model_index="9" background="green" />

        <Label value="right_shoulder" predicted_values="person" model_index="6" background="blue" />
        <Label value="right_elbow" predicted_values="person" model_index="8" background="blue" />
        <Label value="right_wrist" predicted_values="person" model_index="10" background="blue" />
        </View>

        <View>
        <Label value="left_hip" predicted_values="person" model_index="11" background="brown" />
        <Label value="left_knee" predicted_values="person" model_index="13" background="brown" />
        <Label value="left_ankle" predicted_values="person" model_index="15" background="brown" />

        <Label value="right_hip" predicted_values="person" model_index="12" background="orange" />
        <Label value="right_knee" predicted_values="person" model_index="14" background="orange" />
        <Label value="right_ankle" predicted_values="person" model_index="16" background="orange" />
        </View>
        
        <Label value="right_ankle" predicted_values="just_to_test_warning" model_index="0" background="orange" />
      </KeyPointLabels>
      <Image name="image" value="$image" />
    </View>
    """,
    # test 2: no bboxes
    """
    <View>
      <KeyPointLabels name="keypoints" toName="image" 
        model_score_threshold="0.85" model_point_threshold="0.96" model_add_bboxes="false" model_point_size="1"  
      >
        <Label value="nose" predicted_values="person" model_index="0" background="red" />

        <Label value="left_eye" predicted_values="person" model_index="1" background="yellow" />
        <Label value="right_eye" predicted_values="person" model_index="2" background="yellow" />

        <Label value="left_ear" predicted_values="person" model_index="3" background="purple" />
        <Label value="right_ear" predicted_values="person" model_index="4" background="purple" />

        <View>
        <Label value="left_shoulder" predicted_values="person" model_index="5" background="green" />
        <Label value="left_elbow" predicted_values="person" model_index="7" background="green" />
        <Label value="left_wrist" predicted_values="person" model_index="9" background="green" />

        <Label value="right_shoulder" predicted_values="person" model_index="6" background="blue" />
        <Label value="right_elbow" predicted_values="person" model_index="8" background="blue" />
        <Label value="right_wrist" predicted_values="person" model_index="10" background="blue" />
        </View>

        <View>
        <Label value="left_hip" predicted_values="person" model_index="11" background="brown" />
        <Label value="left_knee" predicted_values="person" model_index="13" background="brown" />
        <Label value="left_ankle" predicted_values="person" model_index="15" background="brown" />

        <Label value="right_hip" predicted_values="person" model_index="12" background="orange" />
        <Label value="right_knee" predicted_values="person" model_index="14" background="orange" />
        <Label value="right_ankle" predicted_values="person" model_index="16" background="orange" />
        </View>
      </KeyPointLabels>
      <Image name="image" value="$image" />
    </View>
    """,
]

tasks = [
    # test 1
    {
        "data": {
            "image": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg"
        }
    },
    # test 2: no bbox
    {
        "data": {
            "image": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg"
        }
    },
]

expected = [
    # test 1
    load_file(TEST_DIR + "/test_keypoint_labels_1.json"),
    # test 2: no bbox
    load_file(TEST_DIR + "/test_keypoint_labels_2.json"),
]


@pytest.mark.parametrize(
    "label_config, task, expect", zip(label_configs, tasks, expected)
)
def test_keypoints_predict(client, label_config, task, expect):
    data = {"schema": label_config, "project": "42"}
    response = client.post(
        "/setup", data=json.dumps(data), content_type="application/json"
    )
    assert response.status_code == 200, "Error while setup: " + str(response.content)

    data = {"tasks": [task], "label_config": label_config}
    response = client.post(
        "/predict", data=json.dumps(data), content_type="application/json"
    )
    assert response.status_code == 200, "Error while predict"
    data = response.json
    compare_nested_structures(data["results"], expect, rel=1e-1)
