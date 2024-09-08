"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

```bash
pip install -r requirements-test.txt
```
"""

import pytest
import json

from label_studio_ml.utils import compare_nested_structures
from .test_common import client


label_configs = [
    # test 1: one control tag with single choice
    """
    <View>
      <Image name="image" value="$image"/>
      <Choices name="label" toName="image" model_score_threshold="0.53">
        <Choice value="Airplane" background="green"/>
        <Choice value="Car" background="blue" predicted_values="racer, cab"/>
      </Choices>
    </View>
    """,
    # test 2: one control tag with multi choices
    """
    <View>
      <Image name="image" value="$image"/>
      <Choices name="label" toName="image" choice="multiple" model_score_threshold="0.1">
        <Choice value="Grille" background="green"/>
        <Choice value="Cab" background="blue" predicted_values="racer, cab"/>
      </Choices>
    </View>
    """,
    # test 3: no choices
    """
    <View>
      <Image name="image" value="$image"/>
      <Choices name="label" toName="image" choice="multiple" model_score_threshold="0.9">
        <Choice value="Grille" background="green"/>
        <Choice value="Cab" background="blue" predicted_values="racer, cab"/>
      </Choices>
    </View>
    """,
]

tasks = [
    # test 1: one control tag with single choice
    {
        "data": {
            "image": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg"
        }
    },
    # test 2: one control tag with multi choices
    {
        "data": {
            "image": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg"
        }
    },
    # test 3: no choices
    {
        "data": {
            "image": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg"
        }
    },
]

expected = [
    # test 1: one control tag with single choice
    [
        {
            "model_version": "yolo",
            "result": [
                {
                    "from_name": "label",
                    "score": 0.5582300424575806,
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": ["Car"]},
                }
            ],
            "score": 0.5582300424575806,
        }
    ],
    # test 2: one control tag with multi choices
    [
        {
            "model_version": "yolo",
            "result": [
                {
                    "from_name": "label",
                    "score": 0.4747641831636429,
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": ["Cab", "Grille"]},
                }
            ],
            "score": 0.4747641831636429,
        }
    ],
    # test 3: no choices
    [
        {
            "model_version": "yolo",
            "result": [],
            "score": 0.0,
        }
    ],
]


@pytest.mark.parametrize(
    "label_config, task, expect", zip(label_configs, tasks, expected)
)
def test_choices_predict(client, label_config, task, expect):
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
    compare_nested_structures(data["results"], expect)
