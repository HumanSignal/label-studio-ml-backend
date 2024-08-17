"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

```bash
pip install -r requirements-test.txt
```
"""

import pytest
import json

from label_studio_ml.utils import compare_nested_structures
from test_bboxes import client


label_configs = [
    # test 1: one control tag with rectangle labels
    """
    <View>
      <Image name="image" value="$image"/>
      <Choices name="label" toName="image" score_threshold="0.53">
        <Choice value="Airplane" background="green"/>
        <Choice value="Car" background="blue" predicted_values="racer, cab"/>
      </Choices>
    </View>
    """
]

tasks = [
    # test 1: one control tag with rectangle labels
    {
        "data": {
            "image": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg"
        }
    },
]

expected = [
    # test 1: one control tag with rectangle labels
    [
        {
            "model_version": "yolo",
            "result": [
                {
                    "from_name": "label",
                    "score": 0.5582300424575806,
                    "to_name": "image",
                    "type": "choices",
                    "value": {
                        "choices": ["Car"]
                    }
                }
            ],
            "score": 0.5582300424575806
        }
    ]
]


@pytest.mark.parametrize("label_config, task, expect", zip(label_configs, tasks, expected))
def test_choices_predict(client, label_config, task, expect):
    data = {"schema": label_config, "project": "42"}
    response = client.post("/setup", data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200, "Error while setup: " + str(response.content)

    data = {"tasks": [task], "label_config": label_config}
    response = client.post("/predict", data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200, "Error while predict: " + str(response.content)
    data = response.json
    compare_nested_structures(data["results"], expect)
