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
    # test 1: one control tag with rectangle labels
    """
    <View>
      <Header value="Select label and click the image to start"/>
      <Image name="image" value="$image" zoom="true"/>
      
      <RectangleLabels name="rect" toName="image"
                     model_score_threshold="0.1" model_obb="true">
        <Label value="plane" background="red" 
            predicted_values="plane,helicopter"/>
        <Label value="vehicle" background="blue" 
            predicted_values="ship,storage tank,bridge,large vehicle,small vehicle"/>
      </RectangleLabels>
      
    </View>
    """,
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
                    "from_name": "rect",
                    "score": 0.32253125309944153,
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "height": 3.3118023546502084,
                        "original_height": 576,
                        "original_width": 768,
                        "rectanglelabels": ["vehicle"],
                        "rotation": -89.43998820538127,
                        "width": 2.2955212735479535,
                        "x": 1.9985803710085965,
                        "y": 10.487648558804944,
                    },
                }
            ],
            "score": 0.32253125309944153,
        }
    ]
]


@pytest.mark.parametrize(
    "label_config, task, expect", zip(label_configs, tasks, expected)
)
def test_rectanglelabels_obb_predict(client, label_config, task, expect):
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
