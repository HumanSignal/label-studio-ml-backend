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
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image" model_score_threshold="0.25">
        <Label value="Airplane" background="green"/>
        <Label value="Car" background="blue" predicted_values="car, truck"/>
      </RectangleLabels>
    </View>
    """,
    # test 2: two control tags with rectangle labels and two images
    """
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image" model_score_threshold="0.30">
        <Label value="Airplane" background="green"/>
        <Label value="Car" background="blue" predicted_values="car, truck"/>
      </RectangleLabels>
      
      <Image name="image2" value="$image2"/>
      <RectangleLabels name="label2" toName="image2" model_score_threshold="0.90">
        <Label value="Person" background="green"/>
        <Label value="Animal" background="blue" predicted_values="cat,dog"/>
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
    # test 2: two control tags with rectangle labels and two images
    {
        "data": {
            "image": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg",
            "image2": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg",
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
                    "score": 0.5791077017784119,
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "height": 77.13761925697327,
                        "rectanglelabels": ["Car"],
                        "width": 69.33701038360596,
                        "x": 21.9377338886261,
                        "y": 7.984769344329834,
                    },
                },
                {
                    "from_name": "label",
                    "score": 0.31354132294654846,
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "height": 25.369155406951904,
                        "rectanglelabels": ["Car"],
                        "width": 18.623733520507812,
                        "x": 81.27312660217285,
                        "y": 0.10521858930587769,
                    },
                },
            ],
            "score": 0.44632451236248016,
        }
    ],
    # test 2: two control tags with rectangle labels and two images
    [
        {
            "model_version": "yolo",
            "result": [
                {
                    "from_name": "label",
                    "score": 0.5791077017784119,
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "height": 77.13761925697327,
                        "rectanglelabels": ["Car"],
                        "width": 69.33701038360596,
                        "x": 21.9377338886261,
                        "y": 7.984769344329834,
                    },
                },
                {
                    "from_name": "label",
                    "score": 0.31354132294654846,
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "height": 25.369155406951904,
                        "rectanglelabels": ["Car"],
                        "width": 18.623733520507812,
                        "x": 81.27312660217285,
                        "y": 0.10521858930587769,
                    },
                },
                {
                    "from_name": "label2",
                    "score": 0.9059886932373047,
                    "to_name": "image2",
                    "type": "rectanglelabels",
                    "value": {
                        "height": 39.60925042629242,
                        "rectanglelabels": ["Person"],
                        "width": 10.503808408975601,
                        "x": 89.45398144423962,
                        "y": 6.985808908939362,
                    },
                },
            ],
            "score": 0.5995459059874216,
        }
    ],
]


@pytest.mark.parametrize(
    "label_config, task, expect", zip(label_configs, tasks, expected)
)
def test_rectanglelabels_predict(client, label_config, task, expect):
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
