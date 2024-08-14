"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

```bash
pip install -r requirements-test.txt
```
"""

import pytest
import json

from model import YOLO
from label_studio_ml.utils import compare_nested_structures


label_configs = [
    # test 1: one control tag with rectangle labels
    """
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image" score_threshold="0.25">
        <Label value="Airplane" background="green"/>
        <Label value="Car" background="blue" predicted_values="car, truck"/>
      </RectangleLabels>
    </View>
    """,

    # test 2: two control tags with rectangle labels and two images
    """
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image" score_threshold="0.65">
        <Label value="Airplane" background="green"/>
        <Label value="Car" background="blue" predicted_values="car, truck"/>
      </RectangleLabels>
      
      <Image name="image2" value="$image2"/>
      <RectangleLabels name="label2" toName="image2" score_threshold="0.90">
        <Label value="Person" background="green"/>
        <Label value="Animal" background="blue" predicted_values="cat,dog"/>
      </RectangleLabels>
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

    # test 2: two control tags with rectangle labels and two images
    {
        "data": {
            "image": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg",
            "image2": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg"
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
                    "score": 0.684260904788971,
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "height": 26.102054119110107,
                        "rectanglelabels": ["Car"],
                        "width": 18.652383983135223,
                        "x": 81.26997724175453,
                        "y": 0.07733255624771118
                    }
                },
                {
                    "from_name": "label",
                    "score": 0.6077008843421936,
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "height": 77.03651785850525,
                        "rectanglelabels": ["Car"],
                        "width": 69.53177452087402,
                        "x": 21.781492233276367,
                        "y": 8.59556794166565
                    }
                }
            ],
            "score": 0.6459808945655823
        }
    ],

    # test 2: two control tags with rectangle labels and two images
    [
        {
            "model_version": "yolo",
            "result": [
                {
                    "from_name": "label",
                    "score": 0.684260904788971,
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "height": 26.102054119110107,
                        "rectanglelabels": ["Car"],
                        "width": 18.652383983135223,
                        "x": 81.26997724175453,
                        "y": 0.07733255624771118
                    }
                },
                {
                    "from_name": "label2",
                    "score": 0.9029274582862854,
                    "to_name": "image2",
                    "type": "rectanglelabels",
                    "value": {
                        "height": 39.65578079223633,
                        "rectanglelabels": ["Person"],
                        "width": 10.530853271484375,
                        "x": 89.4278347492218,
                        "y": 6.96789026260376
                    }
                },
            ],
            "score": 0.7935941815376282
        }
    ],
]


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=YOLO)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.mark.parametrize("label_config, task, expect", zip(label_configs, tasks, expected))
def test_rectanglelabels_predict(client, label_config, task, expect):
    data = {"schema": label_config, "project": "42"}
    response = client.post("/setup", data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200, "Error while setup: " + str(response.content)

    data = {"tasks": [task], "label_config": label_config}
    response = client.post("/predict", data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200, "Error while predict: " + str(response.content)
    data = response.json
    compare_nested_structures(data["results"], expect), "Expected and returned results mismatch"
