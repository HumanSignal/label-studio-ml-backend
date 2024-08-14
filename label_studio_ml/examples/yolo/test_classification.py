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
      <Choices name="label" toName="image" score_threshold="0.25">
        <Label value="Airplane" background="green"/>
        <Label value="Car" background="blue" predicted_values="car, truck"/>
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
                }
            ],
            "score": 0.6459808945655823
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
