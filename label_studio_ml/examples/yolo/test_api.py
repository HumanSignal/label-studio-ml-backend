"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""

import pytest
import json

from model import YOLO
from pytest import approx


label_config = """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image" score_threshold="0.25">
    <Label value="Airplane" background="green"/>
    <Label value="Car" background="blue" predicted_values="car, truck"/>
  </RectangleLabels>
</View>
"""
task = {
    "data": {
        "image": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg"
    }
}
expected = [
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
]


def compare_nested_structures(a, b, path=""):
    """Compare two dicts or list with approx() for float values"""
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"Keys mismatch at {path}"
        for key in a.keys():
            compare_nested_structures(a[key], b[key], path + "." + str(key))
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"List size mismatch at {path}"
        for i, (act_item, exp_item) in enumerate(zip(a, b)):
            compare_nested_structures(act_item, exp_item, path + f"[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert a == approx(b), f"Mismatch at {path}"
    else:
        assert a == b, f"Mismatch at {path}"


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=YOLO)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_rectanglelabels_predict(client):
    data = {"schema": label_config, "project": "42"}
    response = client.post("/setup", data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200, "Error while setup: " + str(response.content)

    data = {"tasks": [task], "label_config": label_config}
    response = client.post("/predict", data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200, "Error while predict: " + str(response.content)
    data = response.json
    compare_nested_structures(data["results"], expected), "Expected and returned results mismatch"
