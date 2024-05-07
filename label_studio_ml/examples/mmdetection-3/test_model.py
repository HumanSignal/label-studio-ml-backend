import requests

from mmdetection import MMDetection

from pytest import approx

label_config = """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Airplane" background="green"/>
    <Label value="Car" background="blue"/>
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
        "result": [
            {
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": ["Car"],
                    "x": 22.946878274281822,
                    "y": 9.788729747136433,
                    "width": 66.54588778813681,
                    "height": 76.81492235925462,
                },
                "score": 0.8933283090591431,
            }
        ],
        "score": 0.8933283090591431,
        "model_version": 'MMDetection-v0.0.1',
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


def test_mmdetection_model_predict():
    model = MMDetection(label_config=label_config)
    predictions = model.predict([task])

    print(predictions)
    assert len(predictions) == 1, "Only one prediction should have been returned"
    compare_nested_structures(predictions, expected)


def test_mmdetection_http_request_predict():
    data = {"schema": label_config, "project": "42"}
    response = requests.post("http://0.0.0.0:9090/setup", json=data)
    assert response.status_code == 200, "Error while setup: " + str(response.content)

    data = {"tasks": [task], "label_config": label_config}
    response = requests.post("http://0.0.0.0:9090/predict", json=data)
    assert response.status_code == 200, "Error while predict: " + str(response.content)
    data = response.json()
    compare_nested_structures(data["results"], expected)
