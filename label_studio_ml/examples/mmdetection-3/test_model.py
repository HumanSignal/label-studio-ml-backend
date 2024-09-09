import requests

from mmdetection import MMDetection

from label_studio_ml.utils import compare_nested_structures

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
