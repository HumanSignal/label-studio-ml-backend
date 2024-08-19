"""
This file contains tests for the API
"""

import pytest
import json

from model import YOLO
from label_studio_ml.utils import compare_nested_structures


label_configs = [
    # test 1: wrong key in task data
    """
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image" score_threshold="0.25">
        <Label value="Airplane" background="green"/>
        <Label value="Car" background="blue" predicted_values="car, truck"/>
      </RectangleLabels>
    </View>
    """,
]

tasks = [
    # test 1: wrong key in task data
    {
        "data": {
            "wrong_key": "https://some/path"
        }
    },
]

expected = [
    # test 1: wrong key in task data
    [],
]


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=YOLO)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.mark.parametrize("label_config, task, expect", zip(label_configs, tasks, expected))
def test_wrong_key_in_task_data(client, label_config, task, expect, capsys):
    data = {"schema": label_config, "project": "42"}
    response = client.post("/setup", data=json.dumps(data), content_type='application/json')
    assert response.status_code == 200, "Error while setup: " + str(response.content)

    data = {"tasks": [task], "label_config": label_config}
    response = client.post("/predict", data=json.dumps(data), content_type='application/json')
    assert response.status_code == 500, "Error was expected, but another status code found"

    # Capture stdout and stderr
    captured = capsys.readouterr()

    # Check for specific words in the output
    assert "Can't load path using key" in captured.err or captured.out, "Not expected error text found"
