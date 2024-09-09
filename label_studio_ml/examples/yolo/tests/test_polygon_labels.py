"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

```bash
pip install -r requirements-test.txt
```
"""

import pytest
import json

from label_studio_ml.utils import compare_nested_structures
from .test_common import client, load_file, TEST_DIR


label_configs = [
    # test 1: one control tag with rectangle labels
    """
    <View>
      <Image name="image" value="$image"/>
      <PolygonLabels name="label" toName="image" model_score_threshold="0.6">
        <Label value="Person" background="green"/>
        <Label value="Truck" background="blue"/>
      </PolygonLabels>
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
    load_file(TEST_DIR + "/test_polygon_labels.json")
]


@pytest.mark.parametrize(
    "label_config, task, expect", zip(label_configs, tasks, expected)
)
def test_polygonlabels_predict(client, label_config, task, expect):
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
