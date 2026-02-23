"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

```bash
pip install -r requirements-test.txt
```
"""

import os
import pickle
import pytest
import json

from unittest import mock
from label_studio_ml.utils import compare_nested_structures
from .test_common import client, load_file, TEST_DIR


label_configs = [
    # test 1: one control tag with polygon labels
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
    # test 1: one control tag with polygon labels
    {
        "data": {
            "image": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg"
        }
    },
]

PICKLE_PATH = TEST_DIR + "/test_polygon_labels.pickle"


def get_yolo_results():
    """Load YOLO results from pickle, or return None if file doesn't exist."""
    if os.path.exists(PICKLE_PATH):
        return [load_file(PICKLE_PATH)]
    return [None]


# Mock YOLO prediction results to ensure deterministic polygon points
# (polygon point count varies across ultralytics versions)
yolo_results = get_yolo_results()

expected = [
    # test 1: one control tag with polygon labels
    load_file(TEST_DIR + "/test_polygon_labels.json")
]


@pytest.mark.parametrize(
    "label_config, task, yolo_result, expect",
    zip(label_configs, tasks, yolo_results, expected),
)
def test_polygonlabels_predict(client, label_config, task, yolo_result, expect):
    """Test polygon labels prediction with mocked YOLO results.
    
    We mock YOLO.predict because polygon point counts vary across ultralytics
    versions due to different mask-to-contour conversion algorithms.
    
    To regenerate the pickle and json fixtures, run:
        pytest tests/test_polygon_labels.py::test_create_polygon_labels_pickle -s
    """
    data = {"schema": label_config, "project": "42"}
    response = client.post(
        "/setup", data=json.dumps(data), content_type="application/json"
    )
    assert response.status_code == 200, "Error while setup: " + str(response.content)

    data = {"tasks": [task], "label_config": label_config}

    # If pickle exists, mock YOLO prediction to ensure deterministic results
    if yolo_result is not None:
        with mock.patch("ultralytics.YOLO.predict") as mock_yolo:
            mock_yolo.return_value = yolo_result
            response = client.post(
                "/predict", data=json.dumps(data), content_type="application/json"
            )
    else:
        # No pickle - run without mock (will likely fail comparison)
        # Run test_create_polygon_labels_pickle first to generate fixtures
        pytest.skip(
            "Pickle file not found. Run test_create_polygon_labels_pickle first "
            "to generate test fixtures."
        )

    assert response.status_code == 200, "Error while predict"
    data = response.json
    compare_nested_structures(data["results"], expect)


def test_create_polygon_labels_pickle(client):
    """Generate pickle and json fixtures for polygon labels test.
    
    Run this test to regenerate the test fixtures:
        pytest tests/test_polygon_labels.py::test_create_polygon_labels_pickle -s
    
    This will:
    1. Run the actual YOLO model on the test image
    2. Save the YOLO results to test_polygon_labels.pickle
    3. Save the expected output to test_polygon_labels.json
    """
    label_config = label_configs[0]
    task = tasks[0]
    
    # Setup
    data = {"schema": label_config, "project": "42"}
    response = client.post(
        "/setup", data=json.dumps(data), content_type="application/json"
    )
    assert response.status_code == 200, "Error while setup: " + str(response.content)

    # Capture YOLO results during prediction
    captured_results = []
    original_predict = None
    
    def capture_predict(self, *args, **kwargs):
        results = original_predict(self, *args, **kwargs)
        # Clear orig_img to reduce pickle size
        for r in results:
            r.orig_img = []
        captured_results.append(results)
        return results
    
    from ultralytics import YOLO as UltralyticsYOLO
    original_predict = UltralyticsYOLO.predict
    
    with mock.patch.object(UltralyticsYOLO, 'predict', capture_predict):
        data = {"tasks": [task], "label_config": label_config}
        response = client.post(
            "/predict", data=json.dumps(data), content_type="application/json"
        )
    
    assert response.status_code == 200, "Error while predict"
    result_data = response.json
    
    # Save pickle
    if captured_results:
        with open(PICKLE_PATH, 'wb') as f:
            pickle.dump(captured_results[0], f)
        print(f"\nSaved YOLO results to: {PICKLE_PATH}")
    
    # Save expected JSON
    json_path = TEST_DIR + "/test_polygon_labels.json"
    with open(json_path, 'w') as f:
        json.dump(result_data["results"], f, indent=2)
    print(f"Saved expected results to: {json_path}")
    
    print("\nFixtures regenerated successfully!")
    print("You can now run: pytest tests/test_polygon_labels.py::test_polygonlabels_predict")
