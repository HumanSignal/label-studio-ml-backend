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
from model import NemoASR


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=NemoASR)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = json.loads('{"tasks": [{"id": 102651352, "data": {"audio": "https://htx-pub.s3.amazonaws.com/datasets/audio/f2bjrop1.1.wav"}, "meta": {}, "created_at": "2024-04-08T15:52:07.473205Z", "updated_at": "2024-04-08T15:52:07.473210Z", "is_labeled": false, "overlap": 1, "inner_id": 2, "total_annotations": 0, "cancelled_annotations": 0, "total_predictions": 0, "comment_count": 0, "unresolved_comment_count": 0, "last_comment_updated_at": null, "project": 62075, "updated_by": null, "file_upload": 1430391, "comment_authors": [], "annotations": [], "predictions": []}], "project": "62075.1712591487", "label_config": "<View>\\n  <Audio name=\\"audio\\" value=\\"$audio\\" zoom=\\"true\\" hotkey=\\"ctrl+enter\\" />\\n  <Header value=\\"Provide Transcription\\" />\\n  <TextArea name=\\"transcription\\" toName=\\"audio\\"\\n            rows=\\"4\\" editable=\\"true\\" maxSubmissions=\\"1\\" />\\n</View>", "params": {"login": null, "password": null, "context": null}}')

    expected_response = {
        "results": [
            {
            "model_version": "SpacyMLBackend-v0.0.1",
            "result": [
                {
                "from_name": "ner_tags",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "end": 10,
                    "labels": ["PERSON"],
                    "start": 0,
                    "text": "Katy Perry"
                }
                },
                {
                "from_name": "ner_tags",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "end": 23,
                    "labels": ["ORG"],
                    "start": 15,
                    "text": "Pharrell"
                }
                },
                {
                "from_name": "ner_tags",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "end": 83,
                    "labels": ["PRODUCT"],
                    "start": 47,
                    "text": "Apple Watches http://t.co/k6k3SdwShP"
                }
                }
            ],
            "score": 0.0
            }
        ]
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    assert response == expected_response
