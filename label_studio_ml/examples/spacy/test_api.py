import pytest
import json
from model import SpacyMLBackend


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=SpacyMLBackend)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
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

    response = client.post('/predict', json=json.loads('{"tasks": [{"id": 1428, "data": {"text": "Katy Perry and Pharrell have the same taste in Apple Watches http://t.co/k6k3SdwShP"}, "meta": {}, "created_at": "2024-03-20T20:50:22.955810Z", "updated_at": "2024-03-20T20:50:22.955819Z", "is_labeled": false, "overlap": 1, "inner_id": 18, "total_annotations": 0, "cancelled_annotations": 0, "total_predictions": 0, "comment_count": 0, "unresolved_comment_count": 0, "last_comment_updated_at": null, "project": 6, "updated_by": null, "file_upload": 8, "comment_authors": [], "annotations": [], "predictions": []}], "project": "6.1710967805", "label_config": "<View>\\n  <Labels name=\\"ner_tags\\" toName=\\"text\\">\\n    <Label value=\\"label_1\\" background=\\"#FFA39E\\"/>\\n    <Label value=\\"label_2\\" background=\\"#D4380D\\"/>\\n    <Label value=\\"label_3\\" background=\\"#FFC069\\"/>\\n  </Labels>\\n  <Text name=\\"text\\" value=\\"$text\\"/>\\n</View>", "params": {"login": null, "password": null, "context": null}}'))
    assert response.status_code == 200
    assert response.json == expected_response
