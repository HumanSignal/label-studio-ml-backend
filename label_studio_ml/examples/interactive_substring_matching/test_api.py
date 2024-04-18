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
from model import InteractiveSubstringMatching


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=InteractiveSubstringMatching)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = json.loads('{"tasks": [{"id": 102644832, "annotations": [], "file_upload": "f3331b20-texts.json", "drafts": [], "predictions": [], "agreement": null, "data": {"text": "We got hold of one of the first Apple Watches to leave Apple. Here\'s what we found http://t.co/yUjAbasx9N http://t.co/9JCtPMkEWx"}, "meta": {}, "created_at": "2024-04-08T14:21:29.542516Z", "updated_at": "2024-04-08T14:21:29.542520Z", "inner_id": 40, "total_annotations": 0, "cancelled_annotations": 0, "total_predictions": 0, "comment_count": 0, "unresolved_comment_count": 0, "last_comment_updated_at": null, "project": 62053, "updated_by": null, "comment_authors": []}], "project": "62053.1712586036", "label_config": "<View>\\n  <Labels name=\\"label\\" toName=\\"text\\">\\n    <Label value=\\"PER\\" background=\\"red\\"/>\\n    <Label value=\\"ORG\\" background=\\"darkorange\\"/>\\n    <Label value=\\"LOC\\" background=\\"orange\\"/>\\n    <Label value=\\"MISC\\" background=\\"green\\"/>\\n  </Labels>\\n\\n  <Text name=\\"text\\" value=\\"$text\\"/>\\n</View>", "params": {"login": null, "password": null, "context": {"annotation_id": null, "draft_id": 0, "user_id": 2640, "result": [{"value": {"start": 32, "end": 37, "text": "Apple", "labels": ["ORG"]}, "id": "1aa-14p0B6", "from_name": "label", "to_name": "text", "type": "labels", "origin": "manual"}]}}}')
    expected_response = json.loads('{"results":[{"model_version":"0.0.1","result":[{"from_name":"label","id":"40de","score":0.8,"to_name":"text","type":"labels","value":{"end":37,"labels":["ORG"],"start":32,"text":"Apple"}},{"from_name":"label","id":"9090","score":0.8,"to_name":"text","type":"labels","value":{"end":60,"labels":["ORG"],"start":55,"text":"Apple"}}],"score":0.8}]}')

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    for key in ('from_name', 'to_name', 'type', 'value'):
        assert response['results'][0]['result'][0][key] == expected_response['results'][0]['result'][0][key]
