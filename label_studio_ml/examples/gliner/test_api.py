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
from model import GLiNERModel


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=GLiNERModel)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = {
        'tasks': [{'id': 6,
                   'data': {'id': '5316', 'sample_id': '83dd3f62-4dd5-45eb-8626-ee8539963194',
                            'tokens': ['atomoxetine', '[', 'oral', 'suspension', ']', 'norepinephrine', 'reuptake',
                                       'inhibitor'],
                            'ner_tags': ['B-Medication/Vaccine', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                            'ner_tags_index': [63, 0, 0, 0, 0, 0, 0, 0],
                            'text': 'atomoxetine [ oral suspension ] norepinephrine reuptake inhibitor'},
                   'meta': {},
                   'created_at': '2024-04-13T19:22:37.153686Z',
                   'updated_at': '2024-05-03T00:03:22.356871Z',
                   'is_labeled': False,
                   'overlap': 1,
                   'inner_id': 6,
                   'total_annotations': 1,
                   'cancelled_annotations': 0,
                   'total_predictions': 0,
                   'comment_count': 0,
                   'unresolved_comment_count': 0,
                   'last_comment_updated_at': None,
                   'project': 2,
                   'updated_by': 1,
                   'file_upload': None,
                   'comment_authors': [],
                   'predictions': [],
                   }],
        # Your labeling configuration here
        'label_config': '<View> \\n <Labels name="label" toName="text">\\n<Label value="Medication/Vaccine" background="red"/>\\n<Label value="MedicalProcedure" background="blue"/>\\n<Label value="AnatomicalStructure" background="orange"/>\\n<Label value="Symptom" background="green"/>\\n<Label value="Disease" background="purple"/>\\n</Labels>\\n<Text name="text" value="$text"/>\\n</View>'
    }

    expected_response = {"results": [{"model_version": "GLiNERModel-v0.0.1", "result": [
        {"from_name": "label", "score": 0.922, "to_name": "text", "type": "labels",
         "value": {"end": 11, "labels": ["Medication/Vaccine"], "start": 0, "text": "atomoxetine"}},
        {"from_name": "label", "score": 0.7053, "to_name": "text", "type": "labels",
         "value": {"end": 65, "labels": ["Medication/Vaccine"], "start": 32,
                   "text": "norepinephrine reuptake inhibitor"}}], "score": 0.7053}]}

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    assert expected_response == response
