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


@pytest.fixture
def client():
    from _wsgi import app
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = {
        'tasks': [{
            'data': {
                'text': 'Long text to be classified into one of the topics.'
            }
        }],
        # Your labeling configuration here
        'label_config': '<View> \\n <Labels name="label" toName="text">\\n<Label value="Medication/Vaccine" background="red"/>\\n<Label value="MedicalProcedure" background="blue"/>\\n<Label value="AnatomicalStructure" background="orange"/>\\n<Label value="Symptom" background="green"/>\\n<Label value="Disease" background="purple"/>\\n</Labels>\\n<Text name="text" value="$text"/>\\n</View>'
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    print(response)
    assert expected_response == response
