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
from model import HuggingFaceLLM


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=HuggingFaceLLM)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = {
        'tasks': [{
            'data': {
                'text': 'If I say "I feel like I am walking on air" it means '
            }
        }],
        # Your labeling configuration here
        'label_config': '''<View>
            <Text name="input_text" value="$text"/>
          <TextArea name="generated_text"  toName="input_text"/>
        </View>'''
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)

    # The generated text depends on the model and transformers version, so check the shape, not the exact string.
    assert len(response['results']) == 1
    prediction = response['results'][0]
    assert prediction['model_version'] == 'HuggingFaceLLM-v0.0.1'
    assert len(prediction['result']) == 1
    region = prediction['result'][0]
    assert region['from_name'] == 'generated_text'
    assert region['to_name'] == 'input_text'
    assert region['type'] == 'textarea'
    assert len(region['value']['text']) == 1
    assert isinstance(region['value']['text'][0], str) and region['value']['text'][0].strip()
