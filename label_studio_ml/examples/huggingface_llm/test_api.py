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

    expected_response = {
        'results': [{
            'model_version': 'HuggingFaceLLM-v0.0.1',
            'score': 0.0,
            'result': [{
                'from_name': 'generated_text',
                'to_name': 'input_text',
                'type': 'textarea',
                'value': {'text': ['"I am not walking on air"\nI\'m not sure if you\'re being sarcastic or not, but I think you\'re right.']}
            }]
        }]
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    assert response == expected_response
