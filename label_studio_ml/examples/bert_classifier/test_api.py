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
from model import BertClassifier


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=BertClassifier)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = {
        'tasks': [{
            'data': {
                'text': 'Today is a great day to play football.'
            }
        }],
        # Your labeling configuration here
        'label_config':
            '<View>'
            '<Text name="text" value="$text" />'
            '<Choices name="topic" toName="text" choice="single">'
            '<Choice value="sports" />'
            '<Choice value="politics" />'
            '<Choice value="technology" />'
            '</Choices>'
            '</View>'
    }

    expected_response_results = [{
        'result': [{
            'from_name': 'topic',
            'to_name': 'text',
            'type': 'choices',
            'value': {'choices': ['sports']}
        }]
    }]

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    assert len(expected_response_results) == len(response['results'])


# TODO
# Implement test_fit()
