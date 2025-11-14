"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""

import json

import pytest
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
from model import DeepgramModel


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=DeepgramModel)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client, monkeypatch):
    """
    Scenario: exercise the /predict endpoint with minimal payload.
    Steps   : patch DeepgramModel.setup to avoid env var requirements, POST minimal request.
    Checks  : ensure HTTP 200 is returned with empty results when no context is provided.
    """
    # Patch setup to avoid requiring DEEPGRAM_API_KEY during model instantiation
    monkeypatch.setattr(DeepgramModel, 'setup', lambda self: None)
    
    request = {
        'tasks': [{
            'id': 1,
            'data': {}
        }],
        'label_config': '<View></View>',
        'project': '1.1234567890'
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    body = json.loads(response.data)
    assert 'results' in body
    # When no context is provided, predict returns empty predictions
    assert body['results'] == []


def test_predict_endpoint_returns_stubbed_predictions(client, monkeypatch):
    """
    Scenario: exercise the /predict endpoint without hitting external services.
    Steps   : patch DeepgramModel.setup and predict to avoid env vars and return stubbed data,
              POST realistic payload to /predict, parse the JSON.
    Checks  : ensure HTTP 200 is returned and the payload's `results` field matches the stub.
    """
    # Create a proper PredictionValue object with result structure
    stub_prediction = PredictionValue(
        result=[{
            'from_name': 'text',
            'to_name': 'audio',
            'type': 'textarea',
            'value': {'text': ['Hello from stub']}
        }]
    )

    # Patch setup to avoid requiring DEEPGRAM_API_KEY during model instantiation
    monkeypatch.setattr(DeepgramModel, 'setup', lambda self: None)
    
    def fake_predict(self, tasks, context=None, **params):
        return ModelResponse(predictions=[stub_prediction])

    monkeypatch.setattr(DeepgramModel, 'predict', fake_predict)

    request_payload = {
        'tasks': [{
            'id': 42,
            'data': {'text': 'Sample request text'}
        }],
        'label_config': '<View><TextArea name="text" toName="audio"/></View>',
        'project': '1.1234567890',
        'params': {'context': {'result': []}}
    }

    response = client.post('/predict', data=json.dumps(request_payload), content_type='application/json')

    assert response.status_code == 200
    body = json.loads(response.data)
    # The API returns results which should contain the prediction's result
    assert 'results' in body
    assert len(body['results']) == 1
    # Verify the structure matches what we stubbed
    assert body['results'][0]['result'][0]['value']['text'] == ['Hello from stub']
