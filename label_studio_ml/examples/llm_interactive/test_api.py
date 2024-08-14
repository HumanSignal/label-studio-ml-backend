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
import unittest.mock as mock
from model import OpenAIInteractive

# This is your mocked completion response
mocked_completion_response = {
    "choices": [
        {
            "message": {
                "content": "Mocked response"
            }
        }
    ]
}


class Message:
    def __init__(self, content):
        self.content = content


class Choice:
    def __init__(self, message):
        self.message = message


class MockedCompletionResponse:
    def __init__(self, choices):
        self.choices = choices


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=OpenAIInteractive)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_default_prompt():
    with mock.patch.object(OpenAIInteractive, 'DEFAULT_PROMPT', new='''\
Classify text into different categories. Start each category prediction in a new line.
Text: {text}
Categories: {labels}'''):
        yield


@pytest.fixture
def mock_not_use_internal_prompt_template():
    with mock.patch.object(OpenAIInteractive, 'USE_INTERNAL_PROMPT_TEMPLATE', new=False):
        yield


def test_predict(client, mock_default_prompt, mock_not_use_internal_prompt_template):
    request = {
        'tasks': [{
            'data': {
                'text': 'Long text to be classified into one of the topics.'
            }
        }],
        # Your labeling configuration here
        'label_config':
            '<View>'
            '<Text name="text" value="$text" />'
            '<TextArea name="prompt" toName="text" required="true" />'
            '<Choices name="topic" toName="text" choice="single">'
            '<Choice value="sports" />'
            '<Choice value="politics" />'
            '<Choice value="technology" />'
            '</Choices>'
            '</View>'
    }

    expected_response_results = [{
        # In the current implementation, predictions go first, then the prompt at the end to populate the text area
        'result': [{
            'from_name': 'topic',
            'to_name': 'text',
            'type': 'choices',
            'value': {'choices': ['technology']}
        }, {
            'from_name': 'prompt',
            'to_name': 'text',
            'type': 'textarea',
            'value': {
                'text': [
                    'Classify text into different categories. Start each category prediction in a new line.\nText: {text}\nCategories: {labels}'
                ]
            }
        }]
    }]

    def mock_create(**kwargs):
        assert kwargs["messages"][0]["content"] == '''\
Classify text into different categories. Start each category prediction in a new line.
Text: Long text to be classified into one of the topics.
Categories: ['sports', 'politics', 'technology']'''
        assert kwargs["model"] == "gpt-3.5-turbo"
        assert kwargs['n'] == 1
        assert kwargs['temperature'] == 0.7
        return MockedCompletionResponse(
            choices=[
                Choice(message=Message(content="tech"))
            ]
        )

    with mock.patch('openai.resources.chat.completions.Completions.create', side_effect=mock_create) as mock_chat:
        # mock_chat.completions.create = mock_create
        response = client.post('/predict', data=json.dumps(request), content_type='application/json')
        # ensure mock was called
        mock_chat.assert_called_once()
    assert response.status_code == 200
    response = json.loads(response.data)
    assert len(response['results']) == len(expected_response_results)
    assert response['results'][0]['result'][0]['from_name'] == expected_response_results[0]['result'][0]['from_name']
    assert response['results'][0]['result'][0]['to_name'] == expected_response_results[0]['result'][0]['to_name']
    assert response['results'][0]['result'][0]['type'] == expected_response_results[0]['result'][0]['type']
    assert response['results'][0]['result'][0]['value'] == expected_response_results[0]['result'][0]['value']

    assert response['results'][0]['result'][1]['from_name'] == expected_response_results[0]['result'][1]['from_name']
    assert response['results'][0]['result'][1]['to_name'] == expected_response_results[0]['result'][1]['to_name']
    assert response['results'][0]['result'][1]['type'] == expected_response_results[0]['result'][1]['type']
    assert response['results'][0]['result'][1]['value'] == expected_response_results[0]['result'][1]['value']


# TODO: add tests for interactive modes, choices and textareas