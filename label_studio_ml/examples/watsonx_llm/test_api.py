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


@pytest.fixture
def client():
    from _wsgi import app
    with app.test_client() as client:
        yield client

def mock_create(**kwargs):
    return None

def test_predict(client):
    request = {
        'tasks': [{
            'data': {
                'text': 'Long text to be classified into one of the topics.'
            }
        }],
        # Your labeling configuration here
        'label_config':"""<View>
    <Style>
        .lsf-main-content.lsf-requesting .prompt::before { content: ' loading...'; color: #808080; }

        .text-container {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        font-family: 'Courier New', monospace;
        line-height: 1.6;
        font-size: 16px;
        }
    </Style>
    <Header value="Context:"/>
    <View className="text-container">
        <Text name="context" value="$text"/>
    </View>
    <Header value="Prompt:"/>
    <View className="prompt">
        <TextArea name="prompt"
                  toName="context"
                  rows="4"
                  editable="true"
                  maxSubmissions="1"
                  showSubmitButton="false"
                  placeholder="Type your prompt here then Shift+Enter..."
        />
    </View>
    <Header value="Response:"/>
    <TextArea name="response"
              toName="context"
              rows="4"
              editable="true"
              maxSubmissions="1"
              showSubmitButton="false"
              smart="false"
              placeholder="Generated response will appear here..."
    />
    <Header value="Overall response quality:"/>
    <Rating name="rating" toName="context"/>
</View>"""
    }
    with mock.patch('ibm_watsonx_ai.foundation_models.ModelInference.__init__') as mock_model:
        mock_model.return_value = None
        with mock.patch('ibm_watsonx_ai.foundation_models.ModelInference.generate') as mock_gen:
            mock_gen.return_value = {"results": [{"generated_text": "this is a test"}]}
            response = client.post('/predict', data=json.dumps(request), content_type='application/json')
            mock_model.assert_called()
            mock_gen.assert_called()
    assert response.status_code == 200

#TODO: finish tests