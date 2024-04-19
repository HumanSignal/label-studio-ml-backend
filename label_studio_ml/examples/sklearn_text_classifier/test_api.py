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
from model import SklearnTextClassifier


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=SklearnTextClassifier)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = {
        'tasks': [{
            'data': {
                'text': 'That is positive!'
            }
        }],
        # Your labeling configuration here
        'label_config': '''
        <View>
  <Text name="text" value="$text"/>
  <View style="box-shadow: 2px 2px 5px #999;
               padding: 20px; margin-top: 2em;
               border-radius: 5px;">
    <Header value="Choose text sentiment"/>
    <Choices name="sentiment" toName="text"
             choice="single" showInLine="true">
      <Choice value="Positive"/>
      <Choice value="Negative"/>
      <Choice value="Neutral"/>
    </Choices>
  </View>
</View>
        '''
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    assert response['results'][0]['model_version'] == 'SklearnTextClassifier-v0.0.1'
    assert response['results'][0]['result'][0]['value']['choices'][0] == 'Positive'
