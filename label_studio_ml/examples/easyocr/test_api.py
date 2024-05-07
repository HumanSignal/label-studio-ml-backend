"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""
import os.path

import pytest
import json
from model import EasyOCR
import responses


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=EasyOCR)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def model_dir_env(tmp_path, monkeypatch):
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    monkeypatch.setattr(EasyOCR, 'MODEL_DIR', str(model_dir))
    return model_dir


@responses.activate
def test_predict(client, model_dir_env):
    responses.add(
        responses.GET,
        'http://test_predict.easyocr.ml-backend.com/image.jpeg',
        body=open(os.path.join(os.path.dirname(__file__), 'test_images', 'image.jpeg'), 'rb').read(),
        status=200
    )
    request = {
        'tasks': [{
            'data': {
                'image': 'http://test_predict.easyocr.ml-backend.com/image.jpeg'
            }
        }],
        # Your labeling configuration here
        'label_config': '''
        <View>
  <Image name="image" value="$image"/>

  <Labels name="label" toName="image">
    <Label value="Text" background="green"/>
    <Label value="Handwriting" background="blue"/>
  </Labels>

  <Rectangle name="bbox" toName="image" strokeWidth="3"/>
  <Polygon name="poly" toName="image" strokeWidth="3"/>

  <TextArea name="transcription" toName="image"
            editable="true"
            perRegion="true"
            required="true"
            maxSubmissions="1"
            rows="5"
            placeholder="Recognized Text"
            displayMode="region-list"
            />
</View>
'''
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    expected_texts = {
        'IZIN SOLUTIONS',
        'HlRlS',
        'РШIYH',
        'IMB?',
        'Kenapa Harus Punya IMB?',
        'Swipe'
    }
    texts_response = set()
    for r in response['results'][0]['result']:
        if r['from_name'] == 'transcription':
            assert r['value']['labels'][0] == 'Text'
            texts_response.add(r['value']['text'][0])
    assert texts_response == expected_texts
