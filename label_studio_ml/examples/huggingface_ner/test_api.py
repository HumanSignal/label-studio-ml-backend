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
from model import HuggingFaceNER
import unittest.mock as mock


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=HuggingFaceNER)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = {
        'tasks': [{
            'data': {
                'text': 'President Obama is speaking at 3pm today in New York.'
            }
        }],
        # Your labeling configuration here
        'label_config': '''
        <View>
        <Text name="text" value="$text"/>
        <Labels name="ner" toName="text">
            <Label value="Person"/>
            <Label value="Location"/>
            <Label value="Time"/>
        </Labels>
        </View>
        '''
    }

    expected_response = {
        'results': [{
            'model_version': 'HuggingFaceNER-v0.0.1',
            'result': [{
                'from_name': 'ner',
                'score': 0.9974774718284607,
                'to_name': 'text',
                'type': 'labels',
                'value': {
                    'end': 15,
                    'labels': ['PER'],
                    'start': 10}},
                {'from_name': 'ner',
                 'score': 0.9994751214981079,
                 'to_name': 'text',
                 'type': 'labels',
                 'value': {'end': 52,
                           'labels': ['LOC'],
                           'start': 44}}],
            'score': 0.9984762966632843}]
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    response = json.loads(response.data)
    assert response['results'][0]['model_version'] == expected_response['results'][0]['model_version']
    assert response['results'][0]['result'][0]['value'] == expected_response['results'][0]['result'][0]['value']
    assert response['results'][0]['result'][1]['value'] == expected_response['results'][0]['result'][1]['value']


# mock response of label_studio_sdk.Project.get_labeled_tasks() and return the list of Label Studio tasks with NER annotations
def get_labeled_tasks_mock(self, project_id):
    return [
        {
            'id': '0',
            'data': {'text': 'President Obama is speaking at 3pm today in New York'},
            'annotations': [
                {
                    'result': [
                        {
                            'from_name': 'ner',
                            'to_name': 'text',
                            'type': 'labels',
                            'value': {
                                'start': 10,
                                'end': 15,
                                'labels': ['Person']
                            }
                        },
                        {
                            'from_name': 'ner',
                            'to_name': 'text',
                            'type': 'labels',
                            'value': {
                                'start': 44,
                                'end': 52,
                                'labels': ['Location']
                            }
                        },
                        {
                            'from_name': 'ner',
                            'to_name': 'text',
                            'type': 'labels',
                            'value': {
                                'start': 31,
                                'end': 40,
                                'labels': ['Time']
                            }
                        }
                    ]
                }
            ]
        }
    ]


# mock NewModel.START_TRAINING_EACH_N_UPDATES to 1 to trigger training in the test
@pytest.fixture
def mock_start_training():
    with mock.patch.object(HuggingFaceNER, 'START_TRAINING_EACH_N_UPDATES', new=1):
        yield


@pytest.fixture
def mock_get_labeled_tasks():
    with mock.patch.object(HuggingFaceNER, '_get_tasks', new=get_labeled_tasks_mock):
        yield


@pytest.fixture
def mock_baseline_model_name_for_train():
    with mock.patch('model.BASELINE_MODEL_NAME', new='distilbert/distilbert-base-uncased'):
        yield


def test_fit(client, mock_get_labeled_tasks, mock_start_training, mock_baseline_model_name_for_train):
    request = {
        'action': 'ANNOTATION_CREATED',
        'project': {
            'id': 12345,
            'label_config': '''
        <View>
        <Text name="text" value="$text"/>
        <Labels name="ner" toName="text">
            <Label value="Person"/>
            <Label value="Location"/>
            <Label value="Time"/>
        </Labels>
        </View>
        '''
        },
        'annotation': {
            'project': 12345
        }
    }

    response = client.post('/webhook', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 201

    # assert new model is created in ./results/finetuned_model directory
    import os
    from model import MODEL_DIR
    results_dir = os.path.join(MODEL_DIR, 'finetuned_model')
    assert os.path.exists(os.path.join(results_dir, 'pytorch_model.bin'))

    # now let's test whether the model is trained by running predict
    request = {
        'tasks': [{
            'data': {
                'text': 'President Obama is speaking at 3pm today in New York.'
            }
        }],
        # Your labeling configuration here
        'label_config': '''
        <View>
        <Text name="text" value="$text"/>
        <Labels name="ner" toName="text">
            <Label value="Person"/>
            <Label value="Location"/>
            <Label value="Time"/>
        </Labels>
        </View>
        '''
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200

    # TODO: we also need to check the prediction results to make sure the model is trained correctly
    # but the training needs to be deterministic to make the test stable
    # assert response is as expected

    # remove './results/finetuned_model' directory after testing
    import shutil
    shutil.rmtree(results_dir)
