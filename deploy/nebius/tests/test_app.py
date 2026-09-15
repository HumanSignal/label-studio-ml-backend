"""Real ML-backend protocol with an inexpensive model; no GPU or network."""

import importlib
import json
from pathlib import Path

import pytest
from app import create_app, validate_media_url

from label_studio_ml import api
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

FIXTURES = Path(__file__).parents[1] / 'fixtures'


class ProbeModel(LabelStudioMLBase):
    def predict(self, tasks, context=None, **kwargs):
        return ModelResponse(predictions=[{'result': [], 'model_version': 'probe'}])


@pytest.fixture
def client(monkeypatch):
    importlib.reload(api)
    monkeypatch.setenv('LABEL_STUDIO_URL', 'https://label.example')
    monkeypatch.setenv('LABEL_STUDIO_PROJECT_ID', '1')
    monkeypatch.setenv('LABEL_STUDIO_API_KEY', 'local-test-only')
    monkeypatch.delenv('BASIC_AUTH_USER', raising=False)
    monkeypatch.delenv('BASIC_AUTH_PASS', raising=False)
    return create_app(ProbeModel).test_client()


@pytest.fixture
def payload():
    return json.loads((FIXTURES / 'predict.json').read_text())


def test_connect_setup_and_interactive_prediction(client, payload):
    assert client.get('/health').json['status'] == 'UP'
    response = client.post('/setup', json={'project': '1.1234', 'schema': payload['label_config']})
    assert response.status_code == 200
    assert response.json['model_version'] == '0.0.1'
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    assert len(response.json['results']) == 1
    assert response.json['results'][0]['result'] == []
    assert response.json['results'][0]['model_version'] == 'probe'


@pytest.mark.parametrize(
    'change',
    [
        {'tasks': []},
        {'tasks': [{'data': {}}, {'data': {}}]},
        {'project': '2.0'},
        {'tasks': [1]},
        {'params': []},
        {'label_config': None},
        {'params': {'context': []}},
        {'params': {'context': {'result': [None]}}},
    ],
)
def test_reject_invalid_or_cross_project_requests(client, payload, change):
    payload.update(change)
    assert client.post('/predict', json=payload).status_code == 400


@pytest.mark.parametrize('path', ['/webhook', '/train', '/delete', '/metrics', '/'])
def test_no_training_or_management_routes(client, path):
    assert client.post(path, json={}).status_code == 404


def test_bounded_json_body(client):
    body = b'x' * (16 * 1024 * 1024 + 1024)
    assert client.post('/predict', data=body, content_type='application/json').status_code == 413


def test_accepts_task_with_saved_brush_annotations(client, payload):
    # Label Studio sends saved annotations, drafts and predictions with interactive requests.
    rle = list(range(100_000))
    payload['tasks'][0]['annotations'] = [{'result': [{'type': 'brushlabels', 'value': {'rle': rle}}]}]
    assert len(json.dumps(payload)) > 512 * 1024
    assert client.post('/predict', json=payload).status_code == 200


@pytest.mark.parametrize('dimension', [None, -1, 0, 5000, '64'])
def test_reject_invalid_image_dimensions(client, payload, dimension):
    item = payload['params']['context']['result'][0]
    item['original_width'] = item['original_height'] = dimension
    assert client.post('/predict', json=payload).status_code == 400


@pytest.mark.parametrize(
    'url',
    [
        'http://label.example/data/upload/1/a.png',
        'https://other.example/data/upload/1/a.png',
        'file:///etc/passwd',
        '/etc/passwd',
        '//other.example/data/upload/a.png',
        '/data/upload/../secret',
        '/data/upload/%2e%2e/secret',
        '/data/upload/1/a.png?d=/etc/passwd',
        'https://user:password@label.example/data/upload/1/a.png',
    ],
)
def test_media_boundary_rejects_local_and_foreign_urls(url):
    with pytest.raises(ValueError):
        validate_media_url(url, 'https://label.example')


@pytest.mark.parametrize(
    'url',
    [
        '/data/upload/1/a.png',
        'https://label.example/data/upload/1/a.png',
        'https://label.example/storage-data/uploaded/?filepath=upload/1/a.png',
    ],
)
def test_supports_relative_and_absolute_uploads(url):
    validate_media_url(url, 'https://label.example')


def test_fail_closed_on_backend_basic_configuration(client, monkeypatch):
    monkeypatch.setenv('BASIC_AUTH_USER', 'wrong-auth-layer')
    with pytest.raises(ValueError, match='native Endpoint auth'):
        create_app(ProbeModel)


def test_setup_rejects_non_string_project(client, payload):
    assert client.post('/setup', json={'project': 1, 'schema': payload['label_config']}).status_code == 400


def test_accepts_client_null_context(client, payload):
    payload['params']['context'] = None
    assert client.post('/predict', json=payload).status_code == 200


def test_accepts_box_at_image_edge_with_float_rounding(client, payload):
    item = payload['params']['context']['result'][0]
    item['type'] = 'rectanglelabels'
    item['value'] = {'x': 6.4, 'y': 0, 'width': 93.60000000000001, 'height': 100, 'rectanglelabels': ['Object']}
    assert item['value']['x'] + item['value']['width'] > 100
    assert client.post('/predict', json=payload).status_code == 200


def test_rejects_box_outside_image(client, payload):
    item = payload['params']['context']['result'][0]
    item['type'] = 'rectanglelabels'
    item['value'] = {'x': 50, 'y': 50, 'width': 60, 'height': 10, 'rectanglelabels': ['Object']}
    assert client.post('/predict', json=payload).status_code == 400
