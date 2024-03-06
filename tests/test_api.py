
import pytest
from label_studio_ml.api import _server

@pytest.fixture
def client():
    with _server.test_client() as client:
        yield client

def test_api(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.get_json() ==  {'model_class': 'LabelStudioMLBase', 'status': 'UP'}

def test_metrics(client):
    response = client.get('/metrics')
    assert response.status_code == 200

def test_predict(client):
    response = client.post('/predict', json={
        'tasks': {'id': 1},
        'label_config': '<View></View>',
        'project': '1.1000000000',
        'params': {
            'context': {},
        },
    })
    
    assert response.status_code == 200

def test_setup(client):
    response = client.post('/setup', json={
        'project': '1.1000000000',
        'schema': '<View></View>',
        'extra_params': {}
    })
    
    assert response.status_code == 200

def test_webhook(client):
    response = client.post('/webhook', json={
        'action': 'ANNOTATION_CREATED',
        'project': {
            'id': 1,
            'label_config': '<View></View>'
        }
    })
    
    assert response.status_code == 201

