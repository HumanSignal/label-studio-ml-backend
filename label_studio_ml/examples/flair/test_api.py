import pytest
import json
from model import Flair


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=Flair)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = {
        'tasks': [{
            'data': {
                'text': 'Apple reveals the new ultra-thin MacBook laptop (yep, its thinner than the Air)'
            }
        }],
        'label_config': '''
        <View>
  <Labels name="ner_tags" toName="text">
    <Label value="label_1" background="#FFA39E"/>
    <Label value="label_2" background="#D4380D"/>
    <Label value="label_3" background="#FFC069"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
        '''
    }
    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['results'][0]['model_version'] == 'Flair-v0.0.1'
    assert data['results'][0]['result'][0]['value']['text'] == 'Apple'
    assert data['results'][0]['result'][0]['value']['labels'] == ['ORG']
    assert data['results'][0]['result'][1]['value']['text'] == 'MacBook'
    assert data['results'][0]['result'][1]['value']['labels'] == ['MISC']
