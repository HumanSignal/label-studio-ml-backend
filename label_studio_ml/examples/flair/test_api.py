import pytest
import json
from model import NewModel


@pytest.fixture
def client():
    from _wsgi import init_app
    app = init_app(model_class=NewModel)
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
    assert data == {'results': [{'model_version': '0.0.1', 'result': [{'from_name': 'ner_tags', 'score': 0.9969204664230347, 'to_name': 'text', 'type': 'labels', 'value': {'end': 5, 'labels': ['ORG'], 'start': 0, 'text': 'Apple'}}, {'from_name': 'ner_tags', 'score': 0.5269200205802917, 'to_name': 'text', 'type': 'labels', 'value': {'end': 40, 'labels': ['MISC'], 'start': 33, 'text': 'MacBook'}}], 'score': 0.5269200205802917}]}
