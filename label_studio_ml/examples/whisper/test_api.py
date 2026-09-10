"""
Tests for the whisper backend. Install the test requirements first:

    ```bash
    pip install -r requirements-test.txt
    ```

Then run `pytest` in this directory. The whisper weights are never downloaded
here: `transcribe` is replaced with a stub so the tests check how segments are
turned into Label Studio regions, which is the part this backend owns.
"""

import json
from collections import namedtuple

import pytest
import model as model_module
from model import WhisperASR

Segment = namedtuple('Segment', ['start', 'end', 'text', 'avg_logprob'])

SEGMENTS = [
    Segment(0.0, 1.78, ' expected to be named in March.', -0.3),
    Segment(1.78, 4.04, ' It may be the most important appointment', -0.5),
]

FULL_CONFIG = """
<View>
  <Audio name="audio" value="$audio"/>
  <TextArea name="transcription" toName="audio" rows="4" editable="true" maxSubmissions="1"/>
</View>
"""

SEGMENT_CONFIG = """
<View>
  <Labels name="labels" toName="audio">
    <Label value="Speech"/>
    <Label value="Noise"/>
  </Labels>
  <Audio name="audio" value="$audio"/>
  <TextArea name="transcription" toName="audio" rows="2" editable="true" perRegion="true" required="true"/>
</View>
"""


class _FakeWhisper:
    def transcribe(self, path, **kwargs):
        return iter(SEGMENTS), None


@pytest.fixture(autouse=True)
def fake_whisper(monkeypatch):
    monkeypatch.setattr(model_module, '_get_model', lambda: _FakeWhisper())
    monkeypatch.setattr(WhisperASR, 'get_local_path', lambda self, url, **kwargs: '/tmp/audio.wav')


@pytest.fixture
def client():
    from _wsgi import init_app

    app = init_app(model_class=WhisperASR)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def _predict(client, label_config):
    request = {
        'tasks': [{'id': 1, 'data': {'audio': 'http://example.com/audio.wav'}}],
        'label_config': label_config,
    }
    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200
    return json.loads(response.data)['results'][0]


def test_predict_whole_file(client):
    prediction = _predict(client, FULL_CONFIG)

    assert prediction['result'] == [
        {
            'from_name': 'transcription',
            'to_name': 'audio',
            'type': 'textarea',
            'value': {'text': ['expected to be named in March. It may be the most important appointment']},
        }
    ]
    assert 0 < prediction['score'] <= 1


def test_predict_segments(client):
    prediction = _predict(client, SEGMENT_CONFIG)
    result = prediction['result']

    assert len(result) == 2 * len(SEGMENTS)

    for segment, region, transcript in zip(SEGMENTS, result[::2], result[1::2]):
        # the region and its transcript must carry the same id, otherwise Label
        # Studio shows the text as a region of its own
        assert region['id'] == transcript['id']
        assert region['type'] == 'labels'
        assert region['from_name'] == 'labels'
        assert region['value'] == {'start': segment.start, 'end': segment.end, 'labels': ['Speech']}
        assert transcript['type'] == 'textarea'
        assert transcript['from_name'] == 'transcription'
        assert transcript['value'] == {
            'start': segment.start,
            'end': segment.end,
            'text': [segment.text.strip()],
        }

    assert len({item['id'] for item in result}) == len(SEGMENTS)


def test_segment_label_falls_back_to_the_first_label(client, monkeypatch):
    monkeypatch.setattr(model_module, 'SEGMENT_LABEL', 'Missing')

    prediction = _predict(client, SEGMENT_CONFIG)

    assert prediction['result'][0]['value']['labels'] == ['Speech']


def test_no_speech_gives_an_empty_result(client, monkeypatch):
    class _Silent:
        def transcribe(self, path, **kwargs):
            return iter([]), None

    monkeypatch.setattr(model_module, '_get_model', lambda: _Silent())

    prediction = _predict(client, SEGMENT_CONFIG)

    assert prediction['result'] == []
    assert prediction['score'] == 0.0
