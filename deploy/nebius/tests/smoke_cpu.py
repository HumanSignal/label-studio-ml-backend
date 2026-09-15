"""Run inside the built GPU image with DEVICE=cpu; media transport is mocked."""

import io
import json
from pathlib import Path

import numpy as np
import requests_mock
import torch
from app import create_app
from label_studio_sdk.converter.brush import decode_rle
from PIL import Image, ImageDraw

torch.set_num_threads(2)
image = Image.new('RGB', (64, 64), 'white')
ImageDraw.Draw(image).rectangle((16, 16, 48, 48), fill='blue')
encoded = io.BytesIO()
image.save(encoded, format='PNG')
payload = json.loads(Path('/fixtures/predict.json').read_text())
client = create_app().test_client()
assert client.get('/health').status_code == 200
assert client.post('/setup', json={'project': '1.0', 'schema': payload['label_config']}).status_code == 200
with requests_mock.Mocker() as media:
    media.get(requests_mock.ANY, content=encoded.getvalue())
    response = client.post('/predict', json=payload)
    assert response.status_code == 200, response.json
    assert media.last_request.headers['Authorization'] == 'Token cpu-test-credential'
result = response.json['results'][0]
region = result['result'][0]
assert region['type'] == 'brushlabels'
assert region['from_name'] == 'mask' and region['to_name'] == 'image'
assert region['original_width'] == region['original_height'] == 64
mask = np.asarray(decode_rle(region['value']['rle'])).reshape(64, 64, 4)
assert np.count_nonzero(mask[:, :, 3]) > 0
assert 0 <= result['score'] <= 1
print(
    json.dumps(
        {
            'status': response.status_code,
            'model_version': result['model_version'],
            'mask_nonzero_pixels': int(np.count_nonzero(mask[:, :, 3])),
            'score': result['score'],
            'torch': torch.__version__,
            'device': 'cpu',
            'media': 'mocked authenticated upload',
        }
    )
)

# A fresh denied upload must fail without serializing the media URL/credential.
payload['tasks'][0]['data']['image'] = '/storage-data/uploaded/?filepath=upload/1/denied.png&signature=private-test-marker'
with requests_mock.Mocker() as media:
    media.get(requests_mock.ANY, status_code=403)
    denied = client.post('/predict', json=payload)
    assert media.called, 'the denied upload must reach the media download'
    assert denied.status_code == 500
    assert 'private-test-marker' not in denied.get_data(as_text=True)
    assert 'cpu-test-credential' not in denied.get_data(as_text=True)
print('Denied-media error redaction: passed')
