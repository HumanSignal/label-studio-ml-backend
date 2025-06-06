import os
import json
import sys
from unittest.mock import patch

import pytest

# Skip tests if scikit-learn isn't available
pytest.importorskip('sklearn')

TEST_DIR = os.path.dirname(__file__)
EXAMPLE_DIR = os.path.abspath(os.path.join(TEST_DIR, '..'))
REPO_ROOT = os.path.abspath(os.path.join(TEST_DIR, '../../../..'))
for path in (EXAMPLE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from label_studio_ml.examples.timeseries_segmenter.model import (
        TimeSeriesSegmenter,
    )
    from label_studio_ml.examples.timeseries_segmenter._wsgi import init_app
except ImportError:  # running inside example Docker image
    from model import TimeSeriesSegmenter
    from _wsgi import init_app

TEST_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(TEST_DIR, 'time_series.csv')

LABEL_CONFIG = """
<View>
  <TimeSeriesLabels name=\"label\" toName=\"ts\">
    <Label value=\"Run\"/>
    <Label value=\"Walk\"/>
  </TimeSeriesLabels>
  <TimeSeries name=\"ts\" valueType=\"url\" value=\"$csv_url\" timeColumn=\"time\">
    <Channel column=\"sensorone\" />
    <Channel column=\"sensortwo\" />
  </TimeSeries>
</View>
"""


@pytest.fixture
def client():
    app = init_app(model_class=TimeSeriesSegmenter)
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def make_task():
    return {
        'id': 1,
        'data': {'csv_url': CSV_PATH},
        'annotations': [
            {
                'result': [
                    {
                        'from_name': 'label',
                        'to_name': 'ts',
                        'type': 'timeserieslabels',
                        'value': {
                            'start': 0,
                            'end': 85,
                            'instant': False,
                            'timeserieslabels': ['Run'],
                        },
                    },
                    {
                        'from_name': 'label',
                        'to_name': 'ts',
                        'type': 'timeserieslabels',
        segs = results[0]["result"]
        assert len(segs) == 2
        assert segs[0]["value"]["start"] == 0
        assert segs[0]["value"]["timeserieslabels"] == ["Run"]
        assert segs[1]["value"]["timeserieslabels"] == ["Walk"]
        assert 80 <= segs[1]["value"]["start"] <= 90
        assert segs[1]["value"]["end"] == 99
                            'start': 85,
                            'end': 99,
                            'instant': False,
                            'timeserieslabels': ['Walk'],
                        },
                    },
                ]
            }
        ],
    }


def test_train_and_predict(client, tmp_path):
    setup_data = {'schema': LABEL_CONFIG, 'project': '1'}
    resp = client.post(
        '/setup', data=json.dumps(setup_data), content_type='application/json'
    )
    assert resp.status_code == 200

    task = make_task()

    def fake_preload(self, task, value=None, read_file=True):
        return open(value).read()

    with patch.object(
        TimeSeriesSegmenter, '_get_tasks', return_value=[task]
    ), patch.object(
        TimeSeriesSegmenter, 'preload_task_data', new=fake_preload
    ):
        data = {
            'action': 'START_TRAINING',
            'annotation': {'project': 1},
            'project': {'id': 1, 'label_config': LABEL_CONFIG},
        }
        resp = client.post(
            '/webhook', data=json.dumps(data), content_type='application/json'
        )
        assert resp.status_code == 201

        predict_data = {
            'tasks': [dict(id=1, data={'csv_url': CSV_PATH})],
            'label_config': LABEL_CONFIG,
            'project': '1',
        }
        resp = client.post(
            '/predict',
            data=json.dumps(predict_data),
            content_type='application/json',
        )
        assert resp.status_code == 200
        results = resp.json['results']
        assert len(results) == 1
        assert len(results[0]['result']) >= 1
