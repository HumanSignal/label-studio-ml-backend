import os
import io
import pickle
import logging
import pandas as pd
import numpy as np
import label_studio_sdk

from typing import List, Dict, Optional
from sklearn.linear_model import LogisticRegression
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)

_model: Optional[LogisticRegression] = None


class TimeSeriesSegmenter(LabelStudioMLBase):
    """Simple time series segmentation using logistic regression."""

    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    MODEL_DIR = os.getenv('MODEL_DIR', '.')

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    # util functions
    def _get_model(self, blank: bool = False) -> LogisticRegression:
        global _model
        if _model is not None and not blank:
            return _model
        model_path = os.path.join(self.MODEL_DIR, 'model.pkl')
        if not blank and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                _model = pickle.load(f)
        else:
            _model = LogisticRegression(max_iter=1000)
        return _model

    def _get_labeling_params(self) -> Dict:
        from_name, to_name, value = self.label_interface.get_first_tag_occurence(
            'TimeSeriesLabels', 'TimeSeries')
        tag = self.label_interface.get_tag(from_name)
        labels = list(tag.labels)
        ts_tag = self.label_interface.get_tag(to_name)
        time_col = ts_tag.attr.get('timeColumn')
        # parse channel names from the original config since tag doesn't expose children
        import xml.etree.ElementTree as ET
        root = ET.fromstring(self.label_config)
        ts_elem = root.find(f".//TimeSeries[@name='{to_name}']")
        channels = [ch.attrib['column'] for ch in ts_elem.findall('Channel')]
        return {
            'from_name': from_name,
            'to_name': to_name,
            'value': value,
            'labels': labels,
            'time_col': time_col,
            'channels': channels
        }

    def _read_csv(self, task: Dict, path: str) -> pd.DataFrame:
        csv_str = self.preload_task_data(task, path)
        return pd.read_csv(io.StringIO(csv_str))

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        params = self._get_labeling_params()
        model = self._get_model()
        predictions = []
        for task in tasks:
            df = self._read_csv(task, task['data'][params['value']])
            X = df[params['channels']].values
            if len(X) == 0:
                predictions.append({})
                continue
            probs = model.predict_proba(X)
            labels_idx = np.argmax(probs, axis=1)
            df['pred_label'] = [params['labels'][i] for i in labels_idx]
            df['score'] = probs[np.arange(len(probs)), labels_idx]
            segments = []
            current = None
            for _, row in df.iterrows():
                label = row['pred_label']
                if current and current['label'] == label:
                    current['end'] = row[params['time_col']]
                    current['scores'].append(row['score'])
                else:
                    if current:
                        segments.append(current)
                    current = {
                        'label': label,
                        'start': row[params['time_col']],
                        'end': row[params['time_col']],
                        'scores': [row['score']]
                    }
            if current:
                segments.append(current)
            results = []
            avg_score = 0
            for seg in segments:
                score = float(np.mean(seg['scores']))
                avg_score += score
                results.append({
                    'from_name': params['from_name'],
                    'to_name': params['to_name'],
                    'type': 'timeserieslabels',
                    'value': {
                        'start': seg['start'],
                        'end': seg['end'],
                        'instant': False,
                        'timeserieslabels': [seg['label']]
                    },
                    'score': score
                })
            if results:
                predictions.append({
                    'result': results,
                    'score': avg_score / len(results),
                    'model_version': self.get('model_version')
                })
        return ModelResponse(predictions=predictions, model_version=self.get('model_version'))

    def _get_tasks(self, project_id: int) -> List[Dict]:
        ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project_id)
        return project.get_labeled_tasks()

    def fit(self, event, data, **kwargs):
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            logger.info(f"Skip training: event {event} is not supported")
            return
        project_id = data['annotation']['project']
        tasks = self._get_tasks(project_id)
        if len(tasks) % self.START_TRAINING_EACH_N_UPDATES != 0 and event != 'START_TRAINING':
            logger.info(
                f'Skip training: {len(tasks)} tasks are not multiple of {self.START_TRAINING_EACH_N_UPDATES}')
            return
        params = self._get_labeling_params()
        label2idx = {l: i for i, l in enumerate(params['labels'])}
        X, y = [], []
        for task in tasks:
            df = self._read_csv(task, task['data'][params['value']])
            if df.empty:
                continue
            annotations = [a for a in task['annotations'] if a.get('result')]
            for ann in annotations:
                for r in ann['result']:
                    if r['from_name'] != params['from_name']:
                        continue
                    start = r['value']['start']
                    end = r['value']['end']
                    label = r['value']['timeserieslabels'][0]
                    mask = (df[params['time_col']] >= start) & (df[params['time_col']] <= end)
                    seg = df.loc[mask, params['channels']].values
                    X.extend(seg)
                    y.extend([label2idx[label]] * len(seg))
        if not X:
            logger.warning('No data collected for training')
            return
        model = self._get_model(blank=True)
        model.fit(np.array(X), np.array(y))
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        model_path = os.path.join(self.MODEL_DIR, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        global _model
        _model = None
        self._get_model()

