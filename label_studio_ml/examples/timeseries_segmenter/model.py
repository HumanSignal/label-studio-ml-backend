"""Random forest based time series segmenter.

This example demonstrates a small yet functional ML backend that trains a
classifier on labeled time series CSV files and predicts segments for new
from sklearn.ensemble import RandomForestClassifier
_model: Optional[RandomForestClassifier] = None
    """Simple random forest based segmenter for time series."""

    def _get_model(self, blank: bool = False) -> RandomForestClassifier:
            _model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    def _predict_task(self, task: Dict, model: RandomForestClassifier, params: Dict) -> Dict:
import logging
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import label_studio_sdk

from sklearn.linear_model import LogisticRegression
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)

# Cached model instance to avoid reloading the pickle on each request.
_model: Optional[LogisticRegression] = None


class TimeSeriesSegmenter(LabelStudioMLBase):
    """Simple logistic regression based segmenter for time series."""

    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
    START_TRAINING_EACH_N_UPDATES = int(
        os.getenv('START_TRAINING_EACH_N_UPDATES', 10)
    )
    MODEL_DIR = os.getenv('MODEL_DIR', '.')

    def setup(self):
        """Initialize model metadata."""
        self.set('model_version', f'{self.__class__.__name__}-v0.0.1')

    # ------------------------------------------------------------------
    # Utility helpers

    def _get_model(self, blank: bool = False) -> LogisticRegression:
        """Return a trained model or create a fresh one if needed."""
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
        """Return tag names and channel information from the labeling config."""
        (
            from_name,
            to_name,
            value,
        ) = self.label_interface.get_first_tag_occurence(
            'TimeSeriesLabels', 'TimeSeries'
        )
        tag = self.label_interface.get_tag(from_name)
        labels = list(tag.labels)
        ts_tag = self.label_interface.get_tag(to_name)
        time_col = ts_tag.attr.get('timeColumn')
        # Parse channel names from the original XML because TimeSeries tag
        # does not expose its children via label-studio's interface
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
            'channels': channels,
        }

    def _read_csv(self, task: Dict, path: str) -> pd.DataFrame:
        """Load a CSV referenced by the task using Label Studio utilities."""
        csv_str = self.preload_task_data(task, path)
        return pd.read_csv(io.StringIO(csv_str))

    def _predict_task(
        self, task: Dict, model: LogisticRegression, params: Dict
    ) -> Dict:
        """Return Label Studio-style prediction for a single task."""
        df = self._read_csv(task, task['data'][params['value']])

        # Vector of sensor values per row
        X = df[params['channels']].values
        if len(X) == 0:
            return {}

        # Predict label probabilities for each row
        probs = model.predict_proba(X)
        labels_idx = np.argmax(probs, axis=1)
        df['pred_label'] = [params['labels'][i] for i in labels_idx]
        df['score'] = probs[np.arange(len(probs)), labels_idx]

        segments = self._group_rows(df, params['time_col'])

        results = []
        avg_score = 0
        for seg in segments:
            score = float(np.mean(seg['scores']))
            avg_score += score
            results.append(
                {
                    'from_name': params['from_name'],
                    'to_name': params['to_name'],
                    'type': 'timeserieslabels',
                    'value': {
                        'start': seg['start'],
                        'end': seg['end'],
                        'instant': False,
                        'timeserieslabels': [seg['label']],
                    },
                    'score': score,
                }
            )

        if not results:
            return {}

        return {
            'result': results,
            'score': avg_score / len(results),
            'model_version': self.get('model_version'),
        }

    def _group_rows(self, df: pd.DataFrame, time_col: str) -> List[Dict]:
        """Group consecutive rows with the same predicted label."""
        segments = []
        current = None
        for _, row in df.iterrows():
            label = row['pred_label']
            if current and current['label'] == label:
                current['end'] = row[time_col]
                current['scores'].append(row['score'])
            else:
    def _save_model(self, model: RandomForestClassifier) -> None:
                    segments.append(current)
                current = {
                    'label': label,
                    'start': row[time_col],
                    'end': row[time_col],
                    'scores': [row['score']],
                }
        if current:
            segments.append(current)
        return segments

    def _collect_samples(
        self, tasks: List[Dict], params: Dict, label2idx: Dict[str, int]
    ) -> Tuple[List, List]:
        """Return feature matrix and label vector built from all labeled tasks."""
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
                    mask = (df[params['time_col']] >= start) & (
                        df[params['time_col']] <= end
                    )
                    seg = df.loc[mask, params['channels']].values
                    X.extend(seg)
                    y.extend([label2idx[label]] * len(seg))
        return X, y

    def _save_model(self, model: LogisticRegression) -> None:
        """Persist trained model to disk."""
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        model_path = os.path.join(self.MODEL_DIR, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Return time series segments predicted for the given tasks."""
        params = self._get_labeling_params()
        model = self._get_model()
        predictions = [
            self._predict_task(task, model, params) for task in tasks
        ]

        return ModelResponse(
            predictions=predictions, model_version=self.get('model_version')
        )

    def _get_tasks(self, project_id: int) -> List[Dict]:
        """Fetch labeled tasks from Label Studio."""
        ls = label_studio_sdk.Client(
            self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY
        )
        project = ls.get_project(id=project_id)
        return project.get_labeled_tasks()

    def fit(self, event, data, **kwargs):
        """Train the model on all labeled segments."""
        if event not in (
            'ANNOTATION_CREATED',
            'ANNOTATION_UPDATED',
            'START_TRAINING',
        ):
            logger.info('Skip training: event %s is not supported', event)
            return

        project_id = data['annotation']['project']
        tasks = self._get_tasks(project_id)
        if (
            len(tasks) % self.START_TRAINING_EACH_N_UPDATES != 0
            and event != 'START_TRAINING'
        ):
            logger.info(
                f'Skip training: {len(tasks)} tasks are not multiple of {self.START_TRAINING_EACH_N_UPDATES}'
            )
            return

        params = self._get_labeling_params()
        label2idx = {l: i for i, l in enumerate(params['labels'])}

        X, y = self._collect_samples(tasks, params, label2idx)
        if not X:
            logger.warning('No data collected for training')
            return

        model = self._get_model(blank=True)
        model.fit(np.array(X), np.array(y))
        self._save_model(model)
        global _model
        _model = None  # reload on next predict
        self._get_model()
