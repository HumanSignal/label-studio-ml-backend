"""LSTM-based time series segmenter.

This example implements a simple ML backend that trains a
recurrent neural network on labeled time series CSV files
and predicts segments for new tasks.
"""

import os
import io
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import label_studio_sdk
import tensorflow as tf
from tensorflow.keras import layers, models

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)

_model: Optional[models.Model] = None


class TimeSeriesSegmenter(LabelStudioMLBase):
    """Minimal LSTM-based segmenter for time series."""

    LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST", "http://localhost:8080")
    LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
    START_TRAINING_EACH_N_UPDATES = int(os.getenv("START_TRAINING_EACH_N_UPDATES", 10))
    MODEL_DIR = os.getenv("MODEL_DIR", ".")

    def setup(self):
        self.set("model_version", f"{self.__class__.__name__}-v0.0.1")

    # ------------------------------------------------------------------
    # Utility helpers

    def _build_model(self, n_channels: int, n_labels: int) -> models.Model:
        tf.keras.utils.set_random_seed(42)
        model = models.Sequential(
            [
                layers.Input(shape=(1, n_channels)),
                layers.LSTM(16),
                layers.Dense(n_labels, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _get_model(
        self, n_channels: int, n_labels: int, blank: bool = False
    ) -> models.Model:
        global _model
        if _model is not None and not blank:
            return _model
        model_path = os.path.join(self.MODEL_DIR, "model.keras")
        if not blank and os.path.exists(model_path):
            _model = models.load_model(model_path)
        else:
            _model = self._build_model(n_channels, n_labels)
        return _model

    def _get_labeling_params(self) -> Dict:
        from_name, to_name, value = self.label_interface.get_first_tag_occurence(
            "TimeSeriesLabels", "TimeSeries"
        )
        tag = self.label_interface.get_tag(from_name)
        labels = list(tag.labels)
        ts_tag = self.label_interface.get_tag(to_name)
        time_col = ts_tag.attr.get("timeColumn")

        import xml.etree.ElementTree as ET

        root = ET.fromstring(self.label_config)
        ts_elem = root.find(f".//TimeSeries[@name='{to_name}']")
        channels = [ch.attrib["column"] for ch in ts_elem.findall("Channel")]

        return {
            "from_name": from_name,
            "to_name": to_name,
            "value": value,
            "labels": labels,
            "time_col": time_col,
            "channels": channels,
        }

    def _read_csv(self, task: Dict, path: str) -> pd.DataFrame:
        csv_str = self.preload_task_data(task, path)
        return pd.read_csv(io.StringIO(csv_str))

    def _predict_task(self, task: Dict, model: models.Model, params: Dict) -> Dict:
        df = self._read_csv(task, task["data"][params["value"]])
        X = df[params["channels"]].values.reshape(-1, 1, len(params["channels"]))
        if len(X) == 0:
            return {}

        probs = model.predict(X, verbose=0)
        labels_idx = np.argmax(probs, axis=1)
        df["pred_label"] = [params["labels"][i] for i in labels_idx]
        df["score"] = probs[np.arange(len(probs)), labels_idx]

        segments = self._group_rows(df, params["time_col"])

        results = []
        avg_score = 0
        for seg in segments:
            score = float(np.mean(seg["scores"]))
            avg_score += score
            results.append(
                {
                    "from_name": params["from_name"],
                    "to_name": params["to_name"],
                    "type": "timeserieslabels",
                    "value": {
                        "start": seg["start"],
                        "end": seg["end"],
                        "instant": False,
                        "timeserieslabels": [seg["label"]],
                    },
                    "score": score,
                }
            )

        if not results:
            return {}

        return {
            "result": results,
            "score": avg_score / len(results),
            "model_version": self.get("model_version"),
        }

    def _group_rows(self, df: pd.DataFrame, time_col: str) -> List[Dict]:
        segments = []
        current = None
        for _, row in df.iterrows():
            label = row["pred_label"]
            if current and current["label"] == label:
                current["end"] = row[time_col]
                current["scores"].append(row["score"])
            else:
                if current:
                    segments.append(current)
                current = {
                    "label": label,
                    "start": row[time_col],
                    "end": row[time_col],
                    "scores": [row["score"]],
                }
        if current:
            segments.append(current)
        return segments

    def _collect_samples(
        self, tasks: List[Dict], params: Dict, label2idx: Dict[str, int]
    ) -> Tuple[List, List]:
        X, y = [], []
        for task in tasks:
            df = self._read_csv(task, task["data"][params["value"]])
            if df.empty:
                continue
            annotations = [a for a in task["annotations"] if a.get("result")]
            for ann in annotations:
                for r in ann["result"]:
                    if r["from_name"] != params["from_name"]:
                        continue
                    start = r["value"]["start"]
                    end = r["value"]["end"]
                    label = r["value"]["timeserieslabels"][0]
                    mask = (df[params["time_col"]] >= start) & (
                        df[params["time_col"]] <= end
                    )
                    seg = df.loc[mask, params["channels"]].values
                    X.extend(seg)
                    y.extend([label2idx[label]] * len(seg))
        return X, y

    def _save_model(self, model: models.Model) -> None:
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        model_path = os.path.join(self.MODEL_DIR, "model.keras")
        model.save(model_path)

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        params = self._get_labeling_params()
        model = self._get_model(len(params["channels"]), len(params["labels"]))
        predictions = [self._predict_task(task, model, params) for task in tasks]
        return ModelResponse(
            predictions=predictions, model_version=self.get("model_version")
        )

    def _get_tasks(self, project_id: int) -> List[Dict]:
        ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project_id)
        return project.get_labeled_tasks()

    def fit(self, event, data, **kwargs):
        if event not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED", "START_TRAINING"):
            logger.info("Skip training: event %s is not supported", event)
            return
        project_id = data["annotation"]["project"]
        tasks = self._get_tasks(project_id)
        if (
            len(tasks) % self.START_TRAINING_EACH_N_UPDATES != 0
            and event != "START_TRAINING"
        ):
            logger.info(
                "Skip training: %s tasks are not multiple of %s",
                len(tasks),
                self.START_TRAINING_EACH_N_UPDATES,
            )
            return
        params = self._get_labeling_params()
        label2idx = {l: i for i, l in enumerate(params["labels"])}

        X, y = self._collect_samples(tasks, params, label2idx)
        if not X:
            logger.warning("No data collected for training")
            return

        model = self._get_model(
            len(params["channels"]), len(params["labels"]), blank=True
        )
        X_arr = np.array(X).reshape(-1, 1, len(params["channels"]))
        y_arr = np.array(y)
        model.fit(X_arr, y_arr, epochs=10, verbose=0)
        self._save_model(model)
        global _model
        _model = None

        self._get_model(len(params["channels"]), len(params["labels"]))