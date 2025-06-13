"""PyTorch LSTM-based time series segmenter.

This example implements a ML backend that trains a
recurrent neural network on labeled time series CSV files
and predicts segments for new tasks using proper temporal modeling.
"""

import os
import io
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import label_studio_sdk

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from neural_nets import TimeSeriesLSTM

logger = logging.getLogger(__name__)

# Project-specific model cache
_models: Dict[int, TimeSeriesLSTM] = {}


class TimeSeriesSegmenter(LabelStudioMLBase):
    """PyTorch LSTM-based segmenter for time series with proper temporal modeling."""

    LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST", "http://localhost:8080")
    LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
    START_TRAINING_EACH_N_UPDATES = int(os.getenv("START_TRAINING_EACH_N_UPDATES", 1))
    TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", 1000))
    SEQUENCE_SIZE = int(os.getenv("SEQUENCE_SIZE", 50))
    HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", 64))
    MODEL_DIR = os.getenv("MODEL_DIR", ".")
    
    # New parameters for handling imbalanced data
    BALANCED_ACCURACY_THRESHOLD = float(os.getenv("BALANCED_ACCURACY_THRESHOLD", 0.85))
    MIN_CLASS_F1_THRESHOLD = float(os.getenv("MIN_CLASS_F1_THRESHOLD", 0.70))
    USE_CLASS_WEIGHTS = os.getenv("USE_CLASS_WEIGHTS", "true").lower() == "true"

    def setup(self):
        logger.info("Setting up TimeSeriesSegmenter model")
        self.set("model_version", f"{self.__class__.__name__}-v0.0.3")
        logger.info(f"Model version set to: {self.get('model_version')}")
        logger.info(f"Model directory: {self.MODEL_DIR}")
        logger.info(f"Training trigger: every {self.START_TRAINING_EACH_N_UPDATES} updates")
        logger.info(f"Sequence size: {self.SEQUENCE_SIZE}, Hidden size: {self.HIDDEN_SIZE}")
        logger.info(f"Imbalanced data handling: Class weights={self.USE_CLASS_WEIGHTS}, "
                   f"Balanced accuracy threshold={self.BALANCED_ACCURACY_THRESHOLD}, "
                   f"Min class F1 threshold={self.MIN_CLASS_F1_THRESHOLD}")

    # ------------------------------------------------------------------
    # Utility helpers

    def _build_model(self, n_channels: int, n_labels: int) -> TimeSeriesLSTM:
        logger.info(f"Building new TimeSeriesLSTM model with {n_channels} channels and {n_labels} labels")
        
        model = TimeSeriesLSTM(
            input_size=n_channels,
            output_size=n_labels,
            sequence_size=self.SEQUENCE_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=2,
            learning_rate=1e-3,
            dropout_rate=0.3
        )
        
        logger.info("Model built successfully")
        return model

    def _get_model(
        self, n_channels: int, n_labels: int, project_id: Optional[int] = None, blank: bool = False
    ) -> TimeSeriesLSTM:
        global _models
        
        # Use default project_id if not provided (for backward compatibility)
        if project_id is None:
            project_id = 0
            logger.warning("No project_id provided, using default project_id=0")
        
        # Check if we have this model in memory
        if project_id in _models and not blank:
            logger.info(f"Using existing model for project {project_id} from memory")
            return _models[project_id]
        
        # Try to load from disk
        raw_model_path = os.path.join(self.MODEL_DIR, f"model_project_{project_id}.pt")
        model_path = os.path.normpath(raw_model_path)
        
        # Ensure the normalized path is within the intended directory
        if not model_path.startswith(os.path.abspath(self.MODEL_DIR)):
            raise ValueError(f"Invalid model path: {model_path}")
        
        if not blank and os.path.exists(model_path):
            logger.info(f"Loading saved model for project {project_id} from {model_path}")
            try:
                model = TimeSeriesLSTM.load_model(model_path)
                _models[project_id] = model
                logger.info(f"Model for project {project_id} loaded successfully from disk")
                return model
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}. Creating new model.")
                # Remove corrupted model file
                try:
                    os.remove(model_path)
                    logger.info(f"Removed corrupted model file: {model_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not remove corrupted model file: {cleanup_error}")
        
        # Create new model
        logger.info(f"Creating new model for project {project_id}")
        model = self._build_model(n_channels, n_labels)
        _models[project_id] = model
        
        return model

    def _get_labeling_params(self) -> Dict:
        logger.debug("Extracting labeling parameters from label config")
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

        # Add background class as index 0
        all_labels = ["__background__"] + labels

        params = {
            "from_name": from_name,
            "to_name": to_name,
            "value": value,
            "labels": labels,  # Original labels for UI display
            "all_labels": all_labels,  # All labels including background for training
            "time_col": time_col,
            "channels": channels,
        }
        
        logger.info(f"Labeling parameters - Labels: {labels}, Channels: {channels}, Time column: {time_col}")
        logger.info(f"Training labels (with background): {all_labels}")
        return params

    def _read_csv(self, task: Dict, path: str, params: dict) -> Tuple[pd.DataFrame, str]:
        logger.debug(f"Reading CSV data from path: {path}")
        csv_str = self.preload_task_data(task, value=path)
        df = pd.read_csv(io.StringIO(csv_str))
        logger.debug(f"CSV loaded with shape: {df.shape}")

        # generate index-time if time_col is not in csv
        time_col = params['time_col']
        if time_col is None:
            time_col = 'time_autogenerated'
            first_channel_values = df[params['channels'][0]]
            df[time_col] = range(len(first_channel_values))
            logger.warning("CSV file doesn't contain time column, it was autogenerated")

        return df, time_col

    def _predict_task(self, task: Dict, model: TimeSeriesLSTM, params: Dict) -> Dict:
        task_id = task.get("id", "unknown")
        logger.info(f"Predicting task {task_id}")
        
        df, time_col = self._read_csv(task, task["data"][params["value"]], params)
        
        if df.empty:
            logger.warning(f"Task {task_id}: No data found for prediction")
            return {}

        # Extract features
        X = df[params["channels"]].values.astype(np.float32)
        logger.debug(f"Task {task_id}: Input shape for prediction: {X.shape}")
        
        # Get predictions
        with torch.no_grad():
            probs = model.predict(X)  # Shape: (seq_len, n_classes)
            if len(probs) == 0:
                logger.warning(f"Task {task_id}: No predictions generated")
                return {}
                
            labels_idx = torch.argmax(probs, dim=1).cpu().numpy()
            scores = torch.max(probs, dim=1)[0].cpu().numpy()

        # Map predictions back to original labels
        df["pred_label_idx"] = labels_idx
        df["pred_label"] = [params["all_labels"][i] for i in labels_idx]
        df["score"] = scores

        # Log prediction distribution
        unique, counts = np.unique(labels_idx, return_counts=True)
        pred_dist = {params["all_labels"][idx]: count for idx, count in zip(unique, counts)}
        logger.info(f"Task {task_id}: Prediction distribution: {pred_dist}")

        logger.debug(f"Task {task_id}: Prediction completed, grouping into segments")
        segments = self._group_rows(df, time_col)
        logger.info(f"Task {task_id}: Found {len(segments)} segments")

        results = []
        avg_score = 0
        valid_segments = 0
        
        for seg in segments:
            # Skip background segments
            if seg["label"] == "__background__":
                logger.debug(f"Task {task_id}: Skipping background segment from {seg['start']} to {seg['end']}")
                continue
                
            score = float(np.mean(seg["scores"]))
            avg_score += score
            valid_segments += 1
            
            results.append(
                {
                    "from_name": params["from_name"],
                    "to_name": params["to_name"],
                    "type": "timeserieslabels",
                    "value": {
                        "start": seg["start"],
                        "end": seg["end"],
                        "instant": True if seg["start"] == seg["end"] else False,
                        "timeserieslabels": [seg["label"]],
                    },
                    "score": score,
                }
            )

        if not results:
            logger.warning(f"Task {task_id}: No prediction results generated (all background)")
            return {
                "result": [],
                "score": 0.0,
                "model_version": self.get("model_version"),
            }

        avg_score = avg_score / valid_segments if valid_segments > 0 else 0
        logger.info(f"Task {task_id}: Prediction completed with {valid_segments} segments, average score: {avg_score:.3f}")
        
        return {
            "result": results,
            "score": avg_score,
            "model_version": self.get("model_version"),
        }

    def _group_rows(self, df: pd.DataFrame, time_col: str) -> List[Dict]:
        logger.debug(f"Grouping {len(df)} rows into segments by consecutive labels")
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
        
        logger.debug(f"Grouped into {len(segments)} segments")
        return segments

    def _process_task_annotations(
        self, task: Dict, df: pd.DataFrame, params: Dict, label2idx: Dict[str, int], time_col: str
    ) -> Tuple[np.ndarray, int]:
        """Process annotations for a single task and return row labels.
        
        Args:
            task: Label Studio task dictionary
            df: DataFrame with time series data
            params: Labeling parameters from label config
            label2idx: Mapping from label names to indices
            
        Returns:
            Tuple of (row_labels array, number of labeled rows)
        """
        task_id = task.get("id", "unknown")
        
        # Initialize all rows as background (index 0)
        row_labels = np.zeros(len(df), dtype=np.int64)  # 0 = background
        
        annotations = [a for a in task["annotations"] if a.get("result")]
        logger.debug(f"Task {task_id}: Found {len(annotations)} annotations")
        
        # Mark labeled regions
        labeled_rows = 0
        for ann in annotations:
            for r in ann["result"]:
                if r["from_name"] != params["from_name"]:
                    continue
                start = r["value"]["start"]
                end = r["value"]["end"]
                label = r["value"]["timeserieslabels"][0]
                
                # Convert start/end to same type as time column for comparison
                time_dtype = df[time_col].dtype
                logger.debug(f"Task {task_id}: Converting time range [{start}, {end}] to match column dtype {time_dtype}")
                try:
                    if 'int' in str(time_dtype):
                        start = int(float(start))
                        end = int(float(end))
                    elif 'float' in str(time_dtype):
                        start = float(start)
                        end = float(end)
                    # For string/datetime, keep as is
                    logger.debug(f"Task {task_id}: Converted to [{start}, {end}]")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert start={start}, end={end} to {time_dtype}: {e}, using original values")
                
                # Find rows in this time range
                try:
                    mask = (df[time_col] >= start) & (
                        df[time_col] <= end
                    )
                except TypeError as e:
                    logger.error(f"Task {task_id}: Type error comparing times - start={start} ({type(start)}), end={end} ({type(end)}), time_col dtype={time_dtype}: {e}")
                    # Skip this annotation if we can't compare
                    continue
                
                # Set the appropriate label index
                label_idx = label2idx[label]
                row_labels[mask] = label_idx
                labeled_rows += mask.sum()
                logger.debug(f"Task {task_id}: Labeled {mask.sum()} rows with '{label}' (index {label_idx})")

            if ann.get('ground_truth', False):
                logger.info(f"Task {task_id}: Ground truth annotation found: {ann['ground_truth']}")
                break
                
        return row_labels, labeled_rows

    def _collect_samples(
        self, tasks: List[Dict], params: Dict, label2idx: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Collecting training samples from {len(tasks)} tasks")
        X_list, y_list = [], []
        processed_tasks = 0
        total_samples = 0
        
        for task in tasks:
            task_id = task.get("id", "unknown")
            logger.debug(f"Processing task {task_id} for training data")
            
            df, time_col = self._read_csv(task, task["data"][params["value"]], params)
            if df.empty:
                logger.warning(f"Task {task_id}: Empty dataframe, skipping")
                continue
            
            # Process annotations for this task
            row_labels, labeled_rows = self._process_task_annotations(task, df, params, label2idx, time_col)
            
            # Add ALL rows to training data
            X_list.append(df[params["channels"]].values.astype(np.float32))
            y_list.append(row_labels)
            
            background_rows = len(df) - labeled_rows
            task_samples = len(df)
            
            if task_samples > 0:
                processed_tasks += 1
                total_samples += task_samples
                logger.debug(f"Task {task_id}: Collected {task_samples} samples ({labeled_rows} labeled, {background_rows} background)")
        
        if not X_list:
            logger.warning("No training data collected")
            return np.array([]), np.array([])
            
        # Concatenate all data
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        logger.info(f"Training data collection completed: {processed_tasks}/{len(tasks)} tasks processed, {total_samples} total samples")
        return X, y

    def _save_model(self, model: TimeSeriesLSTM, project_id: Optional[int] = None) -> None:
        # Use default project_id if not provided (for backward compatibility)
        if project_id is None:
            project_id = 0
            logger.warning("No project_id provided for model save, using default project_id=0")
            
        logger.info(f"Saving model for project {project_id} to {self.MODEL_DIR}")
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        model_path = os.path.join(self.MODEL_DIR, f"model_project_{project_id}.pt")
        model.save(model_path)
        logger.info(f"Model for project {project_id} saved successfully to {model_path}")

    def _get_project_id_from_context(self, tasks: List[Dict], context: Optional[Dict] = None) -> Optional[int]:
        """Extract project ID from tasks or context for model selection."""
        # Try to get project_id from context first
        if context and "project" in context:
            if isinstance(context["project"], dict) and "id" in context["project"]:
                project_id = context["project"]["id"]
                logger.debug(f"Found project_id {project_id} from context")
                return project_id
            elif isinstance(context["project"], (int, str)):
                project_id = int(context["project"])
                logger.debug(f"Found project_id {project_id} from context")
                return project_id
        
        # Try to get project_id from tasks (if they have project information)
        for task in tasks:
            if "project" in task:
                project_id = int(task["project"])
                logger.debug(f"Found project_id {project_id} from task")
                return project_id
        
        # Try to get from context directly if it's a string/int
        if context and isinstance(context.get("project"), (int, str)):
            project_id = int(context["project"])
            logger.debug(f"Found project_id {project_id} from context project field")
            return project_id
            
        logger.debug("No project_id found in tasks or context")
        return None

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        logger.info(f"Starting prediction for {len(tasks)} tasks")
        
        # Determine which project's model to use
        project_id = self._get_project_id_from_context(tasks, context)
        if project_id is not None:
            logger.info(f"Using model for project {project_id}")
        else:
            logger.info("No project_id found, using default model")
        
        params = self._get_labeling_params()
        model = self._get_model(len(params["channels"]), len(params["all_labels"]), project_id=project_id)
        
        predictions = []
        for i, task in enumerate(tasks):
            logger.debug(f"Processing prediction {i+1}/{len(tasks)}")
            pred = self._predict_task(task, model, params)
            predictions.append(pred)
            
        successful_predictions = len([p for p in predictions if p])
        logger.info(f"Prediction completed: {successful_predictions}/{len(tasks)} tasks had results")
        
        return ModelResponse(
            predictions=predictions, model_version=self.get("model_version")
        )

    def _get_tasks(self, project_id: int) -> List[Dict]:
        logger.info(f"Fetching labeled tasks from project {project_id}")
        ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project_id)
        tasks = project.get_labeled_tasks()
        logger.info(f"Retrieved {len(tasks)} labeled tasks from project {project_id}")
        return tasks

    def fit(self, event, data, **kwargs):
        logger.info(f"Training event received: {event}")
        
        if event not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED", "START_TRAINING"):
            logger.info("Skip training: event %s is not supported", event)
            return
            
        project_id = data["annotation"]["project"]
        logger.info(f"Training triggered for project {project_id}")
        
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
            
        logger.info("Training conditions met, starting model training")
        params = self._get_labeling_params()
        # Create label mapping with background as index 0
        label2idx = {l: i for i, l in enumerate(params["all_labels"])}
        logger.debug(f"Label mapping: {label2idx}")

        X, y = self._collect_samples(tasks, params, label2idx)
        if len(X) == 0:
            logger.warning("No data collected for training")
            return

        # Log training label distribution
        unique, counts = np.unique(y, return_counts=True)
        train_dist = {params["all_labels"][idx]: count for idx, count in zip(unique, counts)}
        logger.info(f"Training label distribution: {train_dist}")

        logger.info(f"Training model with {len(X)} samples")
        model = self._get_model(
            len(params["channels"]), len(params["all_labels"]), project_id=project_id, blank=True
        )
        
        # Set label mapping for the model
        model.set_label_map(label2idx)
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Starting model training ({self.TRAIN_EPOCHS} epochs)")
        
        # Train the model with improved parameters for imbalanced data
        metrics = model.partial_fit(
            sequence=X,
            labels=y,
            epochs=self.TRAIN_EPOCHS,
            batch_size=16,
            balanced_accuracy_threshold=self.BALANCED_ACCURACY_THRESHOLD,
            min_class_f1_threshold=self.MIN_CLASS_F1_THRESHOLD,
            use_class_weights=self.USE_CLASS_WEIGHTS
        )
        
        logger.info(f"Model training completed with metrics: {metrics}")
        
        # Save the model with error handling
        try:
            self._save_model(model, project_id=project_id)
        except Exception as e:
            logger.error(f"Failed to save model for project {project_id}: {e}")
            # Continue anyway, the model is still in memory
        
        # Reset project-specific model in cache to force reload
        global _models
        if project_id in _models:
            del _models[project_id]
            logger.info(f"Model cache cleared for project {project_id}")

        # Load the newly trained model with fallback
        try:
            self._get_model(len(params["channels"]), len(params["all_labels"]), project_id=project_id)
            logger.info(f"Training process completed successfully for project {project_id}")
        except Exception as e:
            logger.warning(f"Failed to reload model after training for project {project_id}: {e}")
            # Set the trained model directly to project cache
            _models[project_id] = model
            logger.info(f"Using trained model directly from memory for project {project_id}")
        
        return metrics