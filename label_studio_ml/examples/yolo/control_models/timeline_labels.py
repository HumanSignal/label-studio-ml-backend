import logging
import os.path

from control_models.base import ControlModel, MODEL_ROOT
from typing import List, Dict
from utils.neural_nets import BaseNN, MultiLabelLSTM, cached_feature_extraction
from utils.converter import (
    get_label_map,
    convert_timelinelabels_to_probs,
    convert_probs_to_timelinelabels,
)


logger = logging.getLogger(__name__)


class TimelineLabelsModel(ControlModel):
    """
    Class representing a TimelineLabels control tag for YOLO model.
    """

    type = "TimelineLabels"
    model_path = "yolov8n-cls.pt"

    @classmethod
    def is_control_matched(cls, control) -> bool:
        # Check object tag type
        if control.objects[0].tag != "Video":
            return False
        # Support TimelineLabels
        return control.tag == cls.type

    @classmethod
    def create(cls, *args, **kwargs):
        instance = super().create(*args, **kwargs)
        # TODO: check if model_trainable=true, then run this line to avoid skipping of this control model
        instance.label_map = {label: label for label in instance.control.labels}
        return instance

    def predict_regions(self, video_path) -> List[Dict]:
        frame_results = cached_feature_extraction(
            self.model, video_path, self.model.model_name
        )
        return self.create_timelines_active_learning(frame_results, video_path)

    def create_timelines_simple(self, frame_results, video_path):
        logger.debug(f"create_timelines_simple: {self.from_name}")

        # Initialize a dictionary to keep track of ongoing segments for each label
        model_names = self.model.names
        needed_ids = [i for i, name in model_names.items() if name in self.label_map]
        needed_labels = [
            name for i, name in model_names.items() if name in self.label_map
        ]

        probs = [frame.probs[needed_ids] for frame in frame_results]
        label_map = {
            self.label_map[label]: idx for idx, label in enumerate(needed_labels)
        }

        return convert_probs_to_timelinelabels(
            probs, label_map, self.model_score_threshold
        )

    def create_timelines_active_learning(self, frame_results, video_path):
        logger.debug(f"create_timelines_active_learning: {self.from_name}")

        yolo_probs = [frame.probs for frame in frame_results]
        path = self.get_classifier_path(self.project_id)
        classifier = BaseNN.load_cached_model(path)
        if not classifier:
            raise ValueError(
                f"Classifier model '{path}' not found for {self.control}, maybe it's not trained yet"
            )

        # run predict and convert to timelinelabels
        probs = classifier.predict(yolo_probs)
        regions = convert_probs_to_timelinelabels(
            probs, classifier.get_label_map(), self.model_score_threshold
        )

        return regions

    def fit(self, event, data, **kwargs):
        """Fit the model."""
        if event == "START_TRAINING":
            logger.warning(
                f"The event START_TRAINING is not supported for this control model: {self.control.tag}"
            )
            return False

        if event in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
            # Get the task and regions
            task = data["task"]
            project_id = task["project"]
            regions = data["annotation"]["result"]

            get = self.control.attr.get
            # Maximum number of training epochs
            epochs = int(get("model_classifier_epochs", 1000))
            # LSTM sequence size
            sequence_size = int(get("model_classifier_sequence_size", 64))
            # LSTM hidden state size
            hidden_size = int(get("model_classifier_hidden_size", 32))
            # LSTM num layers
            num_layers = int(get("model_classifier_num_layers", 1))
            # Stop training when accuracy reaches this threshold, it helps to avoid overfitting
            # because we partially train it on a small dataset from one annotation only
            f1_score_threshold = float(get("model_classifier_f1_score_threshold", 0.95))

            # Get the features and labels for training
            video_path = self.get_path(task)
            frames = cached_feature_extraction(
                self.model, video_path, self.model.model_name
            )
            features = [frame.probs for frame in frames]
            label_map = get_label_map(self.control.labels)
            labels, used_labels = convert_timelinelabels_to_probs(
                regions, label_map=label_map, max_frame=len(frames)
            )
            # check if all labels from used_labels are in the label_map
            if not used_labels.issubset(label_map.keys()):
                logger.warning(f"Annotation labels ({used_labels}) are not subset "
                               f"of labels from the labeling config: {self.control}")
                return False

            # Load classifier
            path = self.get_classifier_path(project_id)
            classifier = BaseNN.load_cached_model(path)

            # Create a new classifier instance if it doesn't exist
            # or if labeling config has changed
            if (
                not classifier
                or classifier.label_map != label_map
                or classifier.sequence_size != sequence_size
                or classifier.hidden_size != hidden_size
                or classifier.num_layers != num_layers
            ):
                logger.info("Creating a new classifier model for timelinelabels")
                input_size = len(features[0])
                output_size = len(label_map)
                classifier = MultiLabelLSTM(
                    input_size,
                    output_size,
                    sequence_size=sequence_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                )
                classifier.set_label_map(label_map)

            # Train and save
            classifier.partial_fit(
                features, labels, epochs=epochs, f1_score_threshold=f1_score_threshold
            )
            classifier.save(path)
            return True

    def get_classifier_path(self, project_id):
        yolo_base_name = os.path.splitext(os.path.basename(self.model.model_name))[0]
        path = f"{MODEL_ROOT}/timelinelabels-{project_id}-{yolo_base_name}-{self.from_name}.pkl"
        return path


# Preload and cache the default yolo model at startup
TimelineLabelsModel.get_cached_model(TimelineLabelsModel.model_path)
