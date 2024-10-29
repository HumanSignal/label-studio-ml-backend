import logging
import os.path

from control_models.base import ControlModel, MODEL_ROOT, get_bool
from typing import List, Dict
from utils.neural_nets import (
    BaseNN,
    MultiLabelLSTM,
    cached_feature_extraction,
    cached_yolo_predict,
)
from utils.converter import (
    get_label_map,
    convert_timelinelabels_to_probs,
    convert_probs_to_timelinelabels,
)


logger = logging.getLogger(__name__)


class TimelineLabelsModel(ControlModel):
    """
    Class representing a TimelineLabels control tag for YOLO model.
    See README_TIMELINE_LABELS.md for more details.
    """

    type = "TimelineLabels"
    model_path = "yolov8n-cls.pt"
    trainable: bool = False

    @classmethod
    def is_control_matched(cls, control) -> bool:
        # Check object tag type
        if control.objects[0].tag != "Video":
            return False
        return control.tag == cls.type

    @classmethod
    def create(cls, *args, **kwargs):
        instance = super().create(*args, **kwargs)

        # timeline models can be trainable and based on YOLO trained classes directly
        instance.trainable = get_bool(instance.control.attr, "model_trainable", "false")
        # if it's trainable, we need to use labels from the labeling config as is because we will train them
        if instance.trainable:
            instance.label_map = {label: label for label in instance.control.labels}
        elif not instance.label_map:
            raise ValueError(
                f"TimelinesLabels model works in simple mode (without training), "
                f"but no labels from YOLO model names are matched:\n{instance.control.name}\n"
                f"Add labels from YOLO model names to the labeling config or use `predicted_values` to map them. "
                f'As alternative option, you can set `model_trainable="true"` in the TimelineLabels control tag '
                f"to train the model on the labels from the labeling config."
            )
        return instance

    def predict_regions(self, video_path) -> List[Dict]:
        if self.trainable:
            return self.create_timelines_trainable(video_path)
        else:
            return self.create_timelines_simple(video_path)

    def create_timelines_simple(self, video_path):
        logger.debug(f"create_timelines_simple: {self.from_name}")
        # get yolo predictions
        frame_results = cached_yolo_predict(
            self.model, video_path, self.model.model_name
        )

        # Initialize a dictionary to keep track of ongoing segments for each label
        model_names = self.model.names
        needed_ids = [i for i, name in model_names.items() if name in self.label_map]
        needed_labels = [
            name for i, name in model_names.items() if name in self.label_map
        ]

        probs = [frame.probs.data[needed_ids].cpu().numpy() for frame in frame_results]
        label_map = {
            self.label_map[label]: idx for idx, label in enumerate(needed_labels)
        }

        return convert_probs_to_timelinelabels(
            probs, label_map, self.control.name, self.model_score_threshold
        )

    def create_timelines_trainable(self, video_path):
        logger.debug(f"create_timelines_trainable: {self.from_name}")
        # extract features based on pre-trained yolo classification model
        frame_results = cached_feature_extraction(
            self.model, video_path, self.model.model_name
        )

        yolo_probs = [frame.probs for frame in frame_results]
        path = self.get_classifier_path(self.project_id)
        classifier = BaseNN.load_cached_model(path)
        if not classifier:
            raise ValueError(
                f"Temporal classifier model '{path}' not found for "
                f"'{self.control.name}', maybe it's not trained yet"
            )

        # run predict and convert to timelinelabels
        probs = classifier.predict(yolo_probs)
        regions = convert_probs_to_timelinelabels(
            probs,
            classifier.get_label_map(),
            self.control.name,
            self.model_score_threshold,
        )

        return regions

    def fit(self, event, data, **kwargs):
        if not self.trainable:
            logger.debug(
                'TimelineLabels model is in not trainable mode. '
                'Use model_trainable="true" to enable training.'
            )
            return

        """Fit the model."""
        if event == "START_TRAINING":
            # TODO: the full training makes a lot of sense here, but it's not implemented yet
            raise NotImplementedError(
                f"The event START_TRAINING is not supported for this control model: {self.control.tag}"
            )

        if event in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
            features, labels, label_map, project_id = self.load_features_and_labels(
                data
            )
            classifier, path = self.load_classifier(features, label_map, project_id)
            return self.train_classifier(classifier, features, labels, path)

    def train_classifier(self, classifier, features, labels, path):
        """Train the classifier model for timelinelabels using incremental partial learning."""
        # Stop training when accuracy or f1 score reaches this threshold, it helps to avoid overfitting
        # because we partially train it on a small dataset from one annotation only
        get = self.control.attr.get
        epochs = int(
            get("model_classifier_epochs", 1000)
        )  # Maximum number of training epochs
        f1_threshold = float(get("model_classifier_f1_threshold", 0.95))
        accuracy_threshold = float(get("model_classifier_accuracy_threshold", 1.00))

        # Train and save
        result = classifier.partial_fit(
            features,
            labels,
            epochs=epochs,
            f1_threshold=f1_threshold,
            accuracy_threshold=accuracy_threshold,
        )
        classifier.save_and_cache(path)
        return result

    def load_classifier(self, features, label_map, project_id):
        """Load or create a classifier model for timelinelabels.
        1. Load neural network parameters from labeling config.
        2. Try loading classifier model from memory cache, then from disk.
        3. Or create a new classifier instance if there wasn't successful loading, or if parameters have changed.
        """
        get = self.control.attr.get
        # LSTM sequence size
        sequence_size = int(get("model_classifier_sequence_size", 16))
        # LSTM hidden state size
        hidden_size = int(get("model_classifier_hidden_size", 32))
        # LSTM num layers
        num_layers = int(get("model_classifier_num_layers", 1))

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

        return classifier, path

    def load_features_and_labels(self, data):
        """Load features and labels from the annotation
        Args:
            data: event data, dictionary with keys 'task' and 'annotation'
        Returns:
            features: List of features, 2D array with shape (num_frames, num_features)
            labels: List of labels, 2D array with shape (num_frames, num_labels)
            label_map: Label map, dictionary mapping label names to indices in the labels array
            project_id: Project ID from Label Studio
        """
        # Get the task and regions from the annotation
        task = data["task"]
        project_id = task["project"]
        annotation = data["annotation"]
        regions = annotation["result"]

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

        # Check if all labels from used_labels are in the label_map
        if not used_labels.issubset(label_map.keys()):
            raise ValueError(
                f"Annotation labels set ({used_labels}) is not subset "
                f"of labels from the labeling config:\n{self.control}\n"
                f"It can be caused by the mismatch between the labeling config "
                f"and labels in the annotation #{data['annotation']['id']}"
                f"of project #{project_id}."
            )
        return features, labels, label_map, project_id

    def get_classifier_path(self, project_id):
        yolo_base_name = os.path.splitext(os.path.basename(self.model.model_name))[0]
        path = f"{MODEL_ROOT}/timelinelabels-{project_id}-{yolo_base_name}-{self.from_name}.pkl"
        return path


# Preload and cache the default yolo model at startup
TimelineLabelsModel.get_cached_model(TimelineLabelsModel.model_path)
