import gc
import logging
import os.path

from control_models.base import ControlModel, MODEL_ROOT
from typing import List, Dict, ClassVar, Union
from joblib import Memory
from utils.neural_nets import MultiLabelNN, MultiLabelLSTM
from utils.converter import convert_timelinelabels_to_probs, convert_probs_to_timelinelabels


logger = logging.getLogger(__name__)
memory = Memory("./cache_dir", verbose=1)  # Set up disk-based caching for model results


@memory.cache(ignore=["self"])
def cached_model_predict(self, video_path, cache_params):
    last_layer_output_per_frame = []  # Define and register the hook for yolo model to get last layer

    def get_last_layer_output(module, input, output):
        last_layer_output_per_frame.append(output)

    # Register the hook on the last layer of the model
    last_layer = self.model.model.model[-1]  # Adjust depending on your model structure
    hook_handle = last_layer.register_forward_hook(get_last_layer_output)

    # Run model prediction
    frame_results = self.model.predict(video_path)
    
    # Replace probs with last layer outputs
    for i in range(len(frame_results)):
        frame_results[i].probs = last_layer_output_per_frame[i][0] # => tensor
        # frame_results[i].probs = frame_results[i].probs.data  # convert to tensor
        frame_results[i].orig_img = None

    # Remove the hook
    hook_handle.remove()
    gc.collect()
    return frame_results


_classifiers = {}


def get_classifier(model_path: str) -> Union[MultiLabelNN, None]:
    global _classifiers

    if not os.path.exists(model_path):
        return None

    if model_path not in _classifiers:
        _classifiers[model_path] = MultiLabelNN.load(model_path)

    return _classifiers[model_path]


class TimelineLabelsModel(ControlModel):
    """
    Class representing a TimelineLabels control tag for YOLO model.
    """
    class Config:
        arbitrary_types_allowed = True

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
        frame_results = cached_model_predict(self, video_path, self.model.model_name)
        return self.create_timelines_active_learning(frame_results, video_path)

    def create_timelines_simple(self, frame_results, video_path):
        logger.debug(f'create_timelines_simple: {self.from_name}')

        # Initialize a dictionary to keep track of ongoing segments for each label
        model_names = self.model.names
        needed_ids = [i for i, name in model_names.items() if name in self.label_map]
        needed_labels = [name for i, name in model_names.items() if name in self.label_map]

        probs = [frame.probs[needed_ids] for frame in frame_results]
        label_map = {self.label_map[label]: idx for idx, label in enumerate(needed_labels)}

        return convert_probs_to_timelinelabels(probs, label_map, self.model_score_threshold)

    def create_timelines_active_learning(self, frame_results, video_path):
        logger.debug(f'create_timelines_active_learning: {self.from_name}')

        yolo_probs = [frame.probs for frame in frame_results]
        path = self.get_classifier_path(self.project_id)
        classifier = get_classifier(path)
        if not classifier:
            logger.warning(f"Classifier model '{path}' not found for {self.control}, maybe it's not trained yet")
            return []

        # run predict and convert to timelinelabels
        probs = classifier.predict(yolo_probs)
        regions = convert_probs_to_timelinelabels(probs, classifier.get_label_map(), self.model_score_threshold)

        return regions

    def fit(self, event, data, **kwargs):
        """Fit the model."""
        if event == 'START_TRAINING':
            logger.warning(f"The event START_TRAINING is not supported for this control model: {self.control.tag}")
            return False 

        if event in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED'):
            # Get the task and regions
            project_id = data['task']['project']
            task = data['task']
            regions = data['annotation']['result']
            epochs = int(self.control.attr.get('model_epochs', 50))
            classifier_type = self.control.attr.get('model_classifier_type', 'MultiLabelLSTM')
            assert classifier_type in ['MultiLabelNN', 'MultiLabelLSTM']
            sequence_size = int(self.control.attr.get('model_sequence_size', 32))

            # Get the features and labels for training
            video_path = self.get_path(task)
            frame_results = cached_model_predict(self, video_path, self.model.model_name)
            features = [frame.probs for frame in frame_results]
            labels, label_map = convert_timelinelabels_to_probs(regions, max_frame=len(frame_results))
            if not features:
                logger.warning(f"No features or labels found for timelinelabels: {self.control}")
                return False
            if not label_map:
                logger.warning(f"No labels found in regions for timelinelabels: {self.control}")
                return False

            # Load classifier
            path = self.get_classifier_path(project_id)
            classifier = get_classifier(path)

            # Create a new classifier instance if it doesn't exist
            # or if labeling config has changed
            if (
                not classifier
                or classifier.label_map != label_map
                or classifier.__class__.__name__ != classifier_type
            ):
                input_size = len(features[0])
                output_size = len(label_map)
                classifier = (
                    MultiLabelNN(input_size, output_size)
                    if classifier_type == 'MultiLabelNN' else
                    MultiLabelLSTM(input_size, output_size, sequence_size=sequence_size)
                )
                classifier.set_label_map(label_map)

            # Train and save
            classifier.partial_fit(features, labels, epochs=epochs)
            classifier.save(path)
            return True

    def get_classifier_path(self, project_id):
        yolo_base_name = os.path.splitext(os.path.basename(self.model.model_name))[0]
        path = f"{MODEL_ROOT}/timelinelabels-{project_id}-{yolo_base_name}-{self.from_name}.pkl"
        return path


# Preload and cache the default model at startup
TimelineLabelsModel.get_cached_model(TimelineLabelsModel.model_path)
