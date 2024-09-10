import logging
import os.path

from control_models.base import ControlModel, MODEL_ROOT
from typing import List, Dict, ClassVar
from joblib import Memory
from utils.neural_nets import MultiLabelNN
from utils.converter import convert_timelinelabels_to_probs, convert_probs_to_timelinelabels


logger = logging.getLogger(__name__)
memory = Memory("./cache_dir", verbose=0)  # Set up disk-based caching for model results


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
        frame_results[i].probs = last_layer_output_per_frame[i][0]
        frame_results[i].orig_img = None

    # Remove the hook
    hook_handle.remove()

    return frame_results


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

    def predict_regions(self, video_path) -> List[Dict]:
        frame_results = cached_model_predict(self, video_path, self.model.model_name)
        return self.create_timelines(frame_results, video_path)

    def create_timelines(self, frame_results, video_path):
        logger.debug(f'create_timelinelabels: {self.from_name}')

        # Initialize a dictionary to keep track of ongoing segments for each label
        timeline_regions = []
        model_names = self.model.names
        ongoing_segments = {label: {} for label in self.label_map}
        needed_ids = [i for i, name in model_names.items() if name in self.label_map]
        needed_labels = [name for i, name in model_names.items() if name in self.label_map]

        for i, frame_result in enumerate(frame_results):
            # Get only needed probabilities for label config labels for the current frame
            probs = frame_result.probs[needed_ids].data

            for index, prob in enumerate(probs):
                name = needed_labels[index]

                # Only process labels that are in `self.label_map`
                if name in self.label_map:
                    segment = ongoing_segments[name]
                    if prob >= self.model_score_threshold:
                        # Start a segment for this label
                        if not segment:
                            segment["start"] = i
                    else:
                        # If a segment was ongoing, close it
                        if segment:
                            self.add_timeline_region(i, self.label_map[name], segment, timeline_regions)
                            # Reset the segment for this label
                            segment.clear()

        # Close any ongoing segments at the end of the video
        for name, segment in ongoing_segments.items():
            if segment:
                self.add_timeline_region(len(frame_results), self.label_map[name], segment, timeline_regions)

        return timeline_regions

    def add_timeline_region(self, i, label, segment, timeline_labels):
        timeline_labels.append({
            "id": f"{segment['start']}_{i}",
            "type": "timelinelabels",
            "value": {
                "ranges": [{"start": segment['start'], "end": i}],
                "timelinelabels": [label]
            },
            "origin": "manual",
            "to_name": self.to_name,
            "from_name": self.from_name
        })
        return timeline_labels

    def fit(self, event, data, **kwargs):
        """Fit the model."""
        if event == 'START_TRAINING':
            logger.warning(f"The event START_TRAINING is not supported for this control model: {self.control.tag}")
            return False 

        if event in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED'):
            project_id = data['task']['project']
            task = data['task']
            regions = data['annotation']['result']
            
            video_path = self.get_path(task)
            frame_results = cached_model_predict(self, video_path, self.model.model_name)
            features = [frame.probs for frame in frame_results]
            labels, mapping = convert_timelinelabels_to_probs(regions, max_frame=len(frame_results))
            if not label_mapping:
                logger.warning(f"No labels found in regions for timelinelabels: {self.control}")
                return False

            classifier = MultiLabelNN(input_size=len(features[0]), output_size=len(labels[0]))
            classifier.partial_fit(features, labels, epochs=3)
            path = self.get_timeline_labels_model_path(project_id)

            classifier.save(path)
            return True

    def get_timeline_labels_model_path(self, project_id):
        yolo_base_name = os.path.splitext(os.path.basename(self.model.model_name))[0]
        path = f"{MODEL_ROOT}/timelinelabels-{project_id}-{yolo_base_name}-{self.from_name}.pkl"
        return path


# Preload and cache the default model at startup
TimelineLabelsModel.get_cached_model(TimelineLabelsModel.model_path)
