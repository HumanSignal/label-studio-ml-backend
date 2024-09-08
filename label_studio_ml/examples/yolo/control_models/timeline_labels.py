import logging
from control_models.base import ControlModel
from typing import List, Dict

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

    def predict_regions(self, video_path) -> List[Dict]:
        # Assuming that `self.model.predict` can take a video file path and output predictions for each frame
        frame_results = self.model.predict(video_path)
        self.debug_plot(frame_results[0].plot())

        return self.create_timelines(frame_results, video_path)

    def create_timelines(self, frame_results, video_path):
        logger.debug(f"create_timelines: {self.from_name}")

        # Initialize dictionary to keep track of ongoing segments for each label
        ongoing_segments = {label: None for label in self.model.names}
        timeline_labels = []

        for i, frame_result in enumerate(frame_results):
            # Get probabilities for all labels for the current frame
            probs = frame_result.probs.numpy().data

            for label_index, prob in enumerate(probs):
                label = self.model.names[label_index]

                if prob >= self.model_score_threshold:
                    # Start or continue a segment for this label
                    if ongoing_segments[label] is None:
                        # Start a new segment
                        ongoing_segments[label] = {"start": i}
                else:
                    # If a segment was ongoing, close it
                    if ongoing_segments[label] is not None:
                        timeline_labels.append(
                            {
                                "id": f"{video_path}_{label}_{ongoing_segments[label]['start']}_{i}",
                                "type": "timelinelabels",
                                "value": {
                                    "ranges": [
                                        {
                                            "start": ongoing_segments[label]["start"],
                                            "end": i,
                                        }
                                    ],
                                    "timelinelabels": [label],
                                },
                                "origin": "manual",
                                "to_name": self.to_name,
                                "from_name": self.from_name,
                            }
                        )
                        # Reset the segment for this label
                        ongoing_segments[label] = None

        # Close any ongoing segments at the end of the video
        for label, segment in ongoing_segments.items():
            if segment is not None:
                timeline_labels.append(
                    {
                        "id": f"{video_path}_{label}_{segment['start']}_{len(frame_results)}",
                        "type": "timelinelabels",
                        "value": {
                            "ranges": [
                                {"start": segment["start"], "end": len(frame_results)}
                            ],
                            "timelinelabels": [label],
                        },
                        "origin": "manual",
                        "to_name": self.to_name,
                        "from_name": self.from_name,
                    }
                )

        return timeline_labels


# Pre-load and cache the default model at startup
TimelineLabelsModel.get_cached_model(TimelineLabelsModel.model_path)
