import os
import cv2
import logging
import yaml
import hashlib

from collections import defaultdict
from control_models.base import ControlModel, MODEL_ROOT
from label_studio_sdk.label_interface.control_tags import ControlTag
from typing import List, Dict, Union


logger = logging.getLogger(__name__)


class VideoRectangleModel(ControlModel):
    """
    Class representing a RectangleLabels (bounding boxes) control tag for YOLO model.
    """

    type = "VideoRectangle"
    model_path = "yolov8n.pt"

    @classmethod
    def is_control_matched(cls, control: ControlTag) -> bool:
        # check object tag type
        if control.objects[0].tag != "Video":
            return False
        # check control type VideoRectangle
        return control.tag == cls.type

    @staticmethod
    def get_from_name_for_label_map(label_interface, target_name) -> str:
        """VideoRectangle doesn't have labels inside, and we should find a connected Labels tag
        and return its name as a source for the label map.
        """
        target: ControlTag = label_interface.get_control(target_name)
        if not target:
            raise ValueError(f'Control tag with name "{target_name}" not found')

        for connected in label_interface.controls:
            if connected.tag == "Labels" and connected.to_name == target.to_name:
                return connected.name

        logger.error("VideoRectangle detected, but no connected 'Labels' tag found")

    @staticmethod
    def get_video_duration(path):
        if not os.path.exists(path):
            raise ValueError(f"Video file not found: {path}")
        video = cv2.VideoCapture(path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        logger.info(
            f"Video duration: {duration} seconds, {frame_count} frames, {fps} fps"
        )
        return frame_count, duration

    def predict_regions(self, path) -> List[Dict]:
        # bounding box parameters
        # https://docs.ultralytics.com/modes/track/?h=track#tracking-arguments
        conf = float(self.control.attr.get("model_conf", 0.25))
        iou = float(self.control.attr.get("model_iou", 0.70))

        # tracking parameters
        # https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers
        tracker_name = self.control.attr.get(
            "model_tracker", "botsort"
        )  # or 'bytetrack'
        original = f"{MODEL_ROOT}/{tracker_name}.yaml"
        tmp_yaml = self.update_tracker_params(original, prefix=tracker_name + "_")
        tracker = tmp_yaml if tmp_yaml else original

        # run model track
        try:
            results = self.model.track(
                path, conf=conf, iou=iou, tracker=tracker, stream=True
            )
        finally:
            # clean temporary file
            if tmp_yaml and os.path.exists(tmp_yaml):
                os.remove(tmp_yaml)

        # convert model results to label studio regions
        return self.create_video_rectangles(results, path)

    def create_video_rectangles(self, results, path):
        """Create regions of video rectangles from the yolo tracker results"""
        frames_count, duration = self.get_video_duration(path)
        model_names = self.model.names
        logger.debug(
            f"create_video_rectangles: {self.from_name}, {frames_count} frames"
        )

        tracks = defaultdict(list)
        track_labels = dict()
        frame = -1
        for result in results:
            frame += 1
            data = result.boxes
            if not data.is_track:
                continue

            for i, track_id in enumerate(data.id.tolist()):
                score = float(data.conf[i])
                x, y, w, h = data.xywhn[i].tolist()
                # get label
                model_label = model_names[int(data.cls[i])]
                if model_label not in self.label_map:
                    continue
                output_label = self.label_map[model_label]
                track_labels[track_id] = output_label

                box = {
                    "frame": frame + 1,
                    "enabled": True,
                    "rotation": 0,
                    "x": (x - w / 2) * 100,
                    "y": (y - h / 2) * 100,
                    "width": w * 100,
                    "height": h * 100,
                    "time": (frame + 1) * (duration / frames_count),
                    "score": score,
                }
                tracks[track_id].append(box)

        regions = []
        for track_id in tracks:
            sequence = tracks[track_id]
            sequence = self.process_lifespans_enabled(sequence)

            label = track_labels[track_id]
            region = {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "videorectangle",
                "value": {
                    "framesCount": frames_count,
                    "duration": duration,
                    "sequence": sequence,
                    "labels": [label],
                },
                "score": max([frame_info["score"] for frame_info in sequence]),
                "origin": "manual",
            }
            regions.append(region)

        return regions

    @staticmethod
    def process_lifespans_enabled(sequence: List[Dict]) -> List[Dict]:
        """This function detects gaps in the sequence of bboxes
        and disables lifespan line for the gaps assigning "enabled": False
        to the last bboxes in the whole span sequence.
        """
        prev = None
        for i, box in enumerate(sequence):
            if prev is None:
                prev = sequence[i]
                continue
            if box["frame"] - prev["frame"] > 1:
                sequence[i - 1]["enabled"] = False
            prev = sequence[i]

        # the last frame enabled is false to turn off lifespan line
        sequence[-1]["enabled"] = False
        return sequence

    @staticmethod
    def generate_hash_filename(extension=".yaml"):
        """Store yaml configs as temporary files just for one model.track() run"""
        hash_name = hashlib.sha256(os.urandom(16)).hexdigest()
        os.makedirs(f"{MODEL_ROOT}/tmp/", exist_ok=True)
        return f"{MODEL_ROOT}/tmp/{hash_name}{extension}"

    def update_tracker_params(self, yaml_path: str, prefix: str) -> Union[str, None]:
        """Update tracker parameters in the yaml file with the attributes from the ControlTag,
        e.g. <VideoRectangle model_tracker="bytetrack" bytetrack_max_age="10" bytetrack_min_hits="3" />
        or <VideoRectangle model_tracker="botsort" botsort_max_age="10" botsort_min_hits="3" />
        Args:
            yaml_path: Path to the original yaml file.
            prefix: Prefix for attributes of control tag to extract
        Returns:
            The file path for new yaml with updated parameters
        """
        # check if there are any custom parameters in the labeling config
        for attr_name, attr_value in self.control.attr.items():
            if attr_name.startswith(prefix):
                break
        else:
            # no custom parameters, exit
            return None

        # Load the original yaml file
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)

        # Extract parameters with prefix from ControlTag
        for attr_name, attr_value in self.control.attr.items():
            if attr_name.startswith(prefix):
                # Remove prefix and update the corresponding yaml key
                key = attr_name[len(prefix) :]

                # Convert value to the appropriate type (bool, int, float, etc.)
                if isinstance(config[key], bool):
                    attr_value = attr_value.lower() == "true"
                elif isinstance(config[key], int):
                    attr_value = int(attr_value)
                elif isinstance(config[key], float):
                    attr_value = float(attr_value)

                config[key] = attr_value

        # Generate a new filename with a random hash
        new_yaml_filename = self.generate_hash_filename()

        # Save the updated config to a new yaml file
        with open(new_yaml_filename, "w") as file:
            yaml.dump(config, file)

        # Return the new filename
        return new_yaml_filename


# pre-load and cache default model at startup
VideoRectangleModel.get_cached_model(VideoRectangleModel.model_path)
