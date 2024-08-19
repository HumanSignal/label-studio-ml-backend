import cv2
import logging

from collections import defaultdict
from control_models.base import ControlModel, MODEL_ROOT
from label_studio_sdk.label_interface.control_tags import ControlTag
from typing import List, Dict


logger = logging.getLogger(__name__)


class VideoRectangleModel(ControlModel):
    """
    Class representing a RectangleLabels (bounding boxes) control tag for YOLO model.
    """
    type = 'VideoRectangle'
    model_path = 'yolov8m.pt'

    @classmethod
    def is_control_matched(cls, control: ControlTag) -> bool:
        # check object tag type
        if control.objects[0].tag != 'Video':
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
            if connected.tag == 'Labels' and connected.to_name == target.to_name:
                return connected.name

        logger.error("VideoRectangle detected, but no connected 'Labels' tag found")

    @staticmethod
    def get_video_duration(path):
        video = cv2.VideoCapture(path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        return duration

    def predict_regions(self, path) -> List[Dict]:
        # https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers
        tracker_name = self.control.attr.get('tracker', 'botsort.yaml')  # 'bytetrack.yaml'
        tracker_params = {
            'tracker': tracker_name,
            'track_high_thresh': 0.01
        }
        results = self.model.track(path, conf=0.01, iou=0.5, tracker=MODEL_ROOT + '/' + tracker_name)
        return self.create_video_rectangles(results, path)

        x = {
            "result": [
                {
                    "value": {
                        "framesCount": 864,
                        "duration": 34.538667,
                        "sequence": [
                            {
                                "frame": 1,
                                "enabled": true,
                                "rotation": 0,
                                "x": 14.23913043478261,
                                "y": 30.434782608695656,
                                "width": 58.26086956521739,
                                "height": 34.78260869565217,
                                "time": 0.04
                            },
                            {
                                "x": 35.7608695652174,
                                "y": 18.2608695652174,
                                "width": 58.26086956521739,
                                "height": 34.78260869565217,
                                "rotation": 0,
                                "frame": 26,
                                "enabled": true,
                                "time": 1.04
                            },
                            {
                                "x": 18.15217391304348,
                                "y": 35.21739130434784,
                                "width": 58.26086956521739,
                                "height": 34.78260869565217,
                                "rotation": 0,
                                "frame": 65,
                                "enabled": true,
                                "time": 2.6
                            },
                            {
                                "x": 24.891304347826065,
                                "y": 42.82608695652174,
                                "width": 27.608695652173886,
                                "height": 22.826086956521756,
                                "rotation": 0,
                                "frame": 127,
                                "enabled": false,
                                "time": 5.08
                            },
                            {
                                "x": 24.891304347826065,
                                "y": 42.82608695652174,
                                "width": 27.608695652173886,
                                "height": 22.826086956521756,
                                "rotation": 0,
                                "enabled": true,
                                "frame": 143,
                                "time": 5.72
                            }
                        ],
                        "labels": [
                            "Man"
                        ]
                    },
                    "id": "7Ar8lQGrBx",
                    "from_name": "box",
                    "to_name": "video",
                    "type": "videorectangle",
                    "origin": "manual"
                }
            ]
        }

    def create_video_rectangles(self, results, path):
        frames_count, duration = len(results), self.get_video_duration(path)
        logger.debug(f'create_video_rectangles: {self.from_name}, {frames_count} frames')

        tracks = defaultdict(list)
        track_labels = dict()
        for frame, result in enumerate(results):
            data = result.boxes
            for i, track_id in enumerate(data.id):
                score = float(data.conf[i])
                x, y, w, h = data.xywhn[i].tolist()
                # get label
                model_label = self.model.names[int(data.cls[i])]
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
                    "score": score
                }
                tracks[track_id].append(box)

        regions = []
        for track_id in tracks:
            sequence = tracks[track_id]
            sequence[-1]['enabled'] = False  # the last frame enabled is false to turn off lifespan line
            label = track_labels[track_id]
            region = {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "videorectangle",
                "value": {
                    "framesCount": frames_count,
                    "duration": duration,
                    "sequence": sequence,
                    "labels": [label]
                },
                "score": max([frame_info['score'] for frame_info in sequence]),
                "origin": "manual"
            }
            regions.append(region)

        return regions

# pre-load and cache default model at startup
VideoRectangleModel.get_cached_model(VideoRectangleModel.model_path)
