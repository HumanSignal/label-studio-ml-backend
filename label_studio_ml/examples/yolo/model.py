import os
import cv2
import logging
import numpy as np
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_ml.utils import DATA_UNDEFINED_NAME

from control_models.base import ControlModel
from control_models.choices import ChoicesModel
from control_models.rectangle_labels import RectangleLabelsModel


logger = logging.getLogger(__name__)
if not os.getenv('LOG_LEVEL'):
    logger.setLevel(logging.INFO)

# Register available model classes
available_model_classes = [
    ChoicesModel,
    RectangleLabelsModel,
]


class YOLO(LabelStudioMLBase):
    """ Label Studio ML Backend based on Ultralytics YOLO
    """

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "yolo")

    def detect_control_models(self) -> List[ControlModel]:
        control_models = []

        for control in self.label_interface.controls:
            # skipping tags without toName
            if not control.to_name:
                logger.warning(f'{control.tag} {control.name} has no "toName" attribute, skipping it')
                continue

            # match control tag with available control models
            for model_class in available_model_classes:
                if model_class.is_control_matched(control):
                    instance = model_class.create(self, control)
                    if not instance.label_map:
                        logger.error(
                            f"No label map built for the '{control.tag}' control tag '{instance.from_name}'.\n"
                            f"This indicates that your Label Studio config labels do not match the model's labels.\n"
                            f"To fix this, ensure that the 'predicted_values' or 'value' attribute "
                            f"in your Label Studio config matches one or more of these model labels.\n"
                            f'Examples:\n<Label value="Car"/>\n'
                            f'<Label value="YourLabel" predicted_values="label1,label2"/>'
                            f"Available '{instance.model_name}' model labels: {list(instance.model.names.values())}"
                        )
                        continue

                    control_models.append(instance)
                    logger.debug(f"Control tag with model detected: {instance}")
                    break

        if not control_models:
            control_tags = ", ".join([c.type for c in available_model_classes])
            raise ValueError(
                f'No suitable control tags (e.g. {control_tags} connected to Image or Video object tags) '
                f'detected in the label config'
            )

        return control_models

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Run YOLO predictions on the tasks
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions [Predictions array in JSON format]
                (https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(f'Run prediction on {len(tasks)} tasks, project ID = {self.project_id}')
        control_models = self.detect_control_models()

        predictions = []
        for task in tasks:

            regions = []
            for model in control_models:
                task_path = task["data"].get(model.value) or task["data"].get(DATA_UNDEFINED_NAME)
                path = get_local_path(task_path, task_id=task.get('id'))
                logger.debug(f'load_image: {task_path} => {path}')

                regions += model.predict_regions(path)

            # calculate final score
            all_scores = [region['score'] for region in regions if 'score' in region]
            avg_score = sum(all_scores) / max(len(all_scores), 1)

            # compose final prediction
            prediction = {
                'result': regions,
                'score': avg_score,
                'model_version': self.model_version
            }
            predictions.append(prediction)

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event
        (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() is not implemented!')

    def temp(self):

        model = YOLO("yolov8n.pt")
        video_path = "path/to/video.mp4"
        cap = cv2.VideoCapture(video_path)
        track_history = defaultdict(lambda: [])

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = model.track(frame, persist=True)
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                annotated_frame = results[0].plot()
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

            {
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