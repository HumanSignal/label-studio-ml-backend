import os

import cv2
import matplotlib.pyplot as plt
import logging
import numpy as np

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralytics import YOLO

from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from control_model import ControlModel


# default threshold for confidences
SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', 0.5))
# use matplotlib plots for debug
DEBUG_PLOT = os.getenv('DEBUG_PLOT', "false").lower() in ["1", "true"]

logger = logging.getLogger(__name__)
if not os.getenv('LOG_LEVEL'):
    logger.setLevel(logging.INFO)


def load_models():
    # preload default models at the startup, comment out the models you don't need
    models = {
        'RectangleLabels': YOLO('yolov8m.pt'),  # https://docs.ultralytics.com/tasks/detect/
        'RectangleLabels.OBB': YOLO('yolov8n-obb.pt'),  # https://docs.ultralytics.com/tasks/obb/
        'Keypoints': YOLO('yolov8n-pose.pt'),  # https://docs.ultralytics.com/tasks/pose/
        'PolygonLabels': YOLO('yolov8n-seg.pt'),  # https://docs.ultralytics.com/tasks/segment/
        'Choices': YOLO('yolov8n-cls.pt'),  # https://docs.ultralytics.com/tasks/classify/
    }
    # Taxonomy is the same as Choices
    models['Taxonomy'] = models['Choices']

    # print model labels for user info
    for control, model in models.items():
        logger.info(f'Available "{model.model_name}" model labels for {control}: {list(model.names.values())}')

    return models


models = load_models()


class YOLO(LabelStudioMLBase):
    """ Label Studio ML Backend based on Ultralytics YOLO
    """

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "yolo")

    def detect_control_tags(self) -> List[ControlModel]:
        """ Use the label config to detect the models to use.
        You can customize this method using `self.project_id` to load different models for specific projects.
        """
        control_tags = []

        for control in self.label_interface.controls:
            # skip control tags that are not presented in the `models`
            if control.tag not in models:
                continue
            # skip if control is not connected to Image
            if control.objects[0].tag != 'Image':
                continue
            # wrong control tag: it doesn't have toName attribute
            if not control.to_name:
                logger.warning(f'{control.tag} {control.name} has no "toName" attribute')
                continue

            # read parameters from control and object tags
            from_name = control.name
            to_name = control.to_name[0]
            value = control.objects[0].value_name

            # read `score_threshold` attribute from the control tag, e.g.: <RectangleLabels score_threshold="0.5">
            score_threshold = float(control.attr.get('score_threshold') or SCORE_THRESHOLD)
            model = models[control.tag]
            model_names = model.names.values()
            label_map = self.build_label_map(from_name, model_names)
            if not label_map:
                logger.error(
                    f"No label map built for the '{control.tag}' control tag.\n"
                    f"This indicates that your Label Studio config labels do not match the model's labels.\n"
                    f"To fix this, ensure that the 'predicted_values' or 'value' attribute in your Label Studio config "
                    f'matches one or more of these model labels.\n'
                    f'Examples:\n<Label value="Car"/>\n'
                    f'<Label value="YourLabel" predicted_values="label1,label2"/>'
                    f"Available '{model.model_name}' model labels: {list(model_names)}"
                )
                continue

            # add control tag with model that we need to use for predictions
            control_model = ControlModel(
                type=control.tag,
                control=control,
                from_name=from_name,
                to_name=to_name,
                value=value,
                model=model,
                score_threshold=score_threshold,
                label_map=label_map
            )
            control_tags.append(control_model)
            logger.debug(f"Control tag with model detected: {control_model}")

        if not control_tags:
            logger.error(f'No control tags detected in the label config for Image object tag')

        return control_tags

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
        control_tags = self.detect_control_tags()
        if not control_tags:
            raise ValueError(f'No suitable control tags (e.g. {list(models.keys())}) detected in the label config')

        predictions = []
        for task in tasks:

            regions = []
            for tag in control_tags:
                image_rgb = self.load_image(tag, task)
                results = tag.model.predict(image_rgb)
                self.debug_plot(results[0].plot(), tag.type)

                if tag.type == 'RectangleLabels':
                    regions += self.create_rectangles(results, task, tag)
                if tag.type in ['Choices', 'Taxonomy']:
                    regions += self.create_choices(results, task, tag)

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

    @staticmethod
    def create_choices(results, task, tag):
        logger.debug(f'create_choices: {tag.from_name}')
        mode = tag.control.attr.get('choice', 'single')
        data = results[0].probs.numpy().data

        # single
        if mode in ['single', 'single-radio']:
            # we must keep data items that matches label_map only, because we need to search among label_map only
            indexes = [i for i, name in tag.model.names.items() if name in tag.label_map]
            data = data[indexes]
            model_names = [tag.model.names[i] for i in indexes]
            # find the best choice
            index = np.argmax(data)
            probs = [data[index]]
            names = [model_names[index]]
        # multi
        else:
            # get indexes of data where data >= tag.score_threshold
            indexes = np.where(data >= tag.score_threshold)
            probs = data[indexes]
            names = tag.model.names[indexes]

        if not probs:
            logger.debug("No choices found")
            return []

        score = np.mean(probs)
        logger.debug(
            "----------------------\n"
            f"task id > {task.get('id')}\n"
            f"type: {tag.control}\n"
            f"probs > {probs}\n"
            f"score > {score}\n"
            f"names > {names}\n"
        )

        if score < tag.score_threshold:
            logger.debug(f"Score is too low for single choice: {names[0]} = {probs[0]}")
            return []

        # map to Label Studio labels
        output_labels = [
            tag.label_map[name] for name in names if name in tag.label_map
        ]

        # add new region with rectangle
        return [{
            "from_name": tag.from_name,
            "to_name": tag.to_name,
            "type": "choices",
            "value": {
                "choices": output_labels
            },
            "score": float(score),
        }]

    @staticmethod
    def create_rectangles(results, task, tag):
        logger.debug(f'create_rectangles: {tag.from_name}')
        label_map, score_threshold = tag.label_map, tag.score_threshold
        data = results[0].boxes  # take bboxes from the first frame
        regions = []

        for i in range(data.shape[0]):  # iterate over items
            score = float(data.conf[i])  # tensor => float
            x, y, w, h = data.xywhn[i].tolist()
            model_label = tag.model.names[int(data.cls[i])]

            logger.debug(
                "----------------------\n"
                f"task id > {task.get('id')}\n"
                f"type: {tag.control}\n"
                f"x, y, w, h > {x, y, w, h}\n"
                f"model label > {model_label}\n"
                f"score > {score}\n"
            )

            # bbox score is too low
            if score < score_threshold:
                continue

            # there is no mapping between model label and LS label
            if model_label not in label_map:
                continue
            output_label = label_map[model_label]

            # add new region with rectangle
            region = {
                "from_name": tag.from_name,
                "to_name": tag.to_name,
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [output_label],
                    "x": (x - w / 2) * 100,
                    "y": (y - h / 2) * 100,
                    "width": w * 100,
                    "height": h * 100,
                },
                "score": score,
            }
            regions.append(region)
        return regions

    @staticmethod
    def load_image(tag, task):
        path = task["data"].get(tag.value) or task["data"].get(DATA_UNDEFINED_NAME)
        logger.debug(f'load_image: {path}')
        path = get_local_path(path, task_id=task.get('id'))
        image = cv2.imread(path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    @staticmethod
    def debug_plot(image, title):
        if not DEBUG_PLOT:
            return

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(title)
        plt.show()
    
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