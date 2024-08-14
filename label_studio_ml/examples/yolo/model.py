import os

import cv2
import matplotlib.pyplot as plt
import logging

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralytics import YOLO

from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_ml.utils import DATA_UNDEFINED_NAME


# default threshold for confidences
SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', 0.5))

logger = logging.getLogger(__name__)

# preload default models at the startup
models = {
    'RectangleLabels': YOLO('yolov8m.pt'),  # https://docs.ultralytics.com/tasks/detect/
    'RectangleLabels.OBB': YOLO('yolov8n-obb.pt'),  # https://docs.ultralytics.com/tasks/obb/
    'Keypoints': YOLO('yolov8n-pose.pt'),  # https://docs.ultralytics.com/tasks/pose/
    'PolygonLabels': YOLO('yolov8n-seg.pt'),  # https://docs.ultralytics.com/tasks/segment/
    'Choices': YOLO('yolov8n-cls.pt'),  # https://docs.ultralytics.com/tasks/classify/
}
# Taxonomy is the same as Choices
models['Taxonomy'] = models['Choices']


class YOLO(LabelStudioMLBase):
    """Custom ML Backend model
    """
    def detect_control_tags(self):
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
            if not control.to_name:
                logger.warning(f'{control.tag} {control.name} has no "toName" attribute')
                continue

            # read parameters from control and object tags
            from_name = control.name
            to_name = control.to_name[0]
            value = control.objects[0].value_name
            # read `score_threshold` attribute from the control tag, e.g.: <RectangleLabels score_threshold="0.5">
            score_threshold = float(control.attr.get('score_threshold') or SCORE_THRESHOLD)

            # add model that will generate predictions
            tag = {
                'type': control.tag,
                'control_tag': control,
                'from_name': from_name,
                'to_name': to_name,
                'value': value,
                'model': models[control.tag],
                'score_threshold': score_threshold
            }
            tag['label_map'] = self.build_label_map(from_name, tag['model'].names.values())
            control_tags.append(tag)

        if not control_tags:
            logger.error(f'No control tags detected in the label config for Image object tag')

        return control_tags

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "yolo")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Run YOLO predictions on the tasks
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(f'Run prediction on {len(tasks)} tasks, project ID = {self.project_id}')
        control_tags = self.detect_control_tags()

        predictions = []
        for task in tasks:

            regions = []
            for tag in control_tags:
                path = task["data"].get(tag['value']) or task["data"].get(DATA_UNDEFINED_NAME)
                path = get_local_path(path, task_id=task.get('id'))
                image = cv2.imread(path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results = tag['model'].predict(image_rgb)
                self.debug_plot(results[0].plot(), tag['type'])

                if tag['type'] == 'RectangleLabels':
                    regions += self.create_rectangles(results, task, tag)

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
    def debug_plot(image, title):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(title)
        plt.show()

    def create_rectangles(self, results, task, tag):
        label_map, score_threshold = tag['label_map'], tag['score_threshold']
        data = results[0].boxes  # take boxes from the first frame
        regions = []

        for i in range(data.shape[0]):  # iterate over items
            score = float(data.conf[i])  # tensor => float
            x, y, w, h = data.xywhn[i].tolist()
            model_label = tag['model'].names[int(data.cls[i])]

            logger.debug(
                "----------------------\n"
                f"task id > {task.get('id')}\n"
                f"type: {tag['control_tag']}\n"
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
                "from_name": tag['from_name'],
                "to_name": tag['to_name'],
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [output_label],
                    "x": (x - w/2) * 100,
                    "y": (y - h/2) * 100,
                    "width": w * 100,
                    "height": h * 100,
                },
                "score": score,
            }
            regions.append(region)
        return regions
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
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

# example for simple classification
# return [{
#     "model_version": self.get("model_version"),
#     "score": 0.12,
#     "result": [{
#         "id": "vgzE336-a8",
#         "from_name": "sentiment",
#         "to_name": "text",
#         "type": "choices",
#         "value": {
#             "choices": [ "Negative" ]
#         }
#     }]
# }]