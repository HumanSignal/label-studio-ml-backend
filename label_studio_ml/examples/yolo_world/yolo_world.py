import os
import logging
from uuid import uuid4
from urllib.parse import urlparse
from typing import List, Dict, Optional

from ultralytics import YOLOWorld
import boto3
from botocore.exceptions import ClientError

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path


APPLICATION_NAME = "task_label_studio_backend_yolo_world"
logger = logging.getLogger(APPLICATION_NAME)

# Model checkpoint
CHECKPOINT = os.environ.get("CHECKPOINT", "yolov8l-world.pt")

# Label Studio
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")
LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")

# S3 credentials
AWS_ENDPOINT_URL = os.environ.get('AWS_ENDPOINT_URL')
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS = os.environ.get('AWS_SECRET_ACCESS')

# Model thresholds
CONF_THRESHOLD = os.environ.get("CONF_THRESHOLD", 0.1)
IOU_THRESHOLD = os.environ.get("IOU_THRESHOLD", 0.3)


class YOLOWorldBackend(LabelStudioMLBase):

    def __init__(self, project_id, **kwargs):
        # don't forget to initialize base class...
        super().__init__(**kwargs)
        self.model = YOLOWorld(CHECKPOINT)
        self.conf_thres = float(CONF_THRESHOLD)
        self.iou_thres = float(IOU_THRESHOLD)

    @staticmethod
    def _get_image_url(image_url):
        if image_url.startswith('s3://'):
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3',
                                  endpoint_url=AWS_ENDPOINT_URL,
                                  aws_access_key_id=AWS_ACCESS_KEY,
                                  aws_secret_access_key=AWS_SECRET_ACCESS,
                                  )
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs):
        if not context or not context.get('result'):
            return []
        prompt = context['result'][0]['value']['text'][0]
        logger.info(f"PROMPT: {prompt}")        
        self.from_name_r, self.to_name_r, self.value_r = self.get_first_tag_occurence('RectangleLabels', 'Image')
        return self._predict(tasks, prompt)

    def _predict(self, tasks: List[Dict], prompt: str):

        # parse prompt
        labels = prompt.split(", ") 
        self.model.set_classes(labels)

        image_paths = []
        for task in tasks:
            raw_img_path = task['data']['image']
            try:
                image_url = self._get_image_url(raw_img_path)
                img_path = get_image_local_path(
                    image_url,
                    label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                    label_studio_host=LABEL_STUDIO_HOST
                )
            except:
                img_path = raw_img_path
            image_paths.append(img_path)

        predictions = []
        for image_path in image_paths:
            result = self.model.predict(image_path, conf=self.conf_thres, iou=self.iou_thres)[0]
            img_height, img_width = result.orig_shape
            box_by_task = result.boxes.xyxy.cpu().numpy().astype(float)
            scores = result.boxes.conf.cpu().numpy().astype(float)
            classes =result.boxes.cls.cpu().numpy().astype(int)
            all_points = []
            all_scores = []
            all_lengths = []
            all_classes = []

            for box, score, cls in zip(box_by_task, scores, classes):
                all_points.append(box)
                all_scores.append(score)
                all_classes.append(labels[cls])
                all_lengths.append((img_height, img_width))

            predictions.append(self.get_results(all_points, all_scores, all_classes, all_lengths))

        return predictions

    def get_results(self, all_points, all_scores, all_classes, all_lengths):
        results = []
        for box, score, cls, length in zip(all_points, all_scores, all_classes, all_lengths):
            # random ID
            label_id = str(uuid4())[:9]

            height, width = length
            results.append({
                'id': label_id,
                'from_name': self.from_name_r,
                'to_name': self.to_name_r,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'rotation': 0,
                    'rectanglelabels': [cls],
                    'width': (box[2] - box[0]) / width * 100,
                    'height': (box[3] - box[1]) / height * 100,
                    'x': box[0] / width * 100,
                    'y': box[1] / height * 100
                },
                'score': score,
                'type': 'rectanglelabels',
                'readonly': False
            })

        return {
            'result': results
        }
