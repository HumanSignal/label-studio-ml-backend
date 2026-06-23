# based on https://github.com/heartexlabs/label-studio/discussions/1623
# Provides Label Studio ML backend for yolo nas with working custom s3 endpoint for file storage

import os
import logging
import boto3
import io
import json
import torch
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse
from io import BytesIO
from super_gradients.training import models
import super_gradients.training.processing.processing as sg_processing


class MissingYoloLables(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


logger = logging.getLogger(__name__)


class ObjectDetectorModel(LabelStudioMLBase):

    def __init__(self,
                 checkpoint_file=None,
                 image_dir=None,
                 labels_file=None,
                 yolo_labels='labels.txt',
                 score_threshold=0.45,
                 iou_threshold=0.25,
                 img_size=1280,
                 device='cpu', **kwargs):
        """
        Load YoloNAS model from checkpoint into memory.
        Set mappings from yolov classes to target labels
        :param checkpoint_file: Absolute path to yolov serialized model
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param labels_file: file with mappings from yolo labels to custom labels {"airplane": "Boeing"}
        :param yolo_labels: file with yolo label names, plain text with each label on a new line
        :param score_threshold: score threshold to remove predictions below one
        :param iou_threshold: IoU threshold for yolo NMS
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs: endpoint_url - endpoint URL for custom s3 storage
                       yolo_model_type - type of yolo nas model : yolo_nas_s|yolo_nas_m
        """

        super(ObjectDetectorModel, self).__init__(**kwargs)
        self.checkpoint_file = os.environ.get('CHECKPOINT_FILE', None) or checkpoint_file
        self.labels_file = os.environ.get('LABELS_FILE', None) or labels_file
        self.yolo_labels = os.environ.get('YOLO_LABELS', None) or yolo_labels
        self.iou_threshold = float(os.environ.get('IOU_THRESHOLD', None) or iou_threshold)
        self.score_thresh = float(os.environ.get('SCORE_THRESHOLD', None) or score_threshold)
        self.img_size = int(os.environ.get('IMG_SIZE', None) or img_size)
        self.device = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        endpoint_url = kwargs.get('endpoint_url')
        self.endpoint_url = os.environ.get('ENDPOINT_URL', None) or endpoint_url
        if self.endpoint_url:
            logger.info(f'Using s3 endpoint url {self.endpoint_url}')
        yolo_model_type = kwargs.get('yolo_model_type')
        self.yolo_model_type = os.environ.get('YOLO_MODEL_TYPE', None) or yolo_model_type
        if self.yolo_model_type is None:
            self.yolo_model_type == 'yolo_nas_m'

        # read yolo labels from file
        if self.yolo_labels and os.path.exists(self.yolo_labels):
            with open(self.yolo_labels, 'r') as f:
                yolo_labels_list = f.readlines()
            yolo_labels_list = list(map(lambda x: x.strip(), yolo_labels_list))
            yolo_labels_list = list(map(lambda x: x if x[-1] != '\n' else x[:-1], yolo_labels_list))
            self.yolo_labels = yolo_labels_list
            logger.info('Using labels...')
            logger.info(", ".join(self.yolo_labels))
        else:
            raise MissingYoloLables

        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        image_dir = os.environ.get('IMAGE_DIR', None) or image_dir
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')

        # create a label map
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

        logger.info(f"Load new model from {self.checkpoint_file}")

        self.model = models.get(self.yolo_model_type,
                                num_classes=len(self.yolo_labels),
                                checkpoint_path=self.checkpoint_file,
                                checkpoint_num_classes=len(self.yolo_labels)).to(self.device)

        self.model._image_processor = sg_processing.ComposeProcessing(
            [
                sg_processing.ReverseImageChannels(),
                sg_processing.DetectionLongestMaxSizeRescale(output_shape=(self.img_size, self.img_size)),
                sg_processing.DetectionCenterPadding(output_shape=(self.img_size, self.img_size), pad_value=114),
                sg_processing.StandardizeImage(max_value=255.0),
                sg_processing.ImagePermute(permutation=(2, 0, 1)),
            ]
        )

        print(f"confidence threshold {self.score_thresh} and iou threshold {self.iou_threshold}")

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # in case of s3 return raw data
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            if self.endpoint_url:
                client = boto3.client('s3', endpoint_url=self.endpoint_url)
            else:
                client = boto3.client('s3')
            try:
                response = client.get_object(Bucket=bucket_name, Key=key)
                image_data = response['Body'].read()
                # image_url = client.generate_presigned_url(
                #     ClientMethod='get_object',
                #     Params={'Bucket': bucket_name, 'Key': key}
                # )
            except ClientError as ex:
                logger.warning(f'Can\'t process s3 image {image_url}. Reason: {ex}')
            return BytesIO(image_data)
        return image_url

    def predict(self, tasks, **kwargs):
        results = []
        all_scores = []
        for task in tasks:
            image_url = self._get_image_url(task)
            if type(image_url) == str:
                image_path = self.get_local_path(image_url)
            else:
                image_path = image_url
            img = Image.open(image_path)
            img_width, img_height = get_image_size(image_path)
            with torch.no_grad():
                preds = self.model.predict(img,
                                           iou=self.iou_threshold,
                                           conf=self.score_thresh,
                                           fuse_model=False)

            pp = preds[0].prediction

            for (x_min, y_min, x_max, y_max), confidence, label_id in zip(pp.bboxes_xyxy, pp.confidence, pp.labels):
                output_label = self.yolo_labels[int(label_id)]

                # add label name from label_map
                output_label = self.label_map.get(output_label, output_label)
                if output_label not in self.labels_in_config:
                    logger.warning(f'{output_label} label not found in project config.')
                    continue
                if confidence < self.score_thresh:
                    continue

                one_item_response = {
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    "original_width": img_width,
                    "original_height": img_height,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [output_label],
                        'x': float(x_min / img_width * 100),
                        'y': float(y_min / img_height * 100),
                        'width': float((x_max - x_min) / img_width * 100),
                        'height': float((y_max - y_min) / img_height * 100)
                    },
                    'score': float(confidence)
                }

                results.append(one_item_response)
                all_scores.append(confidence)

        avg_score = sum(all_scores) / max(len(all_scores), 1)

        return [{
            'result': results,
            'score': avg_score
        }]


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
