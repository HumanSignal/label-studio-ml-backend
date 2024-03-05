import binascii
import os
import logging
import random
from inspect import trace

import boto3
import io
import json
import easyocr

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, DATA_UNDEFINED_NAME
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class easyOCRLabeling(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(self,
                 lang_list=None,
                 image_dir=None,
                 labels_file=None,
                 score_threshold=0.3,
                 device='cuda',
                 **kwargs):
        """
        Optionally set mappings from COCO classes to target labels
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param labels_file: file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
        :param score_threshold: score threshold to wipe out noisy results
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs:
        """
        super(easyOCRLabeling, self).__init__(**kwargs)

        lang_list = lang_list or ['mn', 'en']

        # default Label Studio image upload folder
        self.labels_file = labels_file
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')

        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        self.from_name, info = list(self.parsed_label_config.items())[0]
        self.to_name = info['to_name'][0]
        self.value = info['inputs'][0]['value']
        self.labels_in_config = set(info['labels'])

        schema = list(self.parsed_label_config.values())[0]

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

        print(f'Loaded detection model.')

        self.model = easyocr.Reader(
            lang_list=lang_list,
            gpu=True if 'cuda' in device else False,
            download_enabled=True,
            detector=True,
            recognizer=True,
        )
        self.score_thresh = score_threshold

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)

        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def predict(self, tasks, **kwargs):
        # assert len(tasks) == 1  # this affects the retrieve predictions function to auto predict all tasks
        print(tasks)
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)
        model_results = self.model.readtext(image_path, height_ths=0.8)
        results = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)
        if not model_results:
            return
        for poly in model_results:
            output_label = 'Text'
            if not poly:
                continue
            score = poly[-1]
            if score < self.score_thresh:
                continue

            # convert the points array from image absolute dimensions to relative dimensions
            rel_pnt = []
            for rp in poly[0]:
                if rp[0] > img_width or rp[1] > img_height:
                    continue
                rel_pnt.append([(rp[0] / img_width) * 100, (rp[1] / img_height) * 100])

            # must add one for the polygon
            id_gen = random.randrange(10**10)
            results.append({
                'original_width': img_width,
                'original_height': img_height,
                'image_rotation': 0,
                'value': {
                    'points': rel_pnt,
                },
                'id': id_gen,
                'from_name': "polygon",
                'to_name': 'image',
                'type': 'polygon',
                'origin': 'manual',
                'score': score,
            })
            # and one for the transcription
            results.append({
                'original_width': img_width,
                'original_height': img_height,
                'image_rotation': 0,
                'value': {
                    'points': rel_pnt,
                    'labels': [output_label],
                    "text": [
                        poly[1]
                    ]
                },
                'id': id_gen,
                'from_name': "transcription",
                'to_name': 'image',
                'type': 'textarea',
                'origin': 'manual',
                'score': score,
            })
            all_scores.append(score)
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
