import boto3
import io
import json
import easyocr
import logging
import os
from uuid import uuid4

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import get_image_size, DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from botocore.exceptions import ClientError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class EasyOCR(LabelStudioMLBase):
    """Custom ML Backend model
    """
    LANG_LIST = list(os.getenv('LANG_LIST', 'mn,en').split(',') or ['mn', 'en'])
    # score threshold to wipe out noisy results
    SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', 0.3))
    # file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
    LABEL_MAPPINGS_FILE = os.getenv('LABEL_MAPPINGS_FILE', 'label_mappings.json')
    # device (cpu, cuda:0, cuda:1, ...)
    DEVICE = os.getenv('DEVICE', 'cuda')
    # Maximum different in box height. Boxes with very different text size should not be merged.
    HEIGHT_THS = float(os.getenv('HEIGHT_THS', 0.8))
    # Label Studio image upload folder:
    # should be used only in case you use direct file upload into Label Studio instead of URLs
    LABEL_STUDIO_ACCESS_TOKEN = (
        os.environ.get("LABEL_STUDIO_ACCESS_TOKEN") or os.environ.get("LABEL_STUDIO_API_KEY")
    )
    LABEL_STUDIO_HOST = (
            os.environ.get("LABEL_STUDIO_HOST") or os.environ.get("LABEL_STUDIO_URL")
    )

    MODEL_DIR = os.getenv('MODEL_DIR', '.')

    _label_map = {}
    _model = None

    def _lazy_init(self):
        if self._model is not None:
            return

        self.model = easyocr.Reader(
            lang_list=self.LANG_LIST,
            gpu=True if 'cuda' in self.DEVICE else False,
            download_enabled=True,
            detector=True,
            recognizer=True,
        )

    def setup(self):
        """Configure any paramaters of your model here
        """
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

        if self.LABEL_MAPPINGS_FILE and os.path.exists(self.LABEL_MAPPINGS_FILE):
            with open(self.LABEL_MAPPINGS_FILE, 'r') as f:
                self._label_map = json.load(f)

    def _get_image_url(self, task, value):
        # TODO: warning! currently only s3 presigned urls are supported with the default keys
        # also it seems not be compatible with file directly uploaded to Label Studio
        # check RND-2 for more details and fix it later
        image_url = task['data'].get(value) or task['data'].get(DATA_UNDEFINED_NAME)

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

    def predict_single(self, task):
        logger.debug('Task data: %s', task['data'])
        from_name_poly, to_name, value = self.get_first_tag_occurence('Polygon', 'Image')
        from_name_labels, _, _ = self.get_first_tag_occurence('Polygon', 'Image')
        from_name_trans, _, _ = self.get_first_tag_occurence('TextArea', 'Image')
        labels = self.label_interface.labels
        labels = sum([list(l) for l in labels], [])
        if len(labels) > 1:
            logger.warning('More than one label in the tag. Only the first one will be used: %s', labels[0])
        label = labels[0]

        image_url = self._get_image_url(task, value)
        cache_dir = os.path.join(self.MODEL_DIR, '.file-cache')
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug(f'Using cache dir: {cache_dir}')
        image_path = get_local_path(
            image_url,
            cache_dir=cache_dir,
            hostname=self.LABEL_STUDIO_HOST,
            access_token=self.LABEL_STUDIO_ACCESS_TOKEN,
            task_id=task.get('id')
        )
        model_results = self.model.readtext(image_path, height_ths=self.HEIGHT_THS)
        if not model_results:
            return
        img_width, img_height = get_image_size(image_path)
        result = []
        all_scores = []
        for res in model_results:
            if not res:
                logger.warning('Empty result from the model')
                continue
            score = res[-1]
            if score < self.SCORE_THRESHOLD:
                logger.info(f'Skipping result with low score: {score}')
                continue

            rel_pnt = []
            for rp in res[0]:
                if rp[0] > img_width or rp[1] > img_height:
                    continue
                rel_pnt.append([(rp[0] / img_width) * 100, (rp[1] / img_height) * 100])

            # must add one for the polygon
            id_gen = str(uuid4())[:4]
            result.append({
                'original_width': img_width,
                'original_height': img_height,
                'image_rotation': 0,
                'value': {
                    'points': rel_pnt,
                },
                'id': id_gen,
                'from_name': from_name_poly,
                'to_name': to_name,
                'type': 'polygon',
                'origin': 'manual',
                'score': score,
            })
            # and one for the transcription
            result.append({
                'original_width': img_width,
                'original_height': img_height,
                'image_rotation': 0,
                'value': {
                    'points': rel_pnt,
                    'labels': [label],
                    "text": [res[1]]
                },
                'id': id_gen,
                'from_name': from_name_trans,
                'to_name': to_name,
                'type': 'textarea',
                'origin': 'manual',
                'score': score,
            })
            all_scores.append(score)

        return {
            'result': result,
            'score': sum(all_scores) / max(len(all_scores), 1),
            'model_version': self.get('model_version'),
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:

        self._lazy_init()
        predictions = []
        for task in tasks:
            # TODO: implement is_skipped() function
            # if is_skipped(task):
            #     continue

            prediction = self.predict_single(task)
            if prediction:
                predictions.append(prediction)

        return ModelResponse(predictions=predictions, model_versions=self.get('model_version'))
