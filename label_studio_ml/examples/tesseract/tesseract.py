import io
import logging
import os

import boto3
import pytesseract as pt
from PIL import Image, ImageOps

from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)
global OCR_config
OCR_config = "--psm 6 -l chi_sim+eng+deu"

LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.environ.get("AWS_SESSION_TOKEN")
AWS_ENDPOINT = os.environ.get("AWS_ENDPOINT")

S3_TARGET = boto3.resource('s3',
                           endpoint_url=AWS_ENDPOINT,
                           aws_access_key_id=AWS_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                           aws_session_token=AWS_SESSION_TOKEN,
                           config=boto3.session.Config(signature_version='s3v4'),
                           verify=False)


class BBOXOCR(LabelStudioMLBase):
    MODEL_DIR = os.environ.get('MODEL_DIR', '.')

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def load_image(self, img_path_url, task_id):
        # load an s3 image, this is very basic demonstration code
        # you may need to modify to fit your own needs
        if img_path_url.startswith("s3:"):
            bucket_name = img_path_url.split("/")[2]
            key = "/".join(img_path_url.split("/")[3:])

            obj = S3_TARGET.Object(bucket_name, key).get()
            data = obj['Body'].read()
            image = Image.open(io.BytesIO(data))
            image = ImageOps.exif_transpose(image)
            return image
        else:
            cache_dir = os.path.join(self.MODEL_DIR, '.file-cache')
            os.makedirs(cache_dir, exist_ok=True)
            logger.debug(f'Using cache dir: {cache_dir}')
            filepath = self.get_local_path(
                img_path_url,
                cache_dir=cache_dir,
                ls_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                ls_host=LABEL_STUDIO_HOST,
                task_id=task_id
            )
            image = Image.open(filepath)
            image = ImageOps.exif_transpose(image)
            return image

    def predict(self, tasks, **kwargs):
        # extract task metadata: labels, from_name, to_name and other
        from_name, to_name, value = self.label_interface.get_first_tag_occurence(
            'TextArea',
            'Image'
        )
        task = tasks[0]
        img_path_url = task["data"][value]

        context = kwargs.get('context')
        if context:
            if not context["result"]:
                return []

            image = self.load_image(img_path_url, task.get('id'))

            result = context.get('result')[-1]
            meta = self._extract_meta({**task, **result})
            x = meta["x"] * meta["original_width"] / 100
            y = meta["y"] * meta["original_height"] / 100
            w = meta["width"] * meta["original_width"] / 100
            h = meta["height"] * meta["original_height"] / 100

            result_text = pt.image_to_string(
                image.crop((x, y, x + w, y + h)),
                config=OCR_config
            ).strip()
            meta["text"] = result_text
            temp = {
                "original_width": meta["original_width"],
                "original_height": meta["original_height"],
                "image_rotation": 0,
                "value": {
                    "x": x / meta["original_width"] * 100,
                    "y": y / meta["original_height"] * 100,
                    "width": w / meta["original_width"] * 100,
                    "height": h / meta["original_height"] * 100,
                    "rotation": 0,
                    "text": [
                        meta["text"]
                    ]
                },
                "id": meta["id"],
                "from_name": from_name,
                "to_name": meta['to_name'],
                "type": "textarea",
                "origin": "manual"
            }
            return [{
                'result': [temp, result],
                'score': 0,
                'model_version': self.get('model_version')
            }]
        else:
            return []

    @staticmethod
    def _extract_meta(task):
        meta = dict()
        if task:
            meta['id'] = task['id']
            meta['from_name'] = task['from_name']
            meta['to_name'] = task['to_name']
            meta['type'] = task['type']
            meta['x'] = task['value']['x']
            meta['y'] = task['value']['y']
            meta['width'] = task['value']['width']
            meta['height'] = task['value']['height']
            meta["original_width"] = task['original_width']
            meta["original_height"] = task['original_height']
        return meta
