from PIL import Image
import io
import pytesseract as pt
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
import logging
import os
import json
import boto3

logger = logging.getLogger(__name__)
global OCR_config
OCR_config = "--psm 6"

LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST         = os.environ.get("LABEL_STUDIO_HOST")

AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN     = os.environ.get("AWS_SESSION_TOKEN")
AWS_ENDPOINT          = os.environ.get("AWS_ENDPOINT")

S3_TARGET = boto3.resource('s3',
        endpoint_url=AWS_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        config=boto3.session.Config(signature_version='s3v4'),
        verify=False)

class BBOXOCR(LabelStudioMLBase):

    @staticmethod
    def load_image(img_path_url):
        # load an s3 image, this is very basic demonstration code
        # you may need to modify to fit your own needs
        if img_path_url.startswith("s3:"):
            bucket_name = img_path_url.split("/")[2]
            key = "/".join(img_path_url.split("/")[3:])

            obj = S3_TARGET.Object(bucket_name, key).get()
            data =  obj['Body'].read()
            image = Image.open(io.BytesIO(data))
            return image
        else:
            filepath = get_image_local_path(img_path_url,
                    label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                    label_studio_host=LABEL_STUDIO_HOST)
            return  Image.open(filepath)


    def predict(self, tasks, **kwargs):
        # extract task meta data: labels, from_name, to_name and other
        task = tasks[0]
        img_path_url = task["data"]["ocr"]


        context = kwargs.get('context')
        if context:
            if not context["result"]:
                return []

            IMG = self.load_image(img_path_url)

            result = context.get('result')[0]
            meta = self._extract_meta({**task, **result})
            x = meta["x"]*meta["original_width"]/100
            y = meta["y"]*meta["original_height"]/100
            w = meta["width"]*meta["original_width"]/100
            h = meta["height"]*meta["original_height"]/100

            result_text = pt.image_to_string(IMG.crop((x,y,x+w,y+h)),
                                            config=OCR_config).strip()
            meta["text"] = result_text
            temp = {
                "original_width": meta["original_width"],
                "original_height": meta["original_height"],
                "image_rotation": 0,
                "value": {
                    "x": x/meta["original_width"]*100,
                    "y": y/meta["original_height"]*100,
                    "width": w/meta["original_width"]*100,
                    "height": h/meta["original_height"]*100,
                    "rotation": 0,
                    "text": [
                    meta["text"]
                    ]
                },
                "id": meta["id"],
                "from_name": "transcription",
                "to_name": meta['to_name'],
                "type": "textarea",
                "origin": "manual"
            }
            return [{
                'result': [temp, result],
                'score': 0
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
