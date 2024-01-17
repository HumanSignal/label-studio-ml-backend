from PIL import Image
import pytesseract as pt
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
import logging
import os

logger = logging.getLogger(__name__)
global OCR_config
OCR_config = "--psm 6"

LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")

class BBOXOCR(LabelStudioMLBase):

    def predict(self, tasks, **kwargs):
        # extract task meta data: labels, from_name, to_name and other
        task = tasks[0]
        img_path_url = task["data"]["ocr"]

        context = kwargs.get('context')
        if context:
            if not context["result"]:
                return []
            result = context.get('result')[0]
            meta = self._extract_meta({**task, **result})
            x = meta["x"]*meta["original_width"]/100
            y = meta["y"]*meta["original_height"]/100
            w = meta["width"]*meta["original_width"]/100
            h = meta["height"]*meta["original_height"]/100
            filepath = get_image_local_path(img_path_url,
                    label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                    label_studio_host=LABEL_STUDIO_HOST)
            IMG = Image.open(filepath)
            result_text = pt.image_to_string(IMG.crop((x,y,x+w,y+h)),
                                            config=OCR_config)
            meta["text"] = result_text
            print(result_text)
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
                'result': [result, temp],
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
