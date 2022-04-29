
from PIL import Image
import pytesseract as pt
import boto3
from label_studio_ml.model import LabelStudioMLBase
import pathlib
import os
import logging

logger = logging.getLogger(__name__)
global OCR_config, aws_credentials
OCR_config = "--psm 6"
aws_credentials = {"aws_access_key_id":"",
                "aws_secret_access_key":"",
                "aws_session_token":""
                }

def split_s3_path(s3_path):
    path_parts=s3_path.replace("s3://","").split("/")
    bucket=path_parts.pop(0)
    key="/".join(path_parts)
    return bucket, key

def download_S3_file(img_path_url=None, aws_credentials=None):
    """
    download image file from S3 and save in ./tmp.{file_extension}
    """
    session = boto3.Session(
            aws_access_key_id=aws_credentials["aws_access_key_id"],
            aws_secret_access_key=aws_credentials["aws_secret_access_key"],
            aws_session_token=aws_credentials["aws_session_token"]
    )
    #Then use the session to get the resource
    # s3 = session.resource('s3')
    resource = session.resource('s3')
    bucket, key = split_s3_path(img_path_url)
    file_extension = pathlib.Path(key).suffix
    key_basename = "tmp{}".format(file_extension)
    my_bucket = resource.Bucket(bucket)
    my_bucket.download_file(key, key_basename)
    return key_basename

class BBOXOCR(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(BBOXOCR, self).__init__(**kwargs)

    def predict(self, tasks, **kwargs):
        # extract task meta data: labels, from_name, to_name and other
        task = tasks[0]
        # print("task", task)
        img_path_url = task["data"]["ocr"]
        # print("img_path_url", img_path_url)
        context = kwargs.get('context')
        # print("context", context)
        if context:
            if not context["result"]:
                return []
            result = context.get('result')[0]
            # print("result", result)
            meta = self._extract_meta({**task, **result})
            # print("meta", meta)
            x = meta["x"]*meta["original_width"]/100
            y = meta["y"]*meta["original_height"]/100
            w = meta["width"]*meta["original_width"]/100
            h = meta["height"]*meta["original_height"]/100
            filepath = download_S3_file(img_path_url, aws_credentials)
            IMG = Image.open(filepath)
            result_text = pt.image_to_string(IMG.crop((x,y,x+w,y+h)),
                                            config=OCR_config)
            meta["text"] = result_text
            # print(meta["text"])
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
            # print("temp",temp)
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
            # meta['text'] = task['value']['text']
            # meta['data'] = list(task['data'].values())[0]
            meta['x'] = task['value']['x']
            meta['y'] = task['value']['y']
            meta['width'] = task['value']['width']
            meta['height'] = task['value']['height']
            meta["original_width"] = task['original_width']
            meta["original_height"] = task['original_height']
        return meta
