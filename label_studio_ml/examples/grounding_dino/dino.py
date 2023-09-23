import os

from label_studio_converter import brush
from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path

from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops

# ----Extra Libraries
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

# LOADING THE MODEL
groundingdino_model = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./GroundingDINO/weights/groundingdino_swint_ogc.pth")

BOX_TRESHOLD = os.environ.get("BOX_THRESHOLD", 0.3)
TEXT_TRESHOLD = os.environ.get("TEXT_THRESHOLD", 0.25)
LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DINOBackend(LabelStudioMLBase):

    def __init__(self, project_id, **kwargs):
        super(DINOBackend, self).__init__(**kwargs)

        self.label = None

        self.from_name, self.to_name, self.value = None, None, None


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:

        
        self.from_name, self.to_name, self.value = self.get_first_tag_occurence('RectangleLabels', 'Image')


        TEXT_PROMPT = context['result'][0]['value']['text'][0]

        self.label = TEXT_PROMPT

        
        all_points = []
        all_scores = []
        all_lengths = []

        for task in tasks:

            raw_img_path = task['data']['image']

            try:
                img_path = get_image_local_path(
                    raw_img_path,
                    label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                    label_studio_host=LABEL_STUDIO_HOST
                )
            except:
                img_path = raw_img_path

            src, img = load_image(img_path)

            boxes, logits, phrases = predict(
                model=groundingdino_model,
                image=img,
                caption=TEXT_PROMPT.strip("_SAM"),
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                device=DEVICE
            )

            H, W, _ = src.shape

            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

            points = boxes_xyxy.cpu().numpy()

            for point, logit in zip(points, logits):
                all_points.append(point)
                all_scores.append(logit)
                all_lengths.append((H, W))


        predictions = self.get_results(all_points, all_scores, all_lengths, self.from_name, self.to_name, self.label)
        
        return predictions

    def get_results(self, all_points, all_scores, all_lengths, from_name, to_name, label):
        
        results = []
        
        for points, scores, lengths in zip(all_points, all_scores, all_lengths):
            # random ID
            label_id = str(uuid4())[:4]

            height, width = lengths
            
            results.append({
                'id': label_id,
                'from_name': from_name,
                'to_name': to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'rotation': 0,
                    'rectanglelabels': [label],
                    'width': (points[2] - points[0]) / width * 100,
                    'height': (points[3] - points[1]) / height * 100,
                    'x': points[0] / width * 100,
                    'y': points[1] / height * 100
                },
                'score': scores.item(),
                'type': 'rectanglelabels',
                'readonly': False
            })

        
        return [{
            'result': results
        }]
    
if __name__ == '__main__':
    # test the model
    model = DINOBackend()
    model.use_label_config('''
    <View>
        <Image name="image" value="$image" zoom="true"/>
        <RectangleLabels name="tag" toName="image">
            <Label value="Fire" background="#FF0000"/>
        </RectangleLabels>
    </View>
    ''')
    results = model.predict(
        tasks=[{
            'data': {
                # 'image': 'https://s3.amazonaws.com/htx-pub/datasets/images/125245483_152578129892066_7843809718842085333_n.jpg'
                'image': 'label_studio_ml/examples/grounding_dino/fire.jpg'
            }}],
        context= None
    )
    # results[0]['result'][0]['value']['rle'] = f'...{len(results[0]["result"][0]["value"]["rle"])} integers...'
    # print(json.dumps(results, indent=2))