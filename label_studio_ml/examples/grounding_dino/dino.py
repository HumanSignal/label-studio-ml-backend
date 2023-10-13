import os

from label_studio_converter import brush
from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
from segment_anything.utils.transforms import ResizeLongestSide

from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops

# ----Extra Libraries
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np


# print("made it here first")

# LOADING THE MODEL
groundingdino_model = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./GroundingDINO/weights/groundingdino_swint_ogc.pth")


BOX_TRESHOLD = os.environ.get("BOX_THRESHOLD", 0.3)
TEXT_TRESHOLD = os.environ.get("TEXT_THRESHOLD", 0.25)
LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")

USE_SAM = os.environ.get("USE_SAM", False)
USE_MOBILE_SAM = os.environ.get("USE_MOBILE_SAM", False)

MOBILESAM_CHECKPOINT = os.environ.get("MOBILESAM_CHECKPOINT", "mobile_sam.pt")
SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")



if USE_MOBILE_SAM:
    from mobile_sam import SamPredictor, sam_model_registry

    model_checkpoint = MOBILESAM_CHECKPOINT
    reg_key = 'vit_t'
elif USE_SAM:
    from segment_anything import SamPredictor, sam_model_registry

    model_checkpoint = SAM_CHECKPOINT
    reg_key = 'vit_h'


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DINOBackend(LabelStudioMLBase):

    def __init__(self, project_id, **kwargs):
        super(DINOBackend, self).__init__(**kwargs)

        self.label = None

        self.from_name, self.to_name, self.value = None, None, None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if USE_MOBILE_SAM or USE_SAM:
            sam = sam_model_registry[reg_key](checkpoint=model_checkpoint)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            self.sam = sam    

        self.use_sam = USE_SAM
        self.use_ms = USE_MOBILE_SAM
            

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:

        self.from_name, self.to_name, self.value = self.get_first_tag_occurence('RectangleLabels', 'Image')

        TEXT_PROMPT = context['result'][0]['value']['text'][0]

        print(f"the tasks are {tasks}")

        print("TESTING IF TEH CODE VOLUME WORKS LETS SEEEE")


        self.label = TEXT_PROMPT.strip("_SAM") # make sure that using as text prompt allows you to label it a certain way

        if self.use_sam == 'True':
            self.use_sam=True
        if self.use_sam == 'False':
            self.use_sam = False
        if self.use_ms == 'True':
            self.use_ms = True
        if self.use_ms == 'False':
            self.use_ms = False
        
        all_points = []
        all_scores = []
        all_lengths = []

        predictions = []

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

            boxes, logits, _ = predict(
                model=groundingdino_model,
                image=img,
                caption=TEXT_PROMPT.strip("_SAM"),
                box_threshold=float(BOX_TRESHOLD),
                text_threshold=float(TEXT_TRESHOLD),
                device=DEVICE
            )

            H, W, _ = src.shape

            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

            points = boxes_xyxy.cpu().numpy()

            for point, logit in zip(points, logits):
                all_points.append(point)
                all_scores.append(logit)
                all_lengths.append((H, W))

            if self.use_ms or self.use_sam:
                predictions.append(self.get_sam_results(img_path, all_points, all_lengths))
            else:
                predictions.append(self.get_results(all_points, all_scores, all_lengths))
            print("this is a test print")
        
        return predictions
    
    # here batch dino



    
    def batch_sam(self, input_boxes_list, image_paths):


        # input_boxes_list should give all the boxes you need, separating info for each tasks in the shape 
        # better if sent in as a tensor
        # but, you will need to change some of the code


        # image_paths are each image for the task

        resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)

        # from SAM code base
        def prepare_image(image, transform, device):
            image = transform.apply_image(image)
            image = torch.as_tensor(image, device=device.device) 
            return image.permute(2, 0, 1).contiguous()


        batched_input = []
        lengths = []
        for input_box, path in zip(input_boxes_list, image_paths):
            input_box = torch.from_numpy(np.array(input_box), device=self.device) # packaging input boxes for each image (change this when you get batched input)
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            batched_input.append({
                'image': prepare_image(image, resize_transform, self.sam),
                'boxes': resize_transform.apply_boxes_torch(input_box, image.shape[:2]),
                'original_size': image.shape[:2]
            })
        
        batched_output = self.sam(batched_input, multimask_output=False)

        predictions = []

        for output in batched_output:
            masks = output['masks']
            masks = masks[:, 0, :, :].cpu().numpy().astype(np.uint8)


            probs = output['iou_predictions'].cpu().numpy()


            num_masks = masks.shape[0]
            height = masks.shape[-2]
            width = masks.shape[-1]

            lengths = [(height, width)] * num_masks

            predictions.append(self.sam_predictions(masks, probs, lengths))

        return predictions











    def get_results(self, all_points, all_scores, all_lengths):
        
        results = []
        
        for points, scores, lengths in zip(all_points, all_scores, all_lengths):
            # random ID
            label_id = str(uuid4())[:9]

            height, width = lengths
            
            results.append({
                'id': label_id,
                'from_name': self.from_name,
                'to_name': self.to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'rotation': 0,
                    'rectanglelabels': [self.label],
                    'width': (points[2] - points[0]) / width * 100,
                    'height': (points[3] - points[1]) / height * 100,
                    'x': points[0] / width * 100,
                    'y': points[1] / height * 100
                },
                'score': scores.item(),
                'type': 'rectanglelabels',
                'readonly': False
            })

        
        return {
            'result': results
        }

    def get_sam_results(
        self,
        img_path,
        input_boxes,
        lengths
    ):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)

        input_boxes = torch.from_numpy(np.array(input_boxes))

        
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, probs, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks = masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
        probs = probs.cpu().numpy()

        return self.sam_predictions(masks, probs, lengths)
    
    def sam_predictions(self, masks, probs, lengths):
        
        results = []

        for mask, prob, length in zip(masks, probs, lengths):
            height, width = length
            # creates a random ID for your label everytime so no chance for errors
            label_id = str(uuid4())[:9]

            # converting the mask from the model to RLE format which is usable in Label Studio
            mask = mask * 255
            rle = brush.mask2rle(mask)

            results.append({
                'id': label_id,
                'from_name': self.from_name,
                'to_name': self.to_name,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': [self.label],
                },
                'score': float(prob[0]),
                'type': 'brushlabels',
                'readonly': False
            })
        return [{
            'result': results
        }]