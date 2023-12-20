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

import importlib
batchutil = importlib.import_module("Grounding-DINO-Batch-Inference.batch_utlities")

predict_batch = getattr(batchutil, "predict_batch")


# LOADING THE MODEL
groundingdino_model = load_model("./Grounding-DINO-Batch-Inference/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./Grounding-DINO-Batch-Inference/GroundingDINO/weights/groundingdino_swint_ogc.pth")


BOX_THRESHOLD = os.environ.get("BOX_THRESHOLD", 0.3)
TEXT_THRESHOLD = os.environ.get("TEXT_THRESHOLD", 0.25)
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
        if not context or not context.get('result'):
            # if there is no context, no interaction has happened yet
            return []

        self.from_name_r, self.to_name_r, self.value_r = self.get_first_tag_occurence('RectangleLabels', 'Image')
        self.from_name_b, self.to_name_b, self.value_b = self.get_first_tag_occurence('BrushLabels', 'Image')

        TEXT_PROMPT = context['result'][0]['value']['text'][0]


        x = TEXT_PROMPT.split("_")

        if len(x) > 1:
            self.label = x[1]
            self.prompt = x[0]
        else:
            self.label = x[0]
            self.prompt = x[0]
        
        print(f"the label is {self.label} and prompt {self.prompt} and {self.from_name_r} and {self.from_name_b}")

        # self.label = TEXT_PROMPT.split("_")[0] # make sure that using as text prompt allows you to label it a certain way

        if self.use_sam == 'True':
            self.use_sam=True
        if self.use_sam == 'False':
            self.use_sam = False
        if self.use_ms == 'True':
            self.use_ms = True
        if self.use_ms == 'False':
            self.use_ms = False

            
        if len(tasks) > 1:
            final_predictions = self.multiple_tasks(tasks)
        elif len(tasks) == 1:
            final_predictions = self.one_task(tasks[0])

        return final_predictions
        
    def one_task(self, task):
        all_points = []
        all_scores = []
        all_lengths = []

        predictions = []


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
            caption=self.prompt,
            box_threshold=float(BOX_THRESHOLD),
            text_threshold=float(TEXT_THRESHOLD),
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
        
        return predictions
    

    def multiple_tasks(self, tasks):

        # first getting all the image paths
        image_paths = []

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

            image_paths.append(img_path)

        boxes, logits, lengths = self.batch_dino(image_paths)

        box_by_task = []
        for (box_task, (H, W)) in zip(boxes, lengths):

            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(box_task) * torch.Tensor([W, H, W, H])

            box_by_task.append(boxes_xyxy)

        if self.use_ms or self.use_sam:
            batched_output = self.batch_sam(input_boxes_list=box_by_task, image_paths=image_paths)
            predictions = self.get_batched_sam_results(batched_output)

        else:
            predictions = []

            for boxes_xyxy, (H, W), logits in zip(box_by_task, lengths, logits): 
                points = boxes_xyxy.cpu().numpy()

                all_points = []
                all_scores = []
                all_lengths = []

                for point, logit in zip(points, logits):
                    all_points.append(point)
                    all_scores.append(logit)
                    all_lengths.append((H, W)) # figure out how to get this
                
                predictions.append(self.get_results(all_points, all_scores, all_lengths))            

        return predictions
            
    # make sure you use new github repo when predicting in batch
    def batch_dino(self, image_paths):
        # text prompt is same as self.label
        loaded_images = []
        lengths = []
        for img in image_paths:
            src, img = load_image(img)
            loaded_images.append(img)

            H, W, _ = src.shape

            lengths.append((H, W))

        images = torch.stack(loaded_images)

        if len(image_paths) <= 3:   
            boxes, logits, _ = predict_batch(
                model=groundingdino_model,
                images=images,
                caption=self.prompt, # text prompt is same as self.label
                box_threshold=float(BOX_THRESHOLD),
                text_threshold = float(TEXT_THRESHOLD),
                device=self.device
            )

        else:
            all_boxes = []
            all_logits = []
            for img in loaded_images:
                boxes, logits, _ = predict(
                    model=groundingdino_model,
                    image=img,
                    caption=self.prompt,
                    box_threshold=float(BOX_THRESHOLD),
                    text_threshold=float(TEXT_THRESHOLD),
                    device=DEVICE
                )
                all_boxes.append(boxes)
                all_logits.append(logits)

            boxes = all_boxes
            logits = all_logits

        return boxes, logits, lengths



    
    def batch_sam(self, input_boxes_list, image_paths):

        resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)

        # from SAM code base
        def prepare_image(image, transform, device):
            image = transform.apply_image(image)
            image = torch.as_tensor(image, device=device.device) 
            return image.permute(2, 0, 1).contiguous()


        batched_input = []
        lengths = []
        for input_box, path in zip(input_boxes_list, image_paths):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            batched_input.append({
                'image': prepare_image(image, resize_transform, self.sam),
                'boxes': resize_transform.apply_boxes_torch(input_box, image.shape[:2]),
                'original_size': image.shape[:2]
            })
        
        batched_output = self.sam(batched_input, multimask_output=False)

        return batched_output
    
    def get_batched_sam_results(self, batched_output):

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
            
            #TODO: add model version
            results.append({
                'id': label_id,
                'from_name': self.from_name_r,
                'to_name': self.to_name_r,
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
    
    # takes straight masks and returns predictions
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
                'from_name': self.from_name_b,
                'to_name': self.to_name_b,
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
        return {
            'result': results

        }
