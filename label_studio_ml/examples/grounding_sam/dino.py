import os
import pathlib
import logging
import cv2
import numpy as np
import torch

from label_studio_sdk.converter import brush
from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase, ModelResponse
from label_studio_sdk._extensions.label_studio_tools.core.utils.params import get_bool_env
from label_studio_sdk.label_interface.objects import PredictionValue
from segment_anything.utils.transforms import ResizeLongestSide

from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops

# ----Extra Libraries

from typing import Tuple, List

from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.inference import preprocess_caption

logger = logging.getLogger(__name__)


def predict_batch(
    model,
    images: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str = "cuda"
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
    # copy from https://github.com/yuwenmichael/Grounding-DINO-Batch-Inference/blob/main/batch_utlities.py
    '''
    return:
        bboxes_batch: list of tensors of shape (n, 4)
        predicts_batch: list of tensors of shape (n,)
        phrases_batch: list of list of strings of shape (n,)
        n is the number of boxes in one image
    '''
    caption = preprocess_caption(caption=caption)
    model = model.to(device)
    image = images.to(device)
    with torch.no_grad():
        outputs = model(image, captions=[caption for _ in range(
            len(images))])  # <------- I use the same caption for all the images for my use-case
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()  # prediction_logits.shape = (num_batch, nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()  # prediction_boxes.shape = (num_batch, nq, 4)

    mask = prediction_logits.max(dim=2)[0] > box_threshold  # mask: torch.Size([num_batch, 256])

    bboxes_batch = []
    predicts_batch = []
    phrases_batch = []  # list of lists
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    for i in range(prediction_logits.shape[0]):
        logits = prediction_logits[i][mask[i]]  # logits.shape = (n, 256)
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit  # logit is a tensor of shape (256,) torch.Size([256])
            in logits  # torch.Size([7, 256])
        ]
        boxes = prediction_boxes[i][mask[i]]  # boxes.shape = (n, 4)
        phrases_batch.append(phrases)
        bboxes_batch.append(boxes)
        predicts_batch.append(logits.max(dim=1)[0])

    return bboxes_batch, predicts_batch, phrases_batch


# LOADING THE MODEL
groundingdino_model = load_model(
    pathlib.Path(os.environ.get('GROUNDINGDINO_REPO_PATH', "./GroundingDINO")) / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
    pathlib.Path(os.environ.get('GROUNDINGDINO_REPO_PATH', "./GroundingDINO")) / "weights" / "groundingdino_swint_ogc.pth"
)


BOX_THRESHOLD = os.environ.get("BOX_THRESHOLD", 0.3)
TEXT_THRESHOLD = os.environ.get("TEXT_THRESHOLD", 0.25)
LABEL_STUDIO_ACCESS_TOKEN = (
        os.environ.get("LABEL_STUDIO_ACCESS_TOKEN") or os.environ.get("LABEL_STUDIO_API_KEY")
)
LABEL_STUDIO_HOST = (
        os.environ.get("LABEL_STUDIO_HOST") or os.environ.get("LABEL_STUDIO_URL")
)

USE_SAM = get_bool_env("USE_SAM", default=False)
USE_MOBILE_SAM = get_bool_env("USE_MOBILE_SAM", default=False)

MOBILESAM_CHECKPOINT = os.environ.get("MOBILESAM_CHECKPOINT", "mobile_sam.pt")
SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device {device}")


if USE_MOBILE_SAM:
    logger.info(f"Using Mobile-SAM with checkpoint {MOBILESAM_CHECKPOINT}")
    from mobile_sam import SamPredictor, sam_model_registry

    model_checkpoint = MOBILESAM_CHECKPOINT
    reg_key = 'vit_t'
elif USE_SAM:
    logger.info(f"Using SAM with checkpoint {SAM_CHECKPOINT}")
    from segment_anything import SamPredictor, sam_model_registry

    model_checkpoint = SAM_CHECKPOINT
    reg_key = 'vit_h'
else:
    reg_key = None
    model_checkpoint = None
    logger.info("Using GroundingDINO without SAM")

if USE_MOBILE_SAM or USE_SAM:
    logger.info(f"Loading SAM model with checkpoint {model_checkpoint}")
    sam = sam_model_registry[reg_key](checkpoint=model_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    logger.info("SAM model successfully loaded!")


class DINOBackend(LabelStudioMLBase):

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:

        if not context or not context.get('result'):
            # if there is no context, no interaction has happened yet
            return []

        from_name_r, to_name_r, value = self.get_first_tag_occurence('RectangleLabels', 'Image')
        from_name_b, to_name_b, _ = self.get_first_tag_occurence('BrushLabels', 'Image')

        text_prompt = context['result'][0]['value']['text'][0]
        logger.debug(f"Prompt: {text_prompt}")
        
        logger.info(f"the prompt is {text_prompt} and {from_name_r} and {from_name_b}")

        final_predictions = []
        if len(tasks) > 1:
            logger.info(f"Running multiple tasks with {len(tasks)} images")
            final_predictions = self.multiple_tasks(
                tasks, text_prompt, from_name_r, to_name_r, from_name_b, to_name_b, value)
        elif len(tasks) == 1:
            logger.info(f"Running single task {tasks[0]}")
            final_predictions = self.one_task(
                tasks[0], text_prompt, from_name_r, to_name_r, from_name_b, to_name_b, value)
        return final_predictions
        
    def one_task(self, task, prompt, from_name_r, to_name_r, from_name_b, to_name_b, value):
        all_points = []
        all_scores = []
        all_lengths = []
        predictions = []
        raw_img_path = task['data'][value]

        try:
            img_path = self.get_local_path(
                raw_img_path,
                ls_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                ls_host=LABEL_STUDIO_HOST,
                task_id=task.get('id')
            )
        except Exception as e:
            logger.error(f"Error getting image path: {e}")
            img_path = raw_img_path

        src, img = load_image(img_path)

        boxes, logits, _ = predict(
            model=groundingdino_model,
            image=img,
            caption=prompt,
            box_threshold=float(BOX_THRESHOLD),
            text_threshold=float(TEXT_THRESHOLD),
            device=device
        )

        H, W, _ = src.shape

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        points = boxes_xyxy.cpu().numpy()

        for point, logit in zip(points, logits):
            all_points.append(point)
            all_scores.append(logit)
            all_lengths.append((H, W))

        if USE_MOBILE_SAM or USE_SAM:
            # get <BrushLabels> results
            predictions.append(self.get_sam_results(img_path, all_points, all_lengths, from_name_b, to_name_b))
        else:
            # get <RectangleLabels> results
            predictions.append(self.get_results(all_points, all_scores, all_lengths, from_name_r, to_name_r))
        
        return predictions

    def multiple_tasks(self, tasks, prompt, from_name_r, to_name_r, from_name_b, to_name_b, value):

        # first getting all the image paths
        image_paths = []

        for task in tasks:
            raw_img_path = task['data'][value]

            try:
                img_path = self.get_local_path(
                    raw_img_path,
                    ls_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                    ls_host=LABEL_STUDIO_HOST,
                    task_id=task.get('id')
                )
            except Exception as e:
                logger.error(f"Error getting local path: {e}")
                img_path = raw_img_path

            image_paths.append(img_path)

        boxes, logits, lengths = self.batch_dino(image_paths, prompt)

        box_by_task = []
        for (box_task, (H, W)) in zip(boxes, lengths):

            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(box_task) * torch.Tensor([W, H, W, H])

            box_by_task.append(boxes_xyxy)

        if USE_MOBILE_SAM or USE_SAM:
            batched_output = self.batch_sam(input_boxes_list=box_by_task, image_paths=image_paths)
            predictions = self.get_batched_sam_results(batched_output, from_name_b, to_name_b)

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
                
                predictions.append(self.get_results(all_points, all_scores, all_lengths, from_name_r, to_name_r))

        return predictions
            
    # make sure you use new github repo when predicting in batch
    def batch_dino(self, image_paths, prompt):
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
                caption=prompt, # text prompt is same as self.label
                box_threshold=float(BOX_THRESHOLD),
                text_threshold=float(TEXT_THRESHOLD),
                device=device
            )

        else:
            all_boxes = []
            all_logits = []
            for img in loaded_images:
                boxes, logits, _ = predict(
                    model=groundingdino_model,
                    image=img,
                    caption=prompt,
                    box_threshold=float(BOX_THRESHOLD),
                    text_threshold=float(TEXT_THRESHOLD),
                    device=device
                )
                all_boxes.append(boxes)
                all_logits.append(logits)

            boxes = all_boxes
            logits = all_logits

        return boxes, logits, lengths

    def batch_sam(self, input_boxes_list, image_paths):

        resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

        # from SAM code base
        def prepare_image(image, transform, device):
            image = transform.apply_image(image)
            image = torch.as_tensor(image, device=device.device) 
            return image.permute(2, 0, 1).contiguous()

        batched_input = []
        for input_box, path in zip(input_boxes_list, image_paths):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            batched_input.append({
                'image': prepare_image(image, resize_transform, sam),
                'boxes': resize_transform.apply_boxes_torch(input_box, image.shape[:2]),
                'original_size': image.shape[:2]
            })
        
        batched_output = sam(batched_input, multimask_output=False)

        return batched_output
    
    def get_batched_sam_results(self, batched_output, from_name_b, to_name_b):

        predictions = []

        for output in batched_output:
            masks = output['masks']
            masks = masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

            probs = output['iou_predictions'].cpu().numpy()

            num_masks = masks.shape[0]
            height = masks.shape[-2]
            width = masks.shape[-1]

            lengths = [(height, width)] * num_masks

            predictions.append(self.sam_predictions(masks, probs, lengths, from_name_b, to_name_b))

        return predictions

    def get_results(self, all_points, all_scores, all_lengths, from_name_r, to_name_r):
        
        results = []
        total_score = 0
        for points, scores, lengths in zip(all_points, all_scores, all_lengths):
            # random ID
            label_id = str(uuid4())[:9]

            height, width = lengths
            score = scores.item()
            total_score += score

            results.append({
                'id': label_id,
                'from_name': from_name_r,
                'to_name': to_name_r,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'rotation': 0,
                    'width': (points[2] - points[0]) / width * 100,
                    'height': (points[3] - points[1]) / height * 100,
                    'x': points[0] / width * 100,
                    'y': points[1] / height * 100
                },
                'score': score,
                'type': 'rectanglelabels',
                'readonly': False
            })

        total_score /= max(len(results), 1)

        return {
            'result': results,
            'score': total_score,
            'model_version': self.get('model_version')
        }

    def get_sam_results(
        self,
        img_path,
        input_boxes,
        lengths,
        from_name_b,
        to_name_b
    ):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        input_boxes = torch.from_numpy(np.array(input_boxes))

        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2]).to(device)
        masks, probs, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks = masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
        probs = probs.cpu().numpy()

        return self.sam_predictions(masks, probs, lengths, from_name_b, to_name_b)
    
    # takes straight masks and returns predictions
    def sam_predictions(self, masks, probs, lengths, from_name_b, to_name_b):
        
        results = []
        total_score = 0
        for mask, prob, length in zip(masks, probs, lengths):
            height, width = length
            # creates a random ID for your label everytime so no chance for errors
            label_id = str(uuid4())[:9]

            # converting the mask from the model to RLE format which is usable in Label Studio
            mask = mask * 255
            rle = brush.mask2rle(mask)
            score = float(prob[0])

            results.append({
                'id': label_id,
                'from_name': from_name_b,
                'to_name': to_name_b,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle
                },
                'score': score,
                'type': 'brushlabels',
                'readonly': False
            })
            total_score += score
        return {
            'result': results,
            'score': total_score / max(len(results), 1),
            'model_version': self.get('model_version')
        }
