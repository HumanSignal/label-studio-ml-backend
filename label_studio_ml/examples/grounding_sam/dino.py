import os
import pathlib
import logging
import cv2
import numpy as np
import torch
import difflib

from label_studio_sdk.converter import brush
from typing import List, Dict
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk._extensions.label_studio_tools.core.utils.params import get_bool_env
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops
<<<<<<< Updated upstream

from typing import Tuple, List

from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.inference import preprocess_caption
=======
from torchvision.ops import nms
>>>>>>> Stashed changes

logger = logging.getLogger(__name__)

# — your UI labels —
UI_LABELS = [
    "fire",
    "smoke plumes",
    "hot embers",
    "smoldering zones",
    "tree trunks",
    "industrial tank",
    "industrial pipe",
    "fences",
    "debris",
    "navigable road",
    "thermal hotspots",
    "flooded road",
    "submerged road surface",
    "flood entry-points",
    "drain inlets",
    "chemical leaks",
    "collapsed rubble",
    "damaged buildings",
    "cracked ground",
    "human",
    "emergency personnel",
    "firetrucks",
    "ambulances",
    "hazard tape",
    "cones",
]

def auto_map_label(raw: str) -> str:
    """
    Normalize DINO phrase to one of UI_LABELS:
     - lowercase, replace underscores
     - strip trailing 's' to singularize
     - fuzzy match full phrase then last token
    """
    r = raw.lower().replace("_", " ").strip()
    # exact
    if r in UI_LABELS:
        return r
    # strip plural
    if r.endswith("s") and r[:-1] in UI_LABELS:
        return r[:-1]
    # fuzzy match full
    m = difflib.get_close_matches(r, UI_LABELS, n=1, cutoff=0.6)
    if m:
        return m[0]
    # fuzzy match last word
    token = r.split()[-1]
    m = difflib.get_close_matches(token, UI_LABELS, n=1, cutoff=0.6)
    if m:
        return m[0]
    return r


# Load GroundingDINO
groundingdino_model = load_model(
    pathlib.Path(os.environ.get('GROUNDINGDINO_REPO_PATH', './GroundingDINO')) /
      'groundingdino/config/GroundingDINO_SwinT_OGC.py',
    pathlib.Path(os.environ.get('GROUNDINGDINO_REPO_PATH', './GroundingDINO')) /
      'weights/groundingdino_swint_ogc.pth'
)

BOX_THRESHOLD   = float(os.environ.get('BOX_THRESHOLD', 0.3))
TEXT_THRESHOLD  = float(os.environ.get('TEXT_THRESHOLD', 0.25))
LABEL_STUDIO_TOKEN = os.environ.get('LABEL_STUDIO_ACCESS_TOKEN') or os.environ.get('LABEL_STUDIO_API_KEY')
LABEL_STUDIO_HOST  = os.environ.get('LABEL_STUDIO_HOST') or os.environ.get('LABEL_STUDIO_URL')
USE_SAM        = get_bool_env('USE_SAM', default=False)
USE_MOBILE_SAM = get_bool_env('USE_MOBILE_SAM', default=False)
SAM_CHECKPOINT    = os.environ.get('SAM_CHECKPOINT','sam_vit_h_4b8939.pth')
MOBILESAM_CHECKPOINT = os.environ.get('MOBILESAM_CHECKPOINT','mobile_sam.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f'Using device {device}')

if USE_MOBILE_SAM:
    from mobile_sam import SamPredictor, sam_model_registry
    ckpt, reg_key = MOBILESAM_CHECKPOINT, 'vit_t'
elif USE_SAM:
    from segment_anything import SamPredictor, sam_model_registry
    ckpt, reg_key = SAM_CHECKPOINT, 'vit_h'
else:
    ckpt = reg_key = None
    logger.info('Running without SAM')

if USE_SAM or USE_MOBILE_SAM:
    sam = sam_model_registry[reg_key](checkpoint=ckpt).to(device)
    predictor = SamPredictor(sam)
    logger.info('Loaded SAM model')


class DINOBackend(LabelStudioMLBase):
    def setup(self):
<<<<<<< Updated upstream
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')
        # load prompt->label mappings from prompt.txt ===
        self.label_map: Dict[str, str] = {}
        try:
            with open("/app/prompt.txt", "r") as f:
                for line in f:
                    line = line.strip().rstrip(',')
                    if not line:
                        continue
                    key, val = line.split("_", 1)
                    self.label_map[key] = val
            logger.info(f"Loaded {len(self.label_map)} label mappings from prompt.txt")
        except Exception as e:
            logger.error(f"Failed to load prompt.txt mappings: {e}")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
=======
        self.set('model_version','DINOBackend-v0.0.1')
>>>>>>> Stashed changes

    def predict(self, tasks, context=None, **kwargs):
        # get prompt
        if context and context.get('result'):
            prompt = context['result'][0]['value']['text'][0]
        else:
<<<<<<< Updated upstream
            # fallback to mounted prompt.txt
            text_prompt = " ".join(self.label_map.keys())
=======
            with open('/app/prompt.txt') as f:
                prompt = f.read().strip()
>>>>>>> Stashed changes

        fn_r, tn_r, img_key = self.get_first_tag_occurence('RectangleLabels','Image')
        fn_b, tn_b, _      = self.get_first_tag_occurence('BrushLabels','Image')

        if len(tasks) > 1:
            return self.multiple_tasks(tasks,prompt,fn_r,tn_r,fn_b,tn_b,img_key)
        return [ self.one_task(tasks[0],prompt,fn_r,tn_r,fn_b,tn_b,img_key) ]

    def one_task(self, task, prompt, fn_r, tn_r, fn_b, tn_b, img_key):
        raw = task['data'][img_key]
        try:
            img_path = self.get_local_path(raw,
                                          ls_access_token=LABEL_STUDIO_TOKEN,
                                          ls_host=LABEL_STUDIO_HOST)
        except:
            img_path = raw

        src, img = load_image(img_path)
        H, W, _  = src.shape

        boxes_xywh, logits, phrases = predict(
            groundingdino_model, img, prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device
        )

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_xywh) * torch.Tensor([W,H,W,H])
        scores_tensor = logits if isinstance(logits,torch.Tensor) else torch.stack(logits)
        keep = nms(boxes_xyxy, scores_tensor, 0.5)[:50]

        boxes_xyxy = boxes_xyxy[keep]
        logits     = [ (logits[i].item() if isinstance(logits,torch.Tensor) else float(logits[i]))
                       for i in keep ]
        phrases    = [ phrases[i] for i in keep ]

        points = boxes_xyxy.cpu().numpy()
<<<<<<< Updated upstream

        for point, logit in zip(points, logits):
            all_points.append(point)
            all_scores.append(logit)
            all_lengths.append((H, W))

        if USE_MOBILE_SAM or USE_SAM:
            # get <BrushLabels> results
            predictions.append(self.get_sam_results(img_path, all_points, all_lengths, from_name_b, to_name_b, prompt))
        else:
            # get <RectangleLabels> results
            predictions.append(self.get_results(all_points, all_scores, all_lengths, from_name_r, to_name_r, prompt))

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
            predictions = self.get_batched_sam_results(batched_output, from_name_b, to_name_b, prompt)

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
                

                predictions.append(self.get_results(all_points, all_scores, all_lengths, from_name_r, to_name_r, prompt))

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
    
    def get_batched_sam_results(self, batched_output, from_name_b, to_name_b, prompt):

        predictions = []

        for output in batched_output:
            masks = output['masks']
            masks = masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

            probs = output['iou_predictions'].cpu().numpy()

            num_masks = masks.shape[0]
            height = masks.shape[-2]
            width = masks.shape[-1]

            lengths = [(height, width)] * num_masks

            predictions.append(self.sam_predictions(masks, probs, lengths, from_name_b, to_name_b, prompt))

        return predictions

    def get_results(self, all_points, all_scores, all_lengths, from_name_r, to_name_r, prompt):
        
        results = []
        total_score = 0
        for points, scores, lengths in zip(all_points, all_scores, all_lengths):
            # random ID
            label_id = str(uuid4())[:9]

            height, width = lengths
            score = scores.item()
            total_score += score

        # map prompt key to final label
        mapped_label = self.label_map.get(prompt, prompt)

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
                'y': points[1] / height * 100,
                'labels': [mapped_label]  
            },
            'score': score,
            'type': 'rectanglelabels',
            'readonly': False
        })

=======
        labels = [ auto_map_label(p) for p in phrases ]

        rect_res = self.get_results(points,logits,labels,[(H,W)]*len(points),
                                    fn_r,tn_r)

        if USE_SAM or USE_MOBILE_SAM:
            mask_res = self.get_sam_results(img_path,points,labels,
                                            [(H,W)]*len(points),
                                            fn_b,tn_b)
            return {
                'result':        rect_res['result'] + mask_res['result'],
                'score':         (rect_res['score'] + mask_res['score'])/2,
                'model_version': self.get('model_version'),
            }

        mapping = [(p, auto_map_label(p)) for p in phrases]
        for raw, ui in mapping:
            logger.info(f"[DEBUG] {raw:<30} → {ui}")
        # optionally: print to stdout so it shows up in your container logs
        print("\n[ DINO → UI LABELS ]")
        print("\n".join(f"{raw:30} → {ui}" for raw, ui in mapping))
        # then unpack labels as usual
        labels = [ui for _, ui in mapping]


        return rect_res

    def multiple_tasks(self,*args,**kwargs):
        raise NotImplementedError

    def get_results(self, points, scores, labels, sizes, fn, tn):
        results, total = [], 0.0
        for (x0,y0,x1,y1), scr, lbl in zip(points,scores,labels):
            score = float(scr)
            total += score

            W, H = sizes[0][1], sizes[0][0]
            x_pct, y_pct = x0/W*100, y0/H*100
            w_pct, h_pct = (x1-x0)/W*100, (y1-y0)/H*100

            results.append({
                'id': str(uuid4())[:9],
                'from_name': fn, 'to_name': tn,
                'type':'rectanglelabels',
                'value':{
                  'x':x_pct,'y':y_pct,
                  'width':w_pct,'height':h_pct,
                  'rotation':0,
                  'rectanglelabels':[lbl]
                },
                'score': score,
                'original_width':  W,
                'original_height': H,
                'image_rotation':  0,
                'readonly': False,
            })
>>>>>>> Stashed changes

        return {
            'result': results,
            'score': total / (len(results) or 1),
            'model_version': self.get('model_version')
        }

<<<<<<< Updated upstream
    def get_sam_results(
        self,
        img_path,
        input_boxes,
        lengths,
        from_name_b,
        to_name_b, 
        prompt
    ):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        input_boxes = torch.from_numpy(np.array(input_boxes))

        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2]).to(device)
=======
    def get_sam_results(self, img_path, points, labels, sizes, fn, tn):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        tensor = torch.from_numpy(np.array(points))
        boxes  = predictor.transform.apply_boxes_torch(tensor, img.shape[:2]).to(device)
>>>>>>> Stashed changes
        masks, probs, _ = predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=boxes, multimask_output=False
        )
        masks = masks[:,0].cpu().numpy().astype(np.uint8)
        probs = probs.cpu().numpy()
<<<<<<< Updated upstream

        return self.sam_predictions(masks, probs, lengths, from_name_b, to_name_b, prompt)

    # takes straight masks and returns predictions
    def sam_predictions(self, masks, probs, lengths, from_name_b, to_name_b, prompt):
        
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

            # map prompt key to final label
            mapped_label = self.label_map.get(prompt, prompt)

            results.append({
                'id': label_id,
                'from_name': from_name_b,
                'to_name': to_name_b,
                'original_width': width,
                'original_height': height,
                'image_rotation': 0,
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'labels': [mapped_label]
=======
        return self.sam_predictions(masks, probs, labels, sizes, fn, tn)

    def sam_predictions(self, masks, probs, labels, sizes, fn, tn):
        results, total = [], 0.0
        for mask, prob, lbl in zip(masks, probs, labels):
            rle = brush.mask2rle(mask*255)
            score = float(prob[0] if prob.ndim>0 else prob)
            total += score
            results.append({
                'id': str(uuid4())[:9],
                'from_name': fn, 'to_name': tn,
                'type':'brushlabels',
                'value':{
                  'format':'rle','rle':rle,
                  'brushlabels':[ lbl ]
>>>>>>> Stashed changes
                },
                'score': score,
                'original_width':  sizes[0][1],
                'original_height': sizes[0][0],
                'image_rotation': 0
            })

        return {
            'result': results,
            'score': total / (len(results) or 1),
            'model_version': self.get('model_version')
        }
