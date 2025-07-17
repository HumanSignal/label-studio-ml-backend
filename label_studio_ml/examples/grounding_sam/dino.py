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
from torchvision.ops import nms

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
        self.set('model_version','DINOBackend-v0.0.1')

    def predict(self, tasks, context=None, **kwargs):
        # get prompt
        if context and context.get('result'):
            prompt = context['result'][0]['value']['text'][0]
        else:
            with open('/app/prompt.txt') as f:
                prompt = f.read().strip()

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

        return {
            'result': results,
            'score': total / (len(results) or 1),
            'model_version': self.get('model_version')
        }

    def get_sam_results(self, img_path, points, labels, sizes, fn, tn):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        tensor = torch.from_numpy(np.array(points))
        boxes  = predictor.transform.apply_boxes_torch(tensor, img.shape[:2]).to(device)
        masks, probs, _ = predictor.predict_torch(
            point_coords=None, point_labels=None,
            boxes=boxes, multimask_output=False
        )
        masks = masks[:,0].cpu().numpy().astype(np.uint8)
        probs = probs.cpu().numpy()
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
