import os
import json
import pathlib
import logging
import cv2
import numpy as np
import torch
import difflib
import xml.etree.ElementTree as ET

from uuid import uuid4
from typing import List, Dict, Optional
from datetime import datetime, timezone

from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk.converter import brush
from label_studio_sdk._extensions.label_studio_tools.core.utils.params import get_bool_env
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path

from groundingdino.util.inference import load_model, load_image, predict, preprocess_caption
from groundingdino.util.utils    import get_phrases_from_posmap
from groundingdino.util           import box_ops 
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.ops import nms

from sentence_transformers import SentenceTransformer
import numpy as np
import torch.nn.functional as F
import ast
import re

logger = logging.getLogger(__name__)
LOG_PATH = '/app/dino_alias_log.jsonl'

# Parse project labels from XML configuration
def parse_project_labels(xml_conf: str) -> List[str]:
    root = ET.fromstring(xml_conf)
    seen, ordered = set(), []
    for tag in root.findall('.//Label'):
        val = tag.attrib.get('value', '').strip()
        if val and val not in seen:
            ordered.append(val)
            seen.add(val)
    return ordered

def embedding_map(raw: str, label_embs: torch.Tensor, ui_labels: List[str]) -> Optional[str]:
    """Return the UI label with highest cosine similarity to raw, or None."""
    # normalize raw
    phrase_emb = embedder.encode(raw, convert_to_tensor=True)       # [D]
    # cosine similarities:
    sims = F.cosine_similarity(phrase_emb, label_embs)              # [N_labels]
    top_idx = int(torch.argmax(sims).cpu().item())
    top_score = float(sims[top_idx].cpu().item())
    if top_score >= EMB_THRESHOLD:
        return ui_labels[top_idx]
    return raw


def safe_parse_extra_params(raw):
    """
    Return dict or {}. Accept dict or JSON string.
    Tolerates raw newlines/tabs by escaping them.
    Never raises.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()

        # Try strict JSON first
        try:
            val = json.loads(s)
            return val if isinstance(val, dict) else {}
        except json.JSONDecodeError:
            pass

        # Sanitize unescaped control chars that break JSON
        s2 = re.sub(r'(?<!\\)\n', r'\\n', s)
        s2 = re.sub(r'(?<!\\)\t', r'\\t', s2)
        try:
            val = json.loads(s2)
            return val if isinstance(val, dict) else {}
        except json.JSONDecodeError:
            pass

        # Python literal fallback for single-quoted dicts
        try:
            import ast
            val = ast.literal_eval(s)
            return val if isinstance(val, dict) else {}
        except Exception:
            logging.warning("extra_params parse failed; using {}")
            return {}
    logging.warning(f"extra_params type unsupported: {type(raw)}; using {{}}")
    return {}

# Load GroundingDINO
groundingdino_model = load_model(
    pathlib.Path(os.environ.get('GROUNDINGDINO_REPO_PATH', './GroundingDINO')) /
      'groundingdino/config/GroundingDINO_SwinT_OGC.py',
    pathlib.Path(os.environ.get('GROUNDINGDINO_REPO_PATH', './GroundingDINO')) /
      'weights/groundingdino_swint_ogc.pth'
)

BOX_THRESHOLD   = float(os.environ.get('BOX_THRESHOLD', 0.45))
TEXT_THRESHOLD  = float(os.environ.get('TEXT_THRESHOLD', 0.5))
LABEL_STUDIO_TOKEN = os.environ.get('LABEL_STUDIO_ACCESS_TOKEN') or os.environ.get('LABEL_STUDIO_API_KEY')
LABEL_STUDIO_HOST  = os.environ.get('LABEL_STUDIO_HOST') or os.environ.get('LABEL_STUDIO_URL')
USE_SAM        = get_bool_env('USE_SAM', default=False)
USE_MOBILE_SAM = get_bool_env('USE_MOBILE_SAM', default=False)
SAM_CHECKPOINT    = os.environ.get('SAM_CHECKPOINT','sam_vit_h_4b8939.pth')
MOBILESAM_CHECKPOINT = os.environ.get('MOBILESAM_CHECKPOINT','mobile_sam.pt')
EMB_THRESHOLD = float(os.environ.get('EMB_THRESHOLD', 0.6))

# Set environment variable for PyTorch CUDA memory allocation
# This helps to avoid fragmentation issues with CUDA memory allocation
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

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
    resize_longest = ResizeLongestSide(sam.image_encoder.img_size)
    logger.info('Loaded SAM model')

# Load SentenceTransformer for label embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Loaded SentenceTransformer embedder")

class DINOBackend(LabelStudioMLBase):
    def __init__(self, project_id=None, label_config=None, **kwargs):
        super().__init__(project_id=project_id, label_config=label_config, **kwargs)
        
        #get labels
        self.ui_labels = parse_project_labels(self.label_config)

        ui_labels = parse_project_labels(label_config)
        # encode once:
        self.label_embs = embedder.encode(ui_labels, convert_to_tensor=True)  # shape [N_labels, D]

        #get prompt from extra_params
        ep_raw = self.get('extra_params') or '{}'   # raw JSON string; don’t json.loads() here
        ep: Dict = safe_parse_extra_params(ep_raw)
        raw_prompt = ep.get('prompt')
        self.project_prompt = self._format_prompt(raw_prompt)
        
        self.set('model_version', 'DINOBackend-v0.0.1')
        logger.info(f'[init] prompt={self.project_prompt!r}, labels={self.ui_labels}')

    def predict(self,
                tasks: List[Dict],
                context: Optional[Dict] = None,
                **kwargs) -> List[Dict]:
        
        if context and context.get('result'):
            prompt = context['result'][0]['value']['text'][0]
        else:
            prompt = self.project_prompt

        logger.info(f"[PREDICT] tasks={len(tasks)} prompt={prompt!r}")

        fn_r, tn_r, img_key = self.get_first_tag_occurence('RectangleLabels', 'Image')
        fn_b, tn_b, _       = self.get_first_tag_occurence('BrushLabels',     'Image')

        if len(tasks) == 1:
            return [self.one_task(tasks[0], prompt, fn_r, tn_r, fn_b, tn_b, img_key)]
        else:
            return self.multiple_tasks(tasks, prompt, fn_r, tn_r, fn_b, tn_b, img_key)

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

        logger.debug(f"Running DINO on task {task['id']} with prompt: {prompt!r}")
        
        boxes_xywh, logits, phrases = predict(
            groundingdino_model, img, prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device
        )
        logger.debug(f"DINO returned phrases: {phrases!r}")

        # convert & nms
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_xywh) * torch.Tensor([W,H,W,H])
        scores_tensor = logits if isinstance(logits, torch.Tensor) else torch.stack(logits)
        keep = nms(boxes_xyxy, scores_tensor, 0.5)[:50]
        boxes_xyxy = boxes_xyxy[keep]
        logits     = [ (logits[i].item() if isinstance(logits, torch.Tensor) else float(logits[i])) for i in keep ]
        phrases    = [ phrases[i] for i in keep ]

        points = boxes_xyxy.cpu().numpy()
        labels = [ embedding_map(p, self.label_embs, self.ui_labels) for p in phrases ]
        self._log_aliases(phrases, labels, task.get('id'))

        # build your rect results dict
        rect_res = self._rect_results(points, logits, [(H,W)]*len(points), labels, fn_r, tn_r)

        # 1) if you _do_ have SAM, merge masks + boxes exactly as before:
        if USE_SAM or USE_MOBILE_SAM:
            mask_res = self._sam_single(img_path, points, labels, (H,W), fn_b, tn_b)
            out = {
                'result': rect_res['result'] + mask_res['result'],
                'score':  (rect_res['score'] + mask_res['score']) / 2,
                'model_version': self.get('model_version'),
            }
            return out

        # 2) otherwise (no SAM), **return the full dict**, not just rect_res['result']:
        mapping = [(p, embedding_map(p, self.label_embs, self.ui_labels)) for p in phrases]
        for raw_phrase, ui_label in mapping:
            logger.info(f"[DEBUG] {raw_phrase:<30} → {ui_label}")
        print("\n[ DINO → UI LABELS ]")
        print("\n".join(f"{r:30} → {u}" for r, u in mapping))

        return rect_res


    def multiple_tasks(self, tasks, prompt, fn_r, tn_r, fn_b, tn_b, img_key):
        img_paths, HWs = [], []
        for t in tasks:
            raw = t['data'][img_key]
            try:
                p = get_local_path(raw, ls_access_token=LABEL_STUDIO_TOKEN, ls_host=LABEL_STUDIO_HOST)
            except:
                p = raw
            img_paths.append(p)
            src, _ = load_image(p)
            HWs.append(src.shape[:2])

        boxes_b, scores_b, phrases_b = [], [], []
        for p in img_paths:
            _, im = load_image(p)
            b, s, ph = predict(
                groundingdino_model, im, prompt,
                BOX_THRESHOLD, TEXT_THRESHOLD, device
            )
            boxes_b.append(b)
            scores_b.append(s)
            phrases_b.append(ph)

        rect_results, sam_boxes, sam_labels = [], [], []
        for b, s, ph, (H, W), task in zip(boxes_b, scores_b, phrases_b, HWs, tasks):
            bxy = box_ops.box_cxcywh_to_xyxy(b) * torch.tensor([W,H,W,H])
            keep = nms(bxy, s, 0.5)[:50]
            bxy, s, ph = bxy[keep], s[keep], [ph[i] for i in keep]

            labels = [embedding_map(x, self.label_embs, self.ui_labels) for x in ph]
            self._log_aliases(ph, labels, task.get('id'))

            rect_results.append(self._rect_results(
                bxy.cpu().numpy(), s, [(H,W)]*len(labels), labels, fn_r, tn_r
            ))
            sam_boxes.append(bxy.cpu().numpy())
            sam_labels.append(labels)

        if not ckpt:
            return [r['result'] for r in rect_results]

        brush_results = [{'result': [], 'score': 0.0} for _ in sam_boxes]
        valid = [i for i,b in enumerate(sam_boxes) if b.shape[0]>0]
        if valid:
            sel_boxes  = [sam_boxes[i]  for i in valid]
            sel_paths  = [img_paths[i]  for i in valid]
            sel_labels = [sam_labels[i] for i in valid]
            sam_out    = self._sam_batch(sel_boxes, sel_paths)
            sel_brush  = self._sam_batch_results(sam_out, sel_labels, fn_b, tn_b)
            for idx, br in zip(valid, sel_brush):
                brush_results[idx] = br

        final = []
        for r, b in zip(rect_results, brush_results):
            final.append({
                'result': r['result'] + b['result'],
                'score':  (r['score'] + b['score']) / 2,
                'model_version': self.get('model_version')
            })
        return final

    def _sam_single(self, img_path, pts, labels, size, fn_b, tn_b):
        H, W = size
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        boxes = predictor.transform.apply_boxes_torch(
            torch.from_numpy(np.asarray(pts)), img.shape[:2]
        ).to(device)
        masks, probs, _ = predictor.predict_torch(
            None, None, boxes, multimask_output=False
        )
        masks = masks[:,0].cpu().numpy().astype(np.uint8)
        probs = probs.cpu().numpy()
        del boxes # free memory
        return self._sam_predictions(masks, probs, labels, [(H,W)]*len(labels), fn_b, tn_b)

    def _sam_batch(self, boxes_list, img_paths):
        resize = ResizeLongestSide(sam.image_encoder.img_size)
        batch = []
        for bxy, p in zip(boxes_list, img_paths):
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            # ← convert numpy→torch BEFORE apply_boxes_torch
            box_tensor = torch.from_numpy(bxy).to(device)
            batch.append({
                "image": self._prep(img, resize),
                "boxes": resize.apply_boxes_torch(box_tensor, img.shape[:2]),
                "original_size": img.shape[:2]
            })
        out = sam(batch, multimask_output=False)
        for o, i in zip(out, batch):
            o["original_size"] = i["original_size"]
        return out

    def _prep(self, img, resize):
        im = resize.apply_image(img)
        t  = torch.as_tensor(im, device=device)
        return t.permute(2,0,1).contiguous()

    def _sam_batch_results(self, sam_out, label_tasks, fn_b, tn_b):
        preds = []
        for out_i, lbls in zip(sam_out, label_tasks):
            H, W = out_i["original_size"]
            masks = out_i["masks"][:,0].cpu().numpy().astype(np.uint8)
            probs = out_i["iou_predictions"].cpu().numpy()
            preds.append(self._sam_predictions(masks, probs, lbls, [(H,W)]*len(lbls), fn_b, tn_b))
        return preds

    def _rect_results(self, points, scores, sizes, labels, fn_r, tn_r):
        res, total = [], 0.0
        for (x0,y0,x1,y1), sc, (H,W), lbl in zip(points, scores, sizes, labels):
            sc = float(sc); total += sc
            res.append({
                'id': str(uuid4())[:9],
                'from_name': fn_r, 'to_name': tn_r,
                'type': 'rectanglelabels','readonly': False,
                'original_width': W,'original_height': H,'image_rotation':0,
                'score': sc,
                'value': {
                    'rotation':0,
                    'width':  (x1-x0)/W*100,
                    'height': (y1-y0)/H*100,
                    'x': x0/W*100,'y': y0/H*100,
                    'rectanglelabels':[lbl]
                }
            })
        return {
            'result': res,
            'score': float(total)/max(len(res),1),
            'model_version': self.get('model_version')
        }

    def _sam_predictions(self, masks, probs, labels, sizes, fn_b, tn_b):
        res, total = [], 0.0
        for m, pr, lbl, (H,W) in zip(masks, probs, labels, sizes):
            score = float(pr[0] if pr.ndim else pr); total += score
            res.append({
                'id': str(uuid4())[:9],
                'from_name': fn_b,'to_name': tn_b,
                'type': 'brushlabels','readonly': False,
                'original_width': W,'original_height': H,'image_rotation':0,
                'score': score,
                'value': {
                    'format':'rle',
                    'rle': brush.mask2rle(m*255),
                    'brushlabels':[lbl]
                }
            })
       
        # free memory
        del masks, probs
        torch.cuda.empty_cache()
        
        return {
            'result': res,
            'score': float(total)/max(len(res),1),
            'model_version': self.get('model_version')
        }

    def _format_prompt(self, prompt):
        if not prompt:
            return ""
        if isinstance(prompt, list):
            # If prompt is a list, join or take first element
            if len(prompt) == 0:
                return ""
            if isinstance(prompt[0], str):
                return prompt[0]
            else:
                return str(prompt[0])
        if not isinstance(prompt, str):
            # Convert anything else to string
            return str(prompt)
        return prompt.strip()


    def _log_aliases(self, raw_phrases: List[str], mapped: List[str], task_id=None):
        for raw, ui in zip(raw_phrases, mapped):
            entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'task_id': task_id,
                'raw_phrase': raw,
                'mapped_label': ui
            }
            logger.info(f"[MAP] task={task_id} {raw!r} → {ui!r}")
            try:
                with open(LOG_PATH, 'a') as fh:
                    fh.write(json.dumps(entry) + '\n')
            except Exception as e:
                logger.warning(f"Could not write alias log: {e}")
