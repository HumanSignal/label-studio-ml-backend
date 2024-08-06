import os
import pathlib
import logging
import torch

from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase, ModelResponse
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops

from typing import Tuple, List

logger = logging.getLogger(__name__)


GROUNDING_DINO_CONFIG = os.getenv('GROUNDING_DINO_CONFIG', 'GroundingDINO_SwinT_OGC.py')
GROUNDING_DINO_WEIGHTS = os.getenv('GROUNDING_DINO_WEIGHTS', 'groundingdino_swint_ogc.pth')

# LOADING THE MODEL
groundingdino_model = load_model(
    pathlib.Path(os.environ.get('GROUNDINGDINO_REPO_PATH', "./GroundingDINO")) / "groundingdino" / "config" / GROUNDING_DINO_CONFIG,
    pathlib.Path(os.environ.get('GROUNDINGDINO_REPO_PATH', "./GroundingDINO")) / "weights" / GROUNDING_DINO_WEIGHTS
)


BOX_THRESHOLD = os.environ.get("BOX_THRESHOLD", 0.3)
TEXT_THRESHOLD = os.environ.get("TEXT_THRESHOLD", 0.25)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device {device}")


class GroundingDINO(LabelStudioMLBase):

    def _get_prompt(self, annotation: Optional[Dict] = None) -> Dict:
        from_name_prompt, _, _ = self.get_first_tag_occurence('TextArea', 'Image')

        if annotation and 'result' in annotation:
            prompt = next(r['value']['text'][0] for r in annotation['result'] if r['from_name'] == from_name_prompt)
            logger.debug(f"Prompt: {prompt}")
            return {
                'prompt': prompt,
                'from_name': from_name_prompt
            }

        prompt = self.get('prompt')
        logger.debug(f"Prompt saved in cache: {prompt}")
        return {
            'prompt': prompt,
            'from_name': from_name_prompt
        }

    def _get_thresholds(self, annotation: Optional[Dict] = None) -> Dict:
        out = {}
        try:
            from_name_box, _, _ = self.get_first_tag_occurence(
                'Number', 'Image', name_filter=lambda n: n.startswith('box_threshold'))
        except Exception as e:
            logger.warning(f"Error getting box_threshold: {e}. Use default values: {BOX_THRESHOLD}")
            out['box_threshold'] = BOX_THRESHOLD
            out['from_name_box'] = None
        else:
            if annotation and 'result' in annotation:
                out['box_threshold'] = next((r['value']['number'] for r in annotation['result'] if r['from_name'] == from_name_box), None)
            else:
                out['box_threshold'] = self.get(from_name_box)

            if not out['box_threshold']:
                out['box_threshold'] = BOX_THRESHOLD
            out['from_name_box'] = from_name_box

        try:
            from_name_text, _, _ = self.get_first_tag_occurence(
                'Number', 'Image', name_filter=lambda n: n.startswith('text_threshold'))
        except Exception as e:
            logger.warning(f"Error getting text_threshold: {e}. Use default values: {TEXT_THRESHOLD}")
            out['text_threshold'] = TEXT_THRESHOLD
            out['from_name_text'] = None
        else:
            if annotation and 'result' in annotation:
                out['text_threshold'] = next((r['value']['number'] for r in annotation['result'] if r['from_name'] == from_name_text), None)
            else:
                out['text_threshold'] = self.get(from_name_text)

            if not out['text_threshold']:
                out['text_threshold'] = TEXT_THRESHOLD
            out['from_name_text'] = from_name_text

        logger.info(f"Thresholds: {out}")
        return out

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
            })

        total_score /= max(len(results), 1)

        return {
            'result': results,
            'score': total_score,
        }

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:

        assert len(tasks) == 1, "Only one task is supported for now"
        task = tasks[0]

        prompt_control = self._get_prompt(context)
        prompt = prompt_control['prompt']
        if not prompt:
            logger.warning("Prompt not found")
            ModelResponse(predictions=[])

        from_name_r, to_name_r, value = self.get_first_tag_occurence('RectangleLabels', 'Image')

        thresh_controls = self._get_thresholds(context)
        BOX_THRESHOLD = float(thresh_controls['box_threshold'])
        TEXT_THRESHOLD = float(thresh_controls['text_threshold'])

        all_points = []
        all_scores = []
        all_lengths = []
        raw_img_path = task['data'][value]

        try:
            img_path = get_local_path(
                raw_img_path,
                task_id=task.get('id')
            )
        except Exception as e:
            logger.error(f"Error getting image path: {e}")
            return ModelResponse(predictions=[])

        src, img = load_image(img_path)

        boxes, logits, _ = predict(
            model=groundingdino_model,
            image=img,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device
        )

        H, W, _ = src.shape

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        points = boxes_xyxy.cpu().numpy()

        for point, logit in zip(points, logits):
            all_points.append(point)
            all_scores.append(logit)
            all_lengths.append((H, W))

        predictions = self.get_results(all_points, all_scores, all_lengths, from_name_r, to_name_r)

        if not context:
            if prompt:
                # prompt restored from cache - show it in the UI
                predictions['result'].append({
                    'from_name': prompt_control['from_name'],
                    'to_name': to_name_r,
                    'type': 'textarea',
                    'value': {
                        'text': [prompt]
                    }
                })
            if thresh_controls['from_name_box']:
                predictions['result'].append({
                    'from_name': thresh_controls['from_name_box'],
                    'to_name': to_name_r,
                    'type': 'number',
                    'value': {
                        'number': BOX_THRESHOLD
                    }
                })
            if thresh_controls['from_name_text']:
                predictions['result'].append({
                    'from_name': thresh_controls['from_name_text'],
                    'to_name': to_name_r,
                    'type': 'number',
                    'value': {
                        'number': TEXT_THRESHOLD
                    }
                })

        return ModelResponse(predictions=[predictions])

    def fit(self, event, data, **additional_params):
        logger.debug(f'Data received: {data}')
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED'):
            return

        prompt = self._get_prompt(data['annotation'])
        if prompt:
            logger.info(f'Storing prompt: {prompt}')
            self.set('prompt', prompt['prompt'])
        else:
            logger.warning("Prompt not found")

        th = self._get_thresholds(data['annotation'])
        self.set('BOX_THRESHOLD', str(th['box_threshold']))
        self.set('TEXT_THRESHOLD', str(th['text_threshold']))
