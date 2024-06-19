import logging
import re
import functools
from uuid import uuid4

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

logger = logging.getLogger(__name__)


class InteractiveSubstringMatching(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def _extract_keywords(self, input_text, keyword_to_search, labels, from_name, to_name) -> PredictionValue:
        result = []
        text = input_text.lower()
        kw = keyword_to_search.lower()
        all_scores = []
        logger.debug(f'Searching for keyword: {kw} in text: {text}')
        for m in re.finditer(kw, text):
            start = m.start()
            d = input_text[start:start + len(kw)]
            score = functools.reduce(lambda a, b: a+b, [1 if k[0] == k[1] else 0 for k in zip(kw, d)]) / len(d)
            result.append({
                'id': str(uuid4())[:4],
                'from_name': from_name,
                'to_name': to_name,
                'type': 'labels',
                'value': {
                    'start': start,
                    'end': start + len(kw),
                    'text': d,
                    'labels': labels
                },
                'score': score
            })
            all_scores.append(score)
        return PredictionValue(
            result=result,
            score=sum(all_scores) / max(len(result), 1),
            model_version=self.get('model_version')
        )

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        if not context:
            # return empty predictions if no context is provided
            return ModelResponse(predictions=[])

        from_name, to_name, value = self.label_interface.get_first_tag_occurence('Labels', 'Text')

        result = next((r for r in context.get('result') if r['from_name'] == from_name), None)
        if not result:
            logger.warning(f'No result found in context. Expected from_name: {from_name}')
            return ModelResponse(predictions=[])

        logger.debug(f"Result: {result}")
        predictions = []
        labels = result['value']['labels']
        keyword_to_search = result['value']['text']
        for task in tasks:
            input_text = self.preload_task_data(task, task['data'].get(value))
            if not input_text:
                logger.warning(f"No input text found in task: {task}, input_text={input_text}")
                continue

            prediction = self._extract_keywords(input_text, keyword_to_search, labels, from_name, to_name)
            predictions.append(prediction)
        
        return ModelResponse(predictions=predictions, model_version=self.get("model_version"))
