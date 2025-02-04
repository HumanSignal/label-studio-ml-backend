import os
import logging

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase, ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
from flair.nn import Classifier
from flair.data import Sentence

logger = logging.getLogger(__name__)

FLAIR_MODEL_NAME = os.getenv("FLAIR_MODEL_NAME", "ner-multi")
logger.info(f"Loading Flair model {FLAIR_MODEL_NAME}")
_model = Classifier.load(FLAIR_MODEL_NAME)


class Flair(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def setup(self):
        """Configure any paramaters of your model here
        """
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def convert_to_ls_annotation(self, flair_sentences, from_name, to_name):
        # convert annotations in flair sentences object to labelstudio annotations
        results = []
        for sent in flair_sentences:
            sent_preds = []  # all predictions results for one sentence = one task
            tags = sent.to_dict('ner')
            for ent in tags['entities']:
                labels = [l['value'] for l in ent['labels']]
                if labels:
                    score = min([l['confidence'] for l in ent['labels']])
                    sent_preds.append({
                        'from_name': from_name,
                        'to_name': to_name,
                        'type': 'labels',
                        "value": {
                            "start": ent['start_pos'],
                            "end": ent['end_pos'],
                            "text": ent['text'],
                            "labels": labels
                        },
                        "score": score
                    })

            # add minimum of certaincy scores of entities in sentence for active learning use
            score = min([p['score'] for p in sent_preds]) if sent_preds else 2.0
            results.append(PredictionValue(
                result=sent_preds,
                score=score,
                model_version=self.get('model_version')
            ))

        return results

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        # make predictions with currently set model
        from_name, to_name, value = self.label_interface.get_first_tag_occurence('Labels', 'Text')
        # collect text data for each task in a list and make flair sentences
        flair_sents = [
            Sentence(self.preload_task_data(task, task['data'][value]))
            for task in tasks
        ]
        # predict with ner model for each flair sentence
        for sent in flair_sents:
            _model.predict(sent)

        predictions = self.convert_to_ls_annotation(flair_sents, from_name, to_name)
        return ModelResponse(predictions=predictions, model_version=self.get('model_version'))
