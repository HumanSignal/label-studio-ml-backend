import os
import spacy
from label_studio_ml.model import LabelStudioMLBase, ModelResponse
from typing import List, Dict, Optional, Union
from label_studio_sdk.objects import PredictionValue

SPACY_MODEL = os.getenv('SPACY_MODEL', 'en_core_web_sm')
nlp = spacy.load(SPACY_MODEL)


class SpacyMLBackend(LabelStudioMLBase):

    # define your custom labels mapping here
    _custom_labels_mapping = {}

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> Union[List[Dict], ModelResponse]:
        from_name, to_name, value = self.label_interface.get_first_tag_occurence('Labels', 'Text')
        predictions = []
        for task in tasks:
            text = task['data'][value]
            doc = nlp(text)
            entities = []
            for ent in doc.ents:
                entities.append({
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'labels',
                    'value': {
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'text': ent.text,
                        'labels': [_custom_labels_mapping.get(ent.label_, ent.label_)]
                    }
                })
            predictions.append(PredictionValue(
                result=entities,
                model_version=SPACY_MODEL
            ))
        return ModelResponse(predictions=predictions, model_version=SPACY_MODEL)
