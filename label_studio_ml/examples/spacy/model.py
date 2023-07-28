import os
import spacy
from label_studio_ml.model import LabelStudioMLBase

SPACY_MODEL = os.getenv('SPACY_MODEL', 'en_core_web_sm')
nlp = spacy.load(SPACY_MODEL)


class SpacyMLBackend(LabelStudioMLBase):

    def predict(self, tasks, context, **kwargs):
        from_name, to_name, value = self.get_first_tag_occurence('Labels', 'Text')
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
                        'labels': [ent.label_]
                    }
                })
            predictions.append({
                'result': entities,
                'model_version': SPACY_MODEL,
            })
        return predictions
