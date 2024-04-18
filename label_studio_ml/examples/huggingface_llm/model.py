import os
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from transformers import pipeline


class HuggingFaceLLM(LabelStudioMLBase):
    """Custom ML Backend model
    """
    MODEL_NAME = os.getenv('MODEL_NAME', 'facebook/opt-125m')
    MAX_LENGTH = int(os.getenv('MAX_LENGTH', 50))
    _model = None
    
    def setup(self):
        """Configure any paramaters of your model here
        """
        self.set("model_version", "0.0.1")


    def _lazy_init(self):
        if self._model is not None:
            return

        self._model = pipeline('text-generation', model=self.MODEL_NAME)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        self._lazy_init()
        from_name, to_name, value = self.label_interface.get_first_tag_occurence('TextArea', 'Text')
        predictions = []
        for task in tasks:
            text = task['data'][value]
            result = self._model(text, max_length=self.MAX_LENGTH)
            generated_text = result[0]['generated_text']
            # cut the `text` prefix
            generated_text = generated_text[len(text):].strip()
            predictions.append({
                'result': [{
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'textarea',
                    'value': {
                        'text': [generated_text]
                    }
                }]
            })
        
        return ModelResponse(predictions=predictions, model_version=self.get("model_version"))
