import logging
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from uuid import uuid4
from transformers import AutoTokenizer, AutoModelForCausalLM

class PalmyraSmallInteractive(LabelStudioMLBase):

    TEMPERATURE = 0.7

    def __init__(self, **kwargs):
        super(PalmyraSmallInteractive, self).__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("Writer/palmyra-small")
        self.model = AutoModelForCausalLM.from_pretrained("Writer/palmyra-small")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        predictions = []
        model_version = "writer/palmyra-small"
        for task in tasks:
            prompt = task['data']['prompt']
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            outputs = self.model.generate(inputs, max_length=512, temperature=self.TEMPERATURE)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = [{
                'id': str(uuid4())[:4],
                'from_name': 'instruction',
                'to_name': 'prompt',
                'type': 'textarea',
                'value': {
                    'text': generated_text
                }
            }]
            predictions.append({'result': result, 'model_version': model_version})
        return predictions

