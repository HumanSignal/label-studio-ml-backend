from label_studio_ml.model import LabelStudioMLBase
import logging
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class GPTIndicBackend(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # Initialization for the ML backend
        super(GPTIndicBackend, self).__init__(**kwargs)
        
        # Load the pre-trained tokenizer and model from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained("aashay96/indic-gpt")
        self.model = AutoModelForCausalLM.from_pretrained("aashay96/indic-gpt")

    def predict(self, tasks, **kwargs):
        predictions = []
        
        for task in tasks:
            # Extract prompt from the task data
            prompt_text = task['data']['prompt']
            inputs = self.tokenizer.encode(prompt_text, return_tensors="pt")
            
            # Generate the response using the model
            outputs = self.model.generate(inputs, max_length=100)
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Structure the prediction result
            predictions.append({
                'result': [{
                    'from_name': 'instruction',
                    'to_name': 'prompt',
                    'type': 'textarea',
                    'value': {'text': [response_text[len(prompt_text):]]},
                }],
                'score': 1.0  # Confidence score
            })
        
        return predictions
