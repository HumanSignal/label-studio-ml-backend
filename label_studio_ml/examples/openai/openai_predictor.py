import pickle
import os
import numpy as np
import requests
import json
from uuid import uuid4

import openai
import re

# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import make_pipeline

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME, get_env


HOSTNAME = get_env('HOSTNAME', 'http://localhost:8080')
API_KEY = get_env('API_KEY')
openai.api_key = os.environ['OPENAI_API_KEY']

print('=> LABEL STUDIO HOSTNAME = ', HOSTNAME)
if not API_KEY:
    print('=> WARNING! API_KEY is not set')


class OpenAIPrecitor(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(OpenAIPrecitor, self).__init__(**kwargs)

        # then collect all keys from config which will be used to extract data from task and to form prediction
        # Parsed label config contains only one output of <Choices> type
        assert len(self.parsed_label_config) == 1
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        assert self.info['type'] == 'Choices'

        # the model has only one textual input
        assert len(self.info['to_name']) == 1
        assert len(self.info['inputs']) == 1
        assert self.info['inputs'][0]['type'] == 'Text'
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']

    def _get_sentiment(self, input_text):
        prompt = f"Respond in the json format: {{'response': sentiment_classification}}\nText: {input_text}\nSentiment (Positive, Neutral, Negative):"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=40,
            n=1,
            stop=None,
            temperature=0.5,
        )
        response_text =  response.choices[0].message['content'].strip()
        sentiment = re.search("Negative|Neutral|Positive", response_text).group(0)
        # Add input_text back in for the result
        return {"text": input_text, "response": sentiment}

    def predict(self, tasks, **kwargs):
        # collect input texts
        input_texts = []
        for task in tasks:
            input_text = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
            input_texts.append(input_text)

        # get model predictions
        predictions = []
        for text in input_texts:
            predicted_label = self._get_sentiment(text)['response']
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]

            predictions.append({'result': result, 'score': 1.0})

        return predictions
    