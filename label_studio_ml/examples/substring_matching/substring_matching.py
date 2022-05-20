import logging
import re

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_env
import random
import string
import functools
import requests

logger = logging.getLogger(__name__)

API_KEY = get_env('API_KEY')

class SubstringMatcher(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(SubstringMatcher, self).__init__(**kwargs)

    def predict(self, tasks, **kwargs):
        # extract task meta data: labels, from_name, to_name and other
        task = tasks[0]
        context = kwargs.get('context')
        result = context.get('result')[0]
        meta = self._extract_meta({**task, **result})
        # if no meta data extracted return empty list
        if not meta:
            return []
        # extract simple result
        extracted_data = self._extract_data(meta['data'], meta['value']) if meta['type'] != 'paragraphlabels' \
            else self._extract_paragraph_data(meta['data'], meta['value'])
        if len(extracted_data) == 0:
            return []
        # construct results from extracted data
        results = []
        avg_score = 0
        for item in extracted_data:
            if item['start'] == meta['start'] and item['end'] == meta['end']:
                if 'endOffset' in meta and 'startOffset' in meta:
                    if item['endOffset'] == meta['endOffset'] and item['startOffset'] == meta['startOffset']:
                        continue
                else:
                    continue
            temp = {
                'id': ''.join(
                        random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                        for _ in
                        range(10)),
                'from_name': meta['from_name'],
                'to_name': meta['to_name'],
                'type': meta['type'],
                'value': {
                    'text': item['text'],
                    'start': item['start'],
                    'end': item['end'],
                    meta['type']: meta['labels'],
                },
                'score': item['score']
            }
            if 'startOffset' in item:
                temp['value']['startOffset'] = item['startOffset']
            if 'endOffset' in item:
                temp['value']['endOffset'] = item['endOffset']
            results.append(temp)
            avg_score += item['score']

        return [{
            'result': results,
            'score': avg_score / max(len(extracted_data)-1, 1)
        }]

    @staticmethod
    def _extract_data(data, value):
        result = []
        if isinstance(data, str) and (data.startswith('http://') or data.startswith('https://')):
            data = requests.get(data).text()
        low = value.lower()
        low_data = data.lower()
        for m in re.finditer(low, low_data):
            start = m.start()
            d = data[start:start + len(low)]
            score = functools.reduce(lambda a, b: a+b, [1 if k[0] == k[1] else 0 for k in zip(value, d)]) / len(d)
            temp = {
                'start': start,
                'end': start + len(low),
                'text': d,
                'score': score
            }
            result.append(temp)
        return result

    @staticmethod
    def _extract_paragraph_data(data, value, text_key='text'):
        result = []
        if isinstance(data, str) and (data.startswith('http://') or data.startswith('https://')):
            data = requests.get(data, headers={'Authorization': f'Token {API_KEY}'}).json()
        # extract data to search
        if not isinstance(data, list):
            print("Data is not a list!")
            print(str(data))
            return result
        try:
            low = data[int(value['start'])]
            low = low[text_key]
            low = low[int(value['startOffset']):int(value['endOffset'])].lower()
        except:
            print("Couldn't extract data from task.")
            print(str(data))
            return result
        i = 0
        # search data in each paragraph
        for item in data:
            low_data = item[text_key].lower()
            # iter through found results
            for m in re.finditer(low, low_data):
                start = m.start()
                d = data[int(value['start'])][text_key][int(value['startOffset']):int(value['endOffset'])]
                score = functools.reduce(lambda a, b: a + b, [1 if k[0] == k[1] else 0 for k in zip(value, d)]) / len(d)
                # create result from found data
                temp = {
                    'start': i,
                    'end': i,
                    'startOffset': start,
                    'endOffset': start + len(low),
                    'text': d,
                    'score': score
                }
                result.append(temp)
            i += 1
        return result

    @staticmethod
    def _extract_meta(task):
        meta = dict()
        paragraph = task['type'] == 'paragraphlabels'
        if task:
            meta['id'] = task['id']
            meta['from_name'] = task['from_name']
            meta['to_name'] = task['to_name']
            meta['type'] = task['type']
            meta['labels'] = task['value'][task['type']]
            meta['value'] = task['value'] if paragraph else task['value']['text']
            # data extracted from the task goes here:
            # change to the right key in your data
            meta['data'] = task['data']['transript']
            meta['start'] = int(task['value']['start'])
            meta['end'] = int(task['value']['end'])
        if 'startOffset' in meta['value']:
            meta['startOffset'] = meta['value']['startOffset']
        if 'endOffset' in meta['value']:
            meta['endOffset'] = meta['value']['endOffset']
        return meta
