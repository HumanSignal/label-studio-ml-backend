import logging
import re

from label_studio_ml.model import LabelStudioMLBase
import random
import string
import functools
import requests

logger = logging.getLogger(__name__)


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
        extracted_data = self._extract_data(meta['data'], meta['value'])
        if len(extracted_data) == 0:
            return []
        # construct results from extracted data
        results = []
        avg_score = 0
        for item in extracted_data:
            if item['start'] == meta['start'] and item['end'] == meta['end']:
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
            results.append(temp)
            avg_score += item['score']

        return [{
            'result': results,
            'score': avg_score / max(len(extracted_data)-1, 1)
        }]

    @staticmethod
    def _extract_data(data, value):
        result = []
        if data.startswith('http://') or data.startswith('https://'):
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
    def _extract_meta(task):
        meta = dict()
        if task:
            meta['id'] = task['id']
            meta['from_name'] = task['from_name']
            meta['to_name'] = task['to_name']
            meta['type'] = task['type']
            meta['labels'] = task['value'][task['type']]
            meta['value'] = task['value']['text']
            meta['data'] = list(task['data'].values())[0]
            meta['start'] = task['value']['start']
            meta['end'] = task['value']['end']
        return meta
