import logging
import re

from label_studio_ml.model import LabelStudioMLBase
import random
import string

logger = logging.getLogger(__name__)


class SubstringMatcher(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(SubstringMatcher, self).__init__(**kwargs)

    def predict(self, task, **kwargs):
        # extract task meta data: labels, from_name, to_name and other
        meta = self._extract_meta(task)
        # if no meta data extracted return empty list
        if not meta:
            return []
        # extract simple result
        extracted_data = self._extract_data(meta['data'], meta['value'])
        if len(extracted_data) == 0:
            return []
        # construct results from extracted data
        results = []
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
                }
            }
            results.append(temp)

        return [{
            'result': results
        }]

    @staticmethod
    def _extract_data(data, value):
        result = []
        low = value.lower()
        low_data = data.lower()
        for m in re.finditer(low, low_data):
            start = m.start()
            temp = {
                'start': start,
                'end': start + len(low),
                'text': data[start:start + len(low)]
            }
            result.append(temp)
        return result

    @staticmethod
    def _extract_meta(task):
        meta = dict()
        annotation = task.get('annotations')
        if annotation:
            data = annotation[0]['result'][0]
            meta['id'] = data['id']
            meta['from_name'] = data['from_name']
            meta['to_name'] = data['to_name']
            meta['type'] = data['type']
            meta['labels'] = data['value'][data['type']]
            meta['value'] = data['value']['text']
            meta['data'] = list(task['data'].values())[0]
            meta['start'] = data['value']['start']
            meta['end'] = data['value']['end']
        return meta
