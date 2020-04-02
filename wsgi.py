import os
import random
import time
import logging

logging.basicConfig(level=logging.DEBUG)

from htx import app, init_model_server
from htx.base_model import BaseModel


class MyModel(BaseModel):

    INPUT_TYPES = ('Image',)
    OUTPUT_TYPES = ('Choices',)

    def load(self, resources, **kwargs):
        self.model_version = resources.get('model_version')
        self.labels = resources.get('labels', [])
        print(f'Model {self.model_version} has been loaded!')
        print(f'Input names: {self.input_names}')
        print(f'Output names: {self.output_names}')

    def predict(self, tasks, **kwargs):
        results = []
        for task in tasks:
            results.append({
                'result': [{
                    'from_name': self.output_names[0],
                    'to_name': self.input_names[0],
                    'type': 'choices',
                    'value': {
                        'choices': [] if not self.labels else random.choice(self.labels)
                    }
                }],
                'score': random.uniform(0, 1)
            })
            print(f'Model {self.model_version} predicts {task}')
        return results

    @classmethod
    def train(cls, input_data, output_dir, **kwargs):
        num_examples = 0
        print(f'Start reading training data...')
        labels = set()
        for task_data, labeling_result in input_data:
            print(f'{task_data} --> {labeling_result}')
            current_image_label = labeling_result[0]['result'][0]['value']['choices'][0]
            labels.add(current_image_label)
            num_examples += 1
        print(f'{num_examples} were read. Start model training...')

        # then actual model training starts...
        time.sleep(10)
        print('Model training finished.')

        # finally we return all training results, possibly any links to the external model storage
        return {
            'model_version': os.path.dirname(output_dir),
            'labels': list(labels)
        }


init_model_server(
    create_model_func=MyModel,
    train_script=MyModel.train,
    model_dir=os.environ.get('MODEL_DIR', '.'),
    redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
    redis_host=os.environ.get('REDIS_HOST', 'localhost'),
    redis_port=os.environ.get('REDIS_PORT', 6379),
)