import logging
import json
from datetime import datetime

from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)


class NewModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(NewModel, self).__init__(**kwargs)

        # Access project labeling configuration via self.parsed_label_config
        # e.g. {"label": {"type": "Labels", "to_name": ["text"], "inputs": ["type": "Text", "value": "text"], "labels": ["Label A", "Label B"]}  # noqa
        logger.debug(f'Create {self.__class__.__name__} with labeling configuration: {json.dumps(self.parsed_label_config, indent=2)}')  # noqa
        if not self.train_output:
            # Initialize your model here...
            logger.debug('Training output is empty: start model from scratch...')
            pass
        else:
            # Read your model checkpoints for example self.train_output["model_checkpoint"]
            logger.debug(f'Read model from previous train run: {self.train_output}')
            pass

    def predict(self, tasks, **kwargs):
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        logger.debug(f'Run prediction on {json.dumps(tasks, indent=2)}')
        return []

    def fit(self, event, data,  **kwargs):
        """
        This method is called each time an annotation is created or updated
        :param kwargs: contains "data" and "event" key, that could be used to retrieve project ID and annotation event type
                        (read more in https://labelstud.io/guide/webhook_reference.html#Annotation-Created)
        :return: dictionary with trained model artefacts that could be used further in code with self.train_output
        """
        if 'data' not in kwargs:
            raise KeyError(f'Project is not identified. Go to Project Settings -> Webhooks, and ensure you have "Send Payload" enabled')
        data = kwargs['data']
        logger.debug(f'Data received: {data}')
        project_id = data['project']['id']
        # write your logic to acquire new tasks, and perform training
        # train_output payload can be retrieved later in subsequent function calls
        return {
            'any_model': 'checkpoint',
            'project': project_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
