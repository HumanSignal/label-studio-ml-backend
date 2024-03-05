import os
import json
import random
import label_studio_sdk
from uuid import uuid4


from label_studio_ml.model import LabelStudioMLBase


LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', 'you-label-studio-api-key')


class MyModel(LabelStudioMLBase):
    """This simple Label Studio ML backend demonstrates training &
    inference steps with a simple scenario: on training: it gets the
    latest created annotation and stores it as "prediction model"
    artifact on inference: it returns the latest annotation as a
    pre-annotation for every incoming task

    When connected to Label Studio, this is a simple repeater model
    that repeats your last action on a new task

    """

    def predict(self, tasks, **kwargs):
        """ This is where inference happens:
            model returns the list of predictions based on input list of tasks

            :param tasks: Label Studio tasks in JSON format
        """
        # self.train_output is a dict that stores the latest result returned by fit() method
        last_annotation = self.get('last_annotation')
        if last_annotation:
            # results are cached as strings, so we need to parse it back to JSON
            prediction_result_example = json.loads(last_annotation)
            output_prediction = [{
                'result': prediction_result_example,
                'score': random.uniform(0, 1),
                # to control the model versioning, you can use the model_version parameter
                # it will be displayed in the UI and also will be available in the exported results
                'model_version': self.model_version
            }] * len(tasks)
        else:
            output_prediction = []

        print(f'Return output prediction: {json.dumps(output_prediction, indent=2)}')

        return output_prediction

    def download_tasks(self, project):
        """
        Download all labeled tasks from project using the Label Studio SDK.
        Read more about SDK here https://labelstud.io/sdk/
        :param project: project ID
        :return:
        """
        ls = label_studio_sdk.Client(LABEL_STUDIO_HOST, LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project)
        tasks = project.get_labeled_tasks()

        return tasks

    def fit(self, event, data,  **kwargs):
        """
        This method is called each time an annotation is created or updated
        It simply stores the latest annotation as a "prediction model" artifact
        """
        self.set('last_annotation', json.dumps(data['annotation']['result']))
        # to control the model versioning, you can use the model_version parameter
        self.set('model_version', str(uuid4())[:8])
