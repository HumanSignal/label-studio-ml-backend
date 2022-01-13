import os
import json
import random
import label_studio_sdk


from label_studio_ml.model import LabelStudioMLBase


LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8000')
LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', 'd6f8a2622d39e9d89ff0dfef1a80ad877f4ee9e3')


class MyModel(LabelStudioMLBase):
    """This simple Label Studio ML backend demonstrates training & inference steps with a simple scenario:
    on training: it gets the latest created annotation and stores it as "prediction model" artifact
    on inference: it returns the latest annotation as a pre-annotation for every incoming task

    When connected to Label Studio, this is a simple repeater model that repeats your last action on a new task
    """

    def predict(self, tasks, **kwargs):
        """ This is where inference happens:
            model returns the list of predictions based on input list of tasks

            :param tasks: Label Studio tasks in JSON format
        """
        # self.train_output is a dict that stores the latest result returned by fit() method
        if self.train_output:
            prediction_result_example = self.train_output['prediction_example']
            output_prediction = [{
                'result': prediction_result_example,
                'score': random.uniform(0, 1)
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

    def fit(self, tasks, workdir=None, **kwargs):
        """
        This method is called each time an annotation is created or updated
        :param kwargs: contains "data" and "event" key, that could be used to retrieve project ID and annotation event type
                        (read more in https://labelstud.io/guide/webhook_reference.html#Annotation-Created)
        :return: dictionary with trained model artefacts that could be used further in code with self.train_output
        """
        if 'data' not in kwargs:
            raise KeyError(f'Project is not identified. Go to Project Settings -> Webhooks, and ensure you have "Send Payload" enabled')
        data = kwargs['data']
        project = data['project']['id']
        tasks = self.download_tasks(project)
        if len(tasks) > 0:
            print(f'{len(tasks)} labeled tasks downloaded for project {project}')
            prediction_example = tasks[-1]['annotations'][0]['result']
            print(f'We\'ll return this as dummy prediction example for every new task:\n{json.dumps(prediction_example, indent=2)}')
            return {
                'prediction_example': prediction_example,
                'also you can put': 'any artefact here'
            }
        else:
            print('No labeled tasks found: make some annotations...')
            return {}
