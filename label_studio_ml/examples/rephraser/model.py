from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse


QUESTION_TEXT_TAG = 'question'
REPHRASER_TEXTAREA_TAG = 'repharesed'
RELEVANT_TEXTAREA_TAG = 'relevant'


class RephraserModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

    def generate_reprhased_questions(self, question):
        return ['test1', 'test2', 'test3']

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        if not tasks:
            raise Exception('No tasks provided in predict()')
        if not tasks[0]['data'].get('question'):
            raise Exception('No "question" provided in task data')
        if len(tasks) > 1:
            raise Exception('Only one task in predict() is supported')

        # get the question from task
        question = tasks[0]['data'][QUESTION_TEXT_TAG]

        # get the rephrased questions from user textarea
        prediction = {"result": []}
        texts = self.get_rephrased_questions(context)
        id_ = 0
        if not texts:
            texts = self.generate_reprhased_questions(question)
            for text in texts:
                prediction['result'].append({
                    "id": str(id_),
                    "type": "textarea",
                    "value": {"text": [text]},
                    "origin": "prediction",
                    "to_name": "question",
                    "from_name": "repharesed"
                })
        
        return ModelResponse(predictions=[prediction])

    def get_rephrased_questions(self, context):
        texts = []
        if context:
            # Interactive mode - get prompt from context
            result = context.get('result', [])
            for item in result:
                if item.get('name') == REPHRASER_TEXTAREA_TAG:
                    texts.append(item['value']['text'])
        return texts

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

