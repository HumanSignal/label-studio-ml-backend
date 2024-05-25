import os
import label_studio_sdk
import logging
import pickle
import numpy as np

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, Pipeline

logger = logging.getLogger(__name__)

_model: Optional[Pipeline] = None


class SklearnTextClassifier(LabelStudioMLBase):
    """Custom ML Backend model
    """

    # define model parameters
    # C is the inverse regularization strength for Logistic Regression
    LOGISTIC_REGRESSION_C = float(os.getenv("LOGISTIC_REGRESSION_C", 10))
    # Label Studio host - to be used for training
    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    # Label Studio API key - to be used for training
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
    # Start training each N updates
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    MODEL_DIR = os.getenv('MODEL_DIR', '.')

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def get_model(self, blank=False):
        global _model
        # Lazy initialization of the model
        # If the model is not already initialized, it is initialized here
        if _model is not None:
            logger.debug('Model is already initialized')
            return _model

        model_path = os.path.join(self.MODEL_DIR, 'model.pkl')
        if not os.path.exists(model_path) or blank:
            _model = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 3), token_pattern=r"(?u)\b\w\w+\b|\w"),
                LogisticRegression(C=self.LOGISTIC_REGRESSION_C, verbose=True)
            )
            config = self.get_label_studio_parameters()
            logger.info(f'Creating a new model using labels: {config["labels"]}')
            _model.fit(X=config['labels'], y=list(range(len(config['labels']))))
            logger.debug('Created a new model with labels: %s', config['labels'])
        else:
            logger.info(f'Loading model from {model_path}')
            with open(model_path, 'rb') as f:
                _model = pickle.load(f)

        return _model

    def get_label_studio_parameters(self) -> Dict:
        # Expect labeling config to have only one output of <Choices> type and one input of <Text> type
        # The first occurrence of the 'Choices' and 'Text' tags in the labeling config is retrieved
        from_name, to_name, value = self.label_interface.get_first_tag_occurence('Choices', 'Text')

        # Get labels from the labeling config
        # The labels are sorted for consistent ordering
        labels = sorted(self.label_interface.get_tag(from_name).labels)
        return {
            'from_name': from_name,
            'to_name': to_name,
            'value': value,
            'labels': labels
        }
        
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        This method is used to predict the labels for a given list of tasks.

        Parameters:
        tasks (List[Dict]): A list of tasks. Each task is a dictionary that contains the data to be labeled.
        context (Optional[Dict]): An optional dictionary that can contain additional information for the prediction. See [interactive labeling](https://labelstud.io/guide/ml#Get-interactive-pre-annotations) for more information.

        Returns:
        ModelResponse: A ModelResponse object that contains the predictions for each task.

        """
        model = self.get_model()
        config = self.get_label_studio_parameters()

        # Collect input texts
        # The input texts are extracted from the tasks and stored in a list
        input_texts = []
        for task in tasks:
            value = task['data'].get(config['value']) or task['data'].get(DATA_UNDEFINED_NAME)
            input_text = self.preload_task_data(task, value)
            input_texts.append(input_text)

        # Get model predictions
        # The model's predict_proba method is used to get the probabilities of each label for each task
        probabilities = model.predict_proba(input_texts)

        # The label with the highest probability is selected as the predicted label for each task
        predicted_label_indices = np.argmax(probabilities, axis=1)
        predicted_scores = probabilities[np.arange(len(predicted_label_indices)), predicted_label_indices]
        logger.debug(
            f'Probabilities: {probabilities}, '
            f'predicted_label_indices: {predicted_label_indices}, '
            f'predicted_scores: {predicted_scores}')

        # The predictions are stored in a list
        predictions = []
        for idx, score in zip(predicted_label_indices, predicted_scores):
            predicted_label = config['labels'][idx]

            # Prediction result for the single task
            result = [{
                'from_name': config['from_name'],
                'to_name': config['to_name'],
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]

            # Expand predictions with their scores for all tasks
            predictions.append({
                'result': result,
                'score': score,
                'model_version': self.get('model_version')
            })

        # Return predictions
        # The predictions are returned as a ModelResponse object
        return ModelResponse(predictions=predictions, model_version=self.get('model_version'))

    def _get_tasks(self, project_id: int) -> List[Dict]:
        """
        Get tasks from Label Studio API

        Parameters:
            project_id (str): The ID of the project

        Returns:
            List[Dict]: A list of tasks
        """
        ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project_id)
        tasks = project.get_labeled_tasks()
        return tasks

    def fit(self, event, data, **kwargs):
        """
        This method is used to fit the Logistic Regression model to the labeled text collected from Label Studio.
        It saves the model to a MODEL_DIR/model.pkl file.

        Parameters:
            event (str): The event that triggered the fitting of the model (e.g., 'ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
            data (Dict): The data that is used to fit the model.
        """
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            logger.info(f"Skip training: event {event} is not supported")
            return

        project_id = data['annotation']['project']
        tasks = self._get_tasks(project_id)

        # Get the labeling configuration parameters like labels and input / output annotation format names
        config = self.get_label_studio_parameters()

        if len(tasks) % self.START_TRAINING_EACH_N_UPDATES != 0 and event != 'START_TRAINING':
            logger.info(
                f'Not starting training, {len(tasks)} '
                f'tasks are not multiple of {self.START_TRAINING_EACH_N_UPDATES}'
            )
            return

        input_texts = []
        output_labels, output_labels_idx = [], []
        label2idx = {l: i for i, l in enumerate(config['labels'])}

        for task in tasks:
            for annotation in task['annotations']:
                if not annotation.get('result') or annotation.get('skipped') or annotation.get('was_cancelled'):
                    continue

                # collect input texts
                value = task['data'].get(config['value']) or task['data'].get(DATA_UNDEFINED_NAME)
                input_text = self.preload_task_data(task, value)
                input_texts.append(input_text)

                # collect output labels
                output_label = annotation['result'][0]['value']['choices'][0]
                output_labels.append(output_label)
                output_label_idx = label2idx[output_label]
                output_labels_idx.append(output_label_idx)

        # fit the model
        model = self.get_model(blank=True)
        model.fit(input_texts, output_labels_idx)

        # save the model
        model_path = os.path.join(self.MODEL_DIR, f'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        # TODO: not thread safe
        global _model
        _model = None
        self.get_model()
