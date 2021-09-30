import pickle
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from label_studio_ml.model import LabelStudioMLBase
from label_studio.core.settings.base import DATA_UNDEFINED_NAME
from label_studio.core.label_config import parse_config
from flask import request, Response
import logging

logger = logging.getLogger(__name__)

class SimpleTextClassifier(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(SimpleTextClassifier, self).__init__(**kwargs)
        if not self.train_output:
            logger.info('Make initial model..')
            # If there is no trainings, define cold-started the simple TF-IDF text classifier
            self.reset_model()
            # make some dummy initialization
            self.model.fit(X=[], y=[])
        else:
            logger.info(f'Found output from the previous train run: {self.train_output}. Initialize model from there..')
            # otherwise load the model from the latest training results
            self.model_file = self.train_output['model_file']
            with open(self.model_file, mode='rb') as f:
                self.model = pickle.load(f)

    def reset_model(self):
        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 3)), LogisticRegression(C=10, verbose=True))

    def predict(self, tasks, **kwargs):
        # collect input texts
        logger.info(f'Predicting {len(tasks)} task(s)')
        input_texts = []
        for task in tasks:
            input_text = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
            input_texts.append(input_text)

        # get model predictions
        probabilities = self.model.predict_proba(input_texts)
        predicted_label_indices = np.argmax(probabilities, axis=1)
        predicted_scores = probabilities[np.arange(len(predicted_label_indices)), predicted_label_indices]
        predictions = []
        for idx, score in zip(predicted_label_indices, predicted_scores):
            predicted_label = self.labels[idx]
            # prediction result for the single task
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': self.result_type,
                'value': {self.result_type: [predicted_label]}
            }]

            # expand predictions with their scores for all tasks
            predictions.append({'result': result, 'score': score})

        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        # loading data
        if os.path.exists("ls_data.dat"):
            with open("ls_data.dat", mode='rb') as f:
                load_data = pickle.load(f)
        else:
            load_data = {}
        all_data = request.data
        # add task data to list
        if all_data["action"] in ["TASKS_CREATED", "TASKS_UPDATED"]:
            logger.info(f'Received webhook {all_data["action"]}: collecting {len(all_data["tasks"])} tasks..')
            for task in all_data['tasks']:
                if not task['id'] in load_data:
                    load_data[task['id']] = {}
                load_data[task['id']]['data'] = task['data']
        # add annotations to tasks
        if all_data["action"] in ["ANNOTATION_UPDATED", "ANNOTATION_CREATED"]:
            if not all_data["annotation"]["task"] in load_data:
                load_data[all_data["annotation"]["task"]]['annotations'] = []
            logger.info(f'Received webhook {all_data["action"]}: add 1 annotations to {len(load_data[all_data["annotation"]["task"]]['annotations'])..')
            load_data[all_data["annotation"]["task"]]['annotations'].append(all_data["annotation"])
        # saving data
        with open("ls_data.dat", mode='wb') as f:
            pickle.dump(load_data, f)
        # return if total annotations count less
        if not all_data['total_annotations_number'] % 10 == 0:
            logger.info(f"Annotations number {all_data['total_annotations_number']}. Train start on 10.")
            return

        logger.info(f"Start training process!")
        # loading config and meta data from it
        self.parsed_label_config = parse_config(all_data['project']['label_config'])
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        self.labels = self.info['labels']
        self.value = self.info['inputs'][0]['value']
        self.to_name = self.info['to_name'][0]

        input_texts = []
        output_labels, output_labels_idx = [], []
        label2idx = {l: i for i, l in enumerate(self.labels)}
        for task_id, completion in load_data.items():
            # get input text from task data
            print(completion)
            input_text = completion['data'].get(self.value) or completion['data'].values()[0]
            # get an annotation
            for annotation in completion['annotations']:
                for result in annotation['result']:
                    for output_label in result['value'][result['type']]:
                        input_texts.append(input_text)
                        self.result_type = result['type']
                        output_labels.append(output_label)
                        output_label_idx = label2idx[output_label]
                        output_labels_idx.append(output_label_idx)

        new_labels = set(output_labels)
        if len(new_labels) != len(self.labels):
            self.labels = list(sorted(new_labels))
            print('Label set has been changed:' + str(self.labels))
            label2idx = {l: i for i, l in enumerate(self.labels)}
            output_labels_idx = [label2idx[label] for label in output_labels]

        # train the model
        self.reset_model()
        self.model.fit(input_texts, output_labels_idx)

        # save output resources
        model_file = os.path.join(workdir, 'model.pkl')
        logger.info(f'Save model to {model_file}..')
        with open(model_file, mode='wb') as fout:
            pickle.dump(self.model, fout)

        train_output = {
            'labels': self.labels,
            'model_file': model_file
        }
        return train_output
