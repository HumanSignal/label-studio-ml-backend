import os
import torch
import logging
import pathlib
import label_studio_sdk

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import pipeline
from label_studio_sdk.label_interface.objects import PredictionValue
from label_studio_ml.response import ModelResponse
from datasets import Dataset

logger = logging.getLogger(__name__)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


class BertClassifier(LabelStudioMLBase):
    """
    BERT-based text classification model for Label Studio

    This model uses the Hugging Face Transformers library to fine-tune a BERT model for text classification.
    Use any model for [AutoModelForSequenceClassification](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelforsequenceclassification)
    The model is trained on the labeled data from Label Studio and then used to make predictions on new data.

    Parameters:
    -----------
    LABEL_STUDIO_HOST : str
        The URL of the Label Studio instance
    LABEL_STUDIO_API_KEY : str
        The API key for the Label Studio instance
    START_TRAINING_EACH_N_UPDATES : int
        The number of labeled tasks to download from Label Studio before starting training
    LEARNING_RATE : float
        The learning rate for the model training
    NUM_TRAIN_EPOCHS : int
        The number of epochs for model training
    WEIGHT_DECAY : float
        The weight decay for the model training
    baseline_model_name : str
        The name of the baseline model to use for training
    MODEL_DIR : str
        The directory to save the trained model
    finetuned_model_name : str
        The name of the finetuned model
    """
    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
    NUM_TRAIN_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 3))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 0.01))
    baseline_model_name = os.getenv('BASELINE_MODEL_NAME', 'bert-base-multilingual-cased')
    MODEL_DIR = os.getenv('MODEL_DIR', './results')
    finetuned_model_name = os.getenv('FINETUNED_MODEL_NAME', 'finetuned-model')
    _model = None

    def get_labels(self):
        li = self.label_interface
        from_name, _, _ = li.get_first_tag_occurence('Choices', 'Text')
        tag = li.get_tag(from_name)
        return tag.labels

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def _lazy_init(self):
        if not self._model:
            try:
                chk_path = str(pathlib.Path(self.MODEL_DIR) / self.finetuned_model_name)
                self._model = pipeline("text-classification", model=chk_path, tokenizer=chk_path)
            except:
                # if finetuned model is not available, use the baseline model, with the labels from the label_interface
                self._model = pipeline(
                    "text-classification",
                    model=self.baseline_model_name,
                    tokenizer=self.baseline_model_name)

                labels = self.get_labels()
                self._model.model.config.id2label = {i: label for i, label in enumerate(labels)}
                self._model.model.config.label2id = {label: i for i, label in enumerate(labels)}

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """

        # TODO: this may result in single-time timeout for large models - consider adjusting the timeout on Label Studio side
        self._lazy_init()

        li = self.label_interface
        from_name, to_name, value = li.get_first_tag_occurence('Choices', 'Text')
        texts = [self.preload_task_data(task, task['data'][value]) for task in tasks]

        model_predictions = self._model(texts)
        predictions = []
        for prediction in model_predictions:
            logger.debug(f"Prediction: {prediction}")
            region = li.get_tag(from_name).label(prediction['label'])
            pv = PredictionValue(
                score=prediction['score'],
                result=[region],
                model_version=self.get('model_version')
            )
            predictions.append(pv)

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **additional_params):
        """Download dataset from Label Studio and prepare data for training in BERT"""
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            logger.info(f"Skip training: event {event} is not supported")
            return
        project_id = data['annotation']['project']

        # dowload annotated tasks from Label Studio
        ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project_id)
        tasks = project.get_labeled_tasks()

        logger.info(f"Downloaded {len(tasks)} labeled tasks from Label Studio")
        logger.debug(f"Tasks: {tasks}")
        if len(tasks) % self.START_TRAINING_EACH_N_UPDATES != 0 and event != 'START_TRAINING':
            # skip training if the number of tasks is not divisible by START_TRAINING_EACH_N_UPDATES
            logger.info(f"Skip training: the number of tasks is not divisible by {self.START_TRAINING_EACH_N_UPDATES}")
            return

        from_name, to_name, value = self.label_interface.get_first_tag_occurence('Choices', 'Text')

        ds_raw = {
            'id': [],
            'text': [],
            'label': []
        }
        for task in tasks:
            for annotation in task['annotations']:
                if 'result' in annotation:
                    for result in annotation['result']:
                        if 'choices' in result['value']:
                            ds_raw['id'].append(task['id'])
                            text = self.preload_task_data(task, task['data'][value])
                            ds_raw['text'].append(text)
                            ds_raw['label'].append(result['value']['choices'])

        hf_dataset = Dataset.from_dict(ds_raw)
        logger.debug(f"Dataset: {hf_dataset}")

        labels = self.get_labels()
        label_to_id = {label: i for i, label in enumerate(labels)}
        id_to_label = {i: label for i, label in enumerate(labels)}
        logger.debug(f"Labels: {labels}")

        # Preprocess the dataset
        tokenizer = AutoTokenizer.from_pretrained(self.baseline_model_name)

        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True)

        tokenized_datasets = hf_dataset.map(preprocess_function, batched=True)
        logger.debug(f"Tokenized dataset: {tokenized_datasets}")

        # Convert labels to ids
        def label_to_id_function(examples):
            examples["label"] = [label_to_id[label] for label in examples["label"]]
            return examples

        tokenized_datasets = tokenized_datasets.map(label_to_id_function)

        # Load model with custom config
        logger.info(f"Start training the model {self.finetuned_model_name}")
        config = AutoConfig.from_pretrained(self.baseline_model_name, num_labels=len(labels))
        logger.debug(f"Config: {config}")
        model = AutoModelForSequenceClassification.from_pretrained(self.baseline_model_name, config=config)
        model.config.id2label = id_to_label
        model.config.label2id = label_to_id
        logger.debug(f"Model: {model}")

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(pathlib.Path(self.MODEL_DIR) / 'training_output'),
            learning_rate=2e-5,
            evaluation_strategy="no",
            num_train_epochs=3,
            weight_decay=0.01,
            log_level='info'
        )
        logger.debug(f"Training arguments: {training_args}")

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            tokenizer=tokenizer,
        )
        logger.debug(f"Trainer: {trainer}")

        # Train the model
        trainer.train()

        chk_path = str(pathlib.Path(self.MODEL_DIR) / self.finetuned_model_name)
        logger.info(f"Model is trained and saved as {chk_path}")
        trainer.save_model(chk_path)
