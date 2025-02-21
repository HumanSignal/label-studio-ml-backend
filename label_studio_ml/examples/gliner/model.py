import logging
import os
from math import floor
from typing import List, Dict, Optional

import label_studio_sdk
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments
from label_studio_sdk.label_interface.objects import PredictionValue

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)

GLINER_MODEL_NAME = os.getenv("GLINER_MODEL_NAME", "urchade/gliner_medium-v2.1")
logger.info(f"Loading GLINER model {GLINER_MODEL_NAME}")
MODEL = GLiNER.from_pretrained(GLINER_MODEL_NAME)


class GLiNERModel(LabelStudioMLBase):
    """
    Custom ML Backend for GILNER model
    """

    def setup(self):
        """Configure any parameters of your model here
        """
        self.LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_URL', 'http://localhost:8080')
        self.LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')

        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')
        self.threshold = float(os.getenv('THRESHOLD', 0.5))
        self.model = MODEL

    def convert_to_ls_annotation(self, prediction, from_name, to_name):
        """
        Convert from GLiNER output format to Label Studio annotastion format
        :param prediction: The prediction output from GLiNER
        :param from_name
        :param to_name
        """
        results = []
        sent_preds = []
        for ent in prediction:
            label = [ent['label']]
            if label:
                score = ent['score']
                sent_preds.append({
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'labels',
                    "value": {
                        "start": ent['start'],
                        "end": ent['end'],
                        "text": ent['text'],
                        "labels": label
                    },
                    "score": round(score, 4)
                })

        # add minimum of certaincy scores of entities in sentence for active learning use
        score = min([p['score'] for p in sent_preds]) if sent_preds else 2.0
        results.append(PredictionValue(
            result=sent_preds,
            score=score,
            model_version=self.get('model_version')
        ))

        return results

    def convert_char_to_token_span(self, text: List, start: int, end: int):
        """
        A helper function to convert character spans to token spans
        text: a list of the tokenized text
        :param start: the first character of the span, as an int
        end: the last character of the span, as an int
        returns: the first and last tokens of the spans, as ints
        """
        start_token = None
        end_token = None
        total_char = 0
        for i, word in enumerate(text):
            if total_char >= start and not start_token:
                start_token = i
            if total_char >= end and not end_token:
                end_token = i
            total_char += (len(word) + 1)
        if not end_token:
            end_token = len(text)
        return start_token, end_token

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ inference logic
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        # make predictions with currently set model
        from_name, to_name, value = self.label_interface.get_first_tag_occurence('Labels', 'Text')

        # get labels from the labeling configuration
        labels = sorted(self.label_interface.get_tag(from_name).labels)

        texts = [task['data'][value] for task in tasks]
        predictions = []
        for text in texts:
            entities = self.model.predict_entities(text, labels, threshold=self.threshold)
            pred = self.convert_to_ls_annotation(entities, from_name, to_name)
            predictions.extend(pred)

        return ModelResponse(predictions=predictions)

    def process_training_data(self, task):
        """
        Process the task from Label Studio export to isolate the information needed for prediction.
        We need the tokenized text of the input, along with the start and end indicies, by word, of the annotated spans
        :param task: the task as output by Label Studio
        """
        # We get the list of tokens from the original data sample we uploaded
        tokens = task['data']['tokens']
        ner = []
        # Parse the annotations
        for annotation in task['annotations']:
            for result in annotation['result']:
                start = result['value']['start']
                end = result['value']['end']
                start_token, end_token = self.convert_char_to_token_span(tokens, start, end)
                label = result['value']['labels'][0]
                ner.append([start_token, end_token, label])
        return tokens, ner

    def train(self, model, training_args, train_data, eval_data=None):
        """
        retrain the GLiNER model. Code adapted from the GLiNER finetuning notebook.
        :param model: the model to train
        :param config: the config object for training parameters
        :param train_data: the training data, as a list of dictionaries
        :param eval_data: the eval data
        """
        logger.info("Training Model")
        if training_args.use_cpu == True:
            model = model.to('cpu')
        else:
            model = model.to("cuda")

        data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=model.data_processor.transformer_tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """
        # we only train the model if the "start training" button is pressed from settings.
        if event == "START_TRAINING":
            logger.info("Fitting model")

            # download annotated tasks from Label Studio
            ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
            project = ls.get_project(id=self.project_id)
            tasks = project.get_labeled_tasks()

            logger.info(f"Downloaded {len(tasks)} labeled tasks from Label Studio")

            training_data = []
            for task in tasks:
                tokens, ner = self.process_training_data(task)
                training_data.append({"tokenized_text": tokens, "ner": ner})

            from_name, to_name, value = self.label_interface.get_first_tag_occurence('Labels', 'Text')
            eval_data = {
                "entity_types": sorted(self.label_interface.get_tag(from_name).labels),
                "samples": training_data[:10]
            }

            training_data = training_data[10:]
            logger.debug(training_data)

            # Define the hyperparameters in a config variable
            # This comes from the pretraining example in the GLiNER repo
            num_steps = 10
            batch_size = 1
            data_size = len(training_data)
            num_batches = floor(data_size / batch_size)
            num_epochs = max(1, floor(num_steps / num_batches))

            training_args = TrainingArguments(
                output_dir="models",
                learning_rate=5e-6,
                weight_decay=0.01,
                others_lr=1e-5,
                others_weight_decay=0.01,
                lr_scheduler_type="linear",  # cosine
                warmup_ratio=0.1,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                focal_loss_alpha=0.75,
                focal_loss_gamma=2,
                num_train_epochs=num_epochs,
                evaluation_strategy="steps",
                save_steps=100,
                save_total_limit=10,
                dataloader_num_workers=0,
                use_cpu=True,
                report_to="none",
            )

            self.train(self.model, training_args, training_data, eval_data)

            logger.info("Saving new fine-tuned model as the default model")
            self.model = GLiNER.from_pretrained(f"models/checkpoint-10", local_files_only=True)
            model_version = int(self.model_version[-1]) + 1
            self.set("model_version", f'{self.__class__.__name__}-v{model_version}')
        else:
            logger.info("Model training not triggered")
