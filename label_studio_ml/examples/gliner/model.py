import logging
import os
from types import SimpleNamespace
from typing import List, Dict, Optional

import label_studio_sdk
import torch
from gliner import GLiNER
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.objects import PredictionValue
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

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

    def train(self, model, config, train_data, eval_data=None):
        """
        retrain the GLiNER model. Code adapted from the GLiNER finetuning notebook.
        :param model: the model to train
        :param config: the config object for training parameters
        :param train_data: the training data, as a list of dictionaries
        :param eval_data: the eval data
        """
        logger.info("Training Model")
        model = model.to(config.device)

        # Set sampling parameters from config
        model.set_sampling_params(
            max_types=config.max_types,
            shuffle_types=config.shuffle_types,
            random_drop=config.random_drop,
            max_neg_type_ratio=config.max_neg_type_ratio,
            max_len=config.max_len
        )

        model.train()
        train_loader = model.create_dataloader(train_data, batch_size=config.train_batch_size, shuffle=True)
        optimizer = model.get_optimizer(config.lr_encoder, config.lr_others, config.freeze_token_rep)
        pbar = tqdm(range(config.num_steps))
        num_warmup_steps = int(config.num_steps * config.warmup_ratio) if config.warmup_ratio < 1 else int(
            config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, config.num_steps)
        iter_train_loader = iter(train_loader)

        for step in pbar:
            try:
                x = next(iter_train_loader)
            except StopIteration:
                iter_train_loader = iter(train_loader)
                x = next(iter_train_loader)

            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(config.device)

            try:
                loss = model(x)  # Forward pass
            except RuntimeError as e:
                print(f"Error during forward pass at step {step}: {e}")
                print(f"x: {x}")
                continue

            if torch.isnan(loss):
                print("Loss is NaN, skipping...")
                continue

            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()  # Reset gradients

            description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
            pbar.set_description(description)

            if (step + 1) % config.eval_every == 0:
                model.eval()
                if eval_data:
                    results, f1 = model.evaluate(eval_data["samples"], flat_ner=True, threshold=0.5, batch_size=12,
                                                 entity_types=eval_data["entity_types"])
                    print(f"Step={step}\n{results}")

                if not os.path.exists(config.save_directory):
                    os.makedirs(config.save_directory)

                model.save_pretrained(f"{config.save_directory}/finetuned_{step}")
                model.train()


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

            eval_data = {
                "entity_types": sorted(self.label_interface.get_tag(from_name).labels)
,
                "samples": training_data[:10]
            }

            training_data = training_data[10:]
            logger.debug(training_data)

            # Define the hyperparameters in a config variable
            # This comes from the pretraining example in the GLiNER repo
            config = SimpleNamespace(
                num_steps=10000,  # number of training iteration
                train_batch_size=2,
                eval_every=1000,  # evaluation/saving steps
                save_directory="logs",  # where to save checkpoints
                warmup_ratio=0.1,  # warmup steps
                device='cpu',
                lr_encoder=1e-5,  # learning rate for the backbone
                lr_others=5e-5,  # learning rate for other parameters
                freeze_token_rep=False,  # freeze of not the backbone

                # Parameters for set_sampling_params
                max_types=25,  # maximum number of entity types during training
                shuffle_types=True,  # if shuffle or not entity types
                random_drop=True,  # randomly drop entity types
                max_neg_type_ratio=1,
                # ratio of positive/negative types, 1 mean 50%/50%, 2 mean 33%/66%, 3 mean 25%/75% ...
                max_len=384  # maximum sentence length
            )

            self.train(self.model, config, training_data, eval_data)

            logger.info("Saving new fine-tuned model as the default model")
            self.model = GLiNERModel.from_pretrained("finetuned", local_files_only=True)
            model_version = self.model_version[-1] + 1
            self.set("model_version", f'{self.__class__.__name__}-v{model_version}')
        else:
            logger.info("Model training not triggered")