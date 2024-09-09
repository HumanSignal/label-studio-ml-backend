import os
import pathlib
import re
import label_studio_sdk
import logging

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from transformers import pipeline, Pipeline
from itertools import groupby
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, ClassLabel, Value, Sequence, Features
from functools import partial

logger = logging.getLogger(__name__)
_model: Optional[Pipeline] = None
MODEL_DIR = os.getenv('MODEL_DIR', './results')
BASELINE_MODEL_NAME = os.getenv('BASELINE_MODEL_NAME', 'dslim/bert-base-NER')
FINETUNED_MODEL_NAME = os.getenv('FINETUNED_MODEL_NAME', 'finetuned_model')


def reload_model():
    global _model
    _model = None
    try:
        chk_path = str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME)
        logger.info(f"Loading finetuned model from {chk_path}")
        _model = pipeline("ner", model=chk_path, tokenizer=chk_path)
    except:
        # if finetuned model is not available, use the baseline model with the original labels
        logger.info(f"Loading baseline model {BASELINE_MODEL_NAME}")
        _model = pipeline("ner", model=BASELINE_MODEL_NAME, tokenizer=BASELINE_MODEL_NAME)


reload_model()


class HuggingFaceNER(LabelStudioMLBase):
    """Custom ML Backend model
    """
    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-3))
    NUM_TRAIN_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 10))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 0.01))

    def get_labels(self):
        li = self.label_interface
        from_name, _, _ = li.get_first_tag_occurence('Labels', 'Text')
        tag = li.get_tag(from_name)
        return tag.labels
    
    def setup(self):
        """Configure any paramaters of your model here
        """
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        li = self.label_interface
        from_name, to_name, value = li.get_first_tag_occurence('Labels', 'Text')
        texts = [self.preload_task_data(task, task['data'][value]) for task in tasks]

        # run predictions
        model_predictions = _model(texts)

        predictions = []
        for prediction in model_predictions:
            # prediction returned in the format: [{'entity': 'B-ORG', 'score': 0.999, 'index': 1, 'start': 0, 'end': 7, 'word': 'Google'}, ...]
            # we need to group them by 'B-' and 'I-' prefixes to form entities
            results = []
            avg_score = 0
            for label, group in groupby(prediction, key=lambda x: re.sub(r'^[BI]-', '', x['entity'])):
                entities = list(group)
                start = entities[0]['start']
                end = entities[-1]['end']
                score = float(sum([entity['score'] for entity in entities]) / len(entities))
                results.append({
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'labels',
                    'value': {
                        'start': start,
                        'end': end,
                        'labels': [label]
                    },
                    'score': score
                })
                avg_score += score
            if results:
                predictions.append({
                    'result': results,
                    'score': avg_score / len(results),
                    'model_version': self.get('model_version')
                })
        
        return ModelResponse(predictions=predictions, model_version=self.get('model_version'))

    def _get_tasks(self, project_id):
        # download annotated tasks from Label Studio
        ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project_id)
        tasks = project.get_labeled_tasks()
        return tasks

    def tokenize_and_align_labels(self, examples, tokenizer):
        """
        From example https://huggingface.co/docs/transformers/en/tasks/token_classification#preprocess
        """
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def fit(self, event, data, **kwargs):
        """Download dataset from Label Studio and prepare data for training in BERT
        """
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            logger.info(f"Skip training: event {event} is not supported")
            return

        project_id = data['annotation']['project']
        tasks = self._get_tasks(project_id)

        if len(tasks) % self.START_TRAINING_EACH_N_UPDATES != 0 and event != 'START_TRAINING':
            logger.info(f"Skip training: {len(tasks)} tasks are not multiple of {self.START_TRAINING_EACH_N_UPDATES}")
            return

        # we need to convert Label Studio NER annotations to hugingface NER format in datasets
        # for example:
        # {'id': '0',
        #  'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
        #  'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
        # }
        ds_raw = []
        from_name, to_name, value = self.label_interface.get_first_tag_occurence('Labels', 'Text')
        tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME)

        no_label = 'O'
        label_to_id = {no_label: 0}
        for task in tasks:
            for annotation in task['annotations']:
                if not annotation.get('result'):
                    continue
                spans = [{'label': r['value']['labels'][0], 'start': r['value']['start'], 'end': r['value']['end']} for r in annotation['result']]
                spans = sorted(spans, key=lambda x: x['start'])
                text = self.preload_task_data(task, task['data'][value])

                # insert tokenizer.pad_token to the unlabeled chunks of the text in-between the labeled spans, as well as to the beginning and end of the text
                last_end = 0
                all_spans = []
                for span in spans:
                    if last_end < span['start']:
                        all_spans.append({'label': no_label, 'start': last_end, 'end': span['start']})
                    all_spans.append(span)
                    last_end = span['end']
                if last_end < len(text):
                    all_spans.append({'label': no_label, 'start': last_end, 'end': len(text)})

                # now tokenize chunks separately and add them to the dataset
                item = {'id': task['id'], 'tokens': [], 'ner_tags': []}
                for span in all_spans:
                    tokens = tokenizer.tokenize(text[span['start']:span['end']])
                    item['tokens'].extend(tokens)
                    if span['label'] == no_label:
                        item['ner_tags'].extend([label_to_id[no_label]] * len(tokens))
                    else:
                        label = 'B-' + span['label']
                        if label not in label_to_id:
                            label_to_id[label] = len(label_to_id)
                        item['ner_tags'].append(label_to_id[label])
                        if len(tokens) > 1:
                            label = 'I-' + span['label']
                            if label not in label_to_id:
                                label_to_id[label] = len(label_to_id)
                            item['ner_tags'].extend([label_to_id[label] for _ in range(1, len(tokens))])
                ds_raw.append(item)

        logger.debug(f"Dataset: {ds_raw}")
        # convert to huggingface dataset
        # Define the features of your dataset
        features = Features({
            'id': Value('string'),
            'tokens': Sequence(Value('string')),
            'ner_tags': Sequence(ClassLabel(names=list(label_to_id.keys())))
        })
        hf_dataset = Dataset.from_list(ds_raw, features=features)
        tokenized_dataset = hf_dataset.map(partial(self.tokenize_and_align_labels, tokenizer=tokenizer), batched=True)

        logger.debug(f"HF Dataset: {tokenized_dataset}")

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        id_to_label = {i: label for label, i in label_to_id.items()}
        logger.debug(f"Labels: {id_to_label}")

        model = AutoModelForTokenClassification.from_pretrained(
            BASELINE_MODEL_NAME, num_labels=len(id_to_label),
            id2label=id_to_label, label2id=label_to_id)
        logger.debug(f"Model: {model}")

        training_args = TrainingArguments(
            output_dir=str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME),
            learning_rate=self.LEARNING_RATE,
            per_device_train_batch_size=8,
            num_train_epochs=self.NUM_TRAIN_EPOCHS,
            weight_decay=self.WEIGHT_DECAY,
            evaluation_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

        chk_path = str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME)
        logger.info(f"Model is trained and saved as {chk_path}")
        trainer.save_model(chk_path)

        # reload model
        # TODO: this is not thread-safe, should be done with critical section
        reload_model()
