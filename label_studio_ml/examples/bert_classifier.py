import torch
from typing import List, Dict, Union 

import transformers
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from label_studio_ml.model import LabelStudioMLBase
from transformers.utils.dummy_pt_objects import AutoModel

from label_studio_ml.utils import prepare_texts, compute_metrics


if torch.cuda.is_available():
    device = "cuda"
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


class TransformersClassifier(LabelStudioMLBase):

    def __init__(
        self, pretrained_model: str= "bert-base-multilingual-cased", maxlen: int=64,
        batch_size: int=32, num_epochs: int=1, logging_steps: int=1, train_logs: int=None, **kwargs
    ):
        super(TransformersClassifier, self).__init__(**kwargs)
        self.pretrained_model = pretrained_model
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.train_logs = train_logs

        # then collect all keys from config which will be used to extract data from task and to form prediction
        # Parsed label config contains only one output of <Choices> type
        assert len(self.parsed_label_config) == 1
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        assert self.info['type'] == 'Choices'

        # the model has only one textual input
        assert len(self.info['to_name']) == 1
        assert len(self.info['inputs']) == 1
        assert self.info['inputs'][0]['type'] == 'Text'

        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']

        if not self.train_output:
            self.labels = self.info['labels']
            self.reset_model(pretrained_model = self.pretrained_model, cache_dir=None, device= device)
            print('Initialized with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
            ))
        else:
            self.load(self.train_output)
            print('Loaded from train output with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
            ))

    def reset_model(self, pretrained_model: str, cache_dir: str, device: str) -> AutoModelForSequenceClassification:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=len(self.labels),
            output_attentions=False,
            output_hidden_states=False,
            cache_dir=cache_dir
        )
        model.to(device)
        return model

    def load(self, train_output) -> None:
        pretrained_model = train_output['model_path']
        self.tokenizer = AutoModel.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
        self.model.to(device)
        self.model.eval()
        self.batch_size = train_output['batch_size']
        self.labels = train_output['labels']
        self.maxlen = train_output['maxlen']

    @property
    def not_trained(self) -> bool:
        return not hasattr(self, 'tokenizer')

    def predict(self, tasks, **kwargs) -> Union[List[str],List[Dict[str,str]]]:
        if self.not_trained:
            print('Can\'t get prediction because model is not trained yet.')
            return []

        # Retrieve text 
        texts = [task['data'][self.value] for task in tasks]

        # Create pipeline for prediction
        TCPipeline = TextClassificationPipeline(
            model = self.model,
            tokenizer = self.tokenizer
        )

        # Get outputs from pipeline
        outputs = TCPipeline(texts, function_to_apply = 'sigmoid')

        pred_labels, pred_scores = [], []
        for output in outputs:

            pred_labels.extend(str(output.get('label')))
            pred_scores.extend(float(output.get('score')))

        predictions = []
        for predicted_label, score in zip(pred_labels, pred_scores):
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]

            predictions.append({'result': result, 'score': score})
        return predictions

    def fit(self, completions: List[str], workdir: str=None, cache_dir: str=None, **kwargs) -> Dict[str,Union[str, int]]:
        input_texts = []
        output_labels, output_labels_idx = [], []
        label2idx = {l: i for i, l in enumerate(self.labels)}

        for completion in completions:
            # get input text from task data

            if completion['annotations'][0].get('skipped'):
                continue

            input_text = completion['data'][self.value]
            # List of input text
            input_texts.append(input_text)

            # get an annotation
            output_label = completion['annotations'][0]['result'][0]['value']['choices'][0]
            output_labels.append(output_label)
            output_label_idx = label2idx[output_label]
            output_labels_idx.append(output_label_idx)

        new_labels = set(output_labels)
        added_labels = new_labels - set(self.labels)

        # Check if new labels
        if len(added_labels) > 0:
            print('Label set has been changed. Added ones: ' + str(list(added_labels)))
            self.labels = list(sorted(new_labels))
            label2idx = {l: i for i, l in enumerate(self.labels)}
            output_labels_idx = [label2idx[label] for label in output_labels]

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, cache_dir=cache_dir)

        train_dataloader = prepare_texts(input_texts, tokenizer, self.maxlen, RandomSampler, 1, output_labels_idx)
        model = self.reset_model(self.pretrained_model, cache_dir, device)

        total_steps = len(train_dataloader) * self.num_epochs

        training_arguments = TrainingArguments(
            output_dir= workdir,
            evaluation_strategy = "no",
            # Change value of the parameter is smaller GPU
            per_device_train_batch_size = 16,
            per_device_eval_batch_size = 16,
            logging_strategy = "steps",
            logging_first_step = True,
            logging_steps = 500,

            learning_rate = 2e-5,
            weight_decay = 0.01,
            adam_epsilon = 1e-8,
            lr_scheduler_type = "linear",
            warmup_steps = 0,
            max_steps = total_steps, # Override num_train_epochs
            report_to = "wandb",
            dataloader_drop_last = True
        )

        trainer = Trainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = train_dataloader,
            args = training_arguments
        )

        trainer.train()

        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training  # noqa
        
        model_to_save.save_pretrained(workdir)
        tokenizer.save_pretrained(workdir)

        return {
            'model_path': workdir,
            'batch_size': self.batch_size,
            'maxlen': self.maxlen,
            'pretrained_model': self.pretrained_model,
            'labels': self.labels
        }
