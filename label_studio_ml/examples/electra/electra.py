import torch
import requests
import json
import os

from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments

from label_studio_ml.model import LabelStudioMLBase
from label_studio_tools.core.label_config import parse_config

HOSTNAME = "https://app.heartex.com/"
API_KEY = ""
MODEL_FILE = "my_model"

class ElectraTextClassifier(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(ElectraTextClassifier, self).__init__(**kwargs)
        try:
            self.from_name, self.info = list(self.parsed_label_config.items())[0]
            self.to_name = self.info['to_name'][0]
            self.value = self.info['inputs'][0]['value']
            self.labels = sorted(self.info['labels'])
        except:
            print("Couldn't load label config")

        self.tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")

        if os.path.exists(MODEL_FILE):
            self.model = ElectraForSequenceClassification.from_pretrained(MODEL_FILE)
        else:
            self.model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator")

    def load_config(self, config):
        if not self.parsed_label_config:
            self.parsed_label_config = parse_config(config)
        try:
            self.from_name, self.info = list(self.parsed_label_config.items())[0]
            self.to_name = self.info['to_name'][0]
            self.value = self.info['inputs'][0]['value']
            self.labels = sorted(self.info['labels'])
        except:
            print("Couldn't load label config")

    def predict(self, tasks, **kwargs):
        # get data for prediction from tasks
        final_results = []
        for task in tasks:
            input_texts = ""
            input_text = task['data'].get(self.value)
            if input_text.startswith("http://"):
                input_text = self._get_text_from_s3(input_text)
            input_texts += input_text

            labels = torch.tensor([1], dtype=torch.long)
            # tokenize data
            input_ids = torch.tensor(self.tokenizer.encode(input_texts, add_special_tokens=True)).unsqueeze(0)
            # predict label
            predictions = self.model(input_ids, labels=labels).logits
            predictions = torch.softmax(predictions.flatten(), 0)
            label_count = torch.argmax(predictions).item()
            final_results.append({
                'result': [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'choices',
                    'value': {
                        'choices': [self.labels[label_count]]
                    }
                }],
                "task": task['id'],
                "score": predictions.flatten().tolist()[label_count]
            })
        return final_results

    def fit(self, event, data, workdir=None, **kwargs):
        # check if training is from web hook
        project_id = data['project']['id']
        tasks = self._get_annotated_dataset(project_id)
        if not self.parsed_label_config:
            self.load_config(kwargs['data']['project']['label_config'])
        # Create training params with batch size = 1 as text are different size
        training_args = TrainingArguments("test_trainer", per_device_train_batch_size=1, per_device_eval_batch_size=1)
        # Prepare training data
        input_texts = []
        input_labels = []
        for task in tasks:
            if not task.get('annotations'):
                continue
            input_text = task['data'].get(self.value)
            if input_text.startswith("http://"):
                input_text = self._get_text_from_s3(input_text)
            input_texts.append(torch.flatten(self.tokenizer.encode(input_text, return_tensors="pt")))
            annotation = task['annotations'][0]
            output_label = annotation['result'][0]['value']['choices'][0]
            output_label_idx = self.labels.index(output_label)
            output_label_idx = torch.tensor([[output_label_idx]], dtype=torch.int)
            input_labels.append(output_label_idx)

        print(f"Train dataset length: {len(tasks)}")

        my_dataset = Custom_Dataset((input_texts, input_labels))

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=my_dataset,
            #eval_dataset=small_eval_dataset
        )

        result = trainer.train()

        self.model.save_pretrained(MODEL_FILE)

        train_output = {
            'labels': self.labels,
            'model_file': MODEL_FILE
        }
        return train_output

    def _get_annotated_dataset(self, project_id):
        """Just for demo purposes: retrieve annotated data from Label Studio API"""
        download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export'
        response = requests.get(download_url, headers={'Authorization': f'Token {API_KEY}'})
        return json.loads(response.content)

    def _get_text_from_s3(self, url):
        text = requests.get(url)
        return text.text

class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[0][index], self.dataset[1][index]
        return {"input_ids": example, "label": target}

    def __len__(self):
        return len(self.dataset)
