import torch
import numpy as np
import os

from torch.utils.data import SequentialSampler
from tqdm import tqdm, trange
from collections import deque
from tensorboardX import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from label_studio_ml.model import LabelStudioMLBase

from utils import prepare_texts, calc_slope


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


class BertClassifier(LabelStudioMLBase):

    def __init__(
        self, pretrained_model='bert-base-multilingual-cased', maxlen=64,
        batch_size=32, num_epochs=100, logging_steps=1, train_logs=None, **kwargs
    ):
        super(BertClassifier, self).__init__(**kwargs)
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
            self.model = self.reset_model('bert-base-multilingual-cased', cache_dir=None, device='cpu')
            print('Initialized with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
            ))
        else:
            self.load(self.train_output)
            print('Loaded from train output with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
            ))

    def reset_model(self, pretrained_model, cache_dir, device):
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=len(self.labels),
            output_attentions=False,
            output_hidden_states=False,
            cache_dir=cache_dir
        )
        model.to(device)
        return model

    def load(self, train_output):
        pretrained_model = train_output['model_path']
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.model = BertForSequenceClassification.from_pretrained(pretrained_model)
        self.model.to(device)
        self.model.eval()
        self.batch_size = train_output['batch_size']
        self.labels = train_output['labels']
        self.maxlen = train_output['maxlen']

    @property
    def not_trained(self):
        return not hasattr(self, 'tokenizer')

    def predict(self, tasks, **kwargs):
        if self.not_trained:
            print('Can\'t get prediction because model is not trained yet.')
            return []

        texts = [task['data'][self.value] for task in tasks]
        predict_dataloader = prepare_texts(texts, self.tokenizer, self.maxlen, SequentialSampler, self.batch_size)

        pred_labels, pred_scores = [], []
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1]
            }
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0]

            batch_preds = logits.detach().cpu().numpy()

            argmax_batch_preds = np.argmax(batch_preds, axis=-1)
            pred_labels.extend(str(self.labels[i]) for i in argmax_batch_preds)

            max_batch_preds = np.max(batch_preds, axis=-1)
            pred_scores.extend(float(s) for s in max_batch_preds)

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

    def _get_annotated_dataset(self, project_id):
        raise NotImplementedError('For this model, you need to implement data ingestion pipeline: '
                                  'go to ner.py > _get_annotated_dataset() and put your logic to retrieve'
                                  f'the list of annotated tasks from Label Studio project ID = {project_id}')

    def fit(self, event, data, cache_dir=None, **kwargs):
        input_texts = []
        output_labels, output_labels_idx = [], []
        label2idx = {l: i for i, l in enumerate(self.labels)}
        completions = self._get_annotated_dataset(data['project_id'])
        workdir = os.getenv('WORK_DIR')
        if workdir is None:
            raise ValueError('Specify "WORK_DIR" environmental variable to store model checkpoints.')
        for completion in completions:
            # get input text from task data

            if completion['annotations'][0].get('skipped'):
                continue

            input_text = completion['data'][self.value]
            input_texts.append(input_text)

            # get an annotation
            output_label = completion['annotations'][0]['result'][0]['value']['choices'][0]
            output_labels.append(output_label)
            output_label_idx = label2idx[output_label]
            output_labels_idx.append(output_label_idx)

        new_labels = set(output_labels)
        added_labels = new_labels - set(self.labels)
        if len(added_labels) > 0:
            print('Label set has been changed. Added ones: ' + str(list(added_labels)))
            self.labels = list(sorted(new_labels))
            label2idx = {l: i for i, l in enumerate(self.labels)}
            output_labels_idx = [label2idx[label] for label in output_labels]

        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model, cache_dir=cache_dir)

        train_dataloader = prepare_texts(input_texts, tokenizer, self.maxlen, RandomSampler, self.batch_size, output_labels_idx)
        model = self.reset_model(self.pretrained_model, cache_dir, device)

        total_steps = len(train_dataloader) * self.num_epochs
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        global_step = 0
        total_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(self.num_epochs, desc='Epoch')
        if self.train_logs:
            tb_writer = SummaryWriter(logdir=os.path.join(self.train_logs, os.path.basename(self.output_dir)))
        else:
            tb_writer = None
        loss_queue = deque(maxlen=10)
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iteration')
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[2]}
                outputs = model(**inputs)
                loss = outputs[0]
                loss.backward()
                total_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if global_step % self.logging_steps == 0:
                    last_loss = (total_loss - logging_loss) / self.logging_steps
                    loss_queue.append(last_loss)
                    if tb_writer:
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', last_loss, global_step)
                    logging_loss = total_loss

            # slope-based early stopping
            if len(loss_queue) == loss_queue.maxlen:
                slope = calc_slope(loss_queue)
                if tb_writer:
                    tb_writer.add_scalar('slope', slope, global_step)
                if abs(slope) < 1e-2:
                    break

        if tb_writer:
            tb_writer.close()

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
