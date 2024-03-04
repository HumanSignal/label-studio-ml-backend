import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader


def pad_sequences(input_ids, maxlen):
    padded_ids = []
    for ids in input_ids:
        nonpad = min(len(ids), maxlen)
        pids = [ids[i] for i in range(nonpad)]
        for i in range(nonpad, maxlen):
            pids.append(0)
        padded_ids.append(pids)
    return padded_ids


def prepare_texts(texts, tokenizer, maxlen, sampler_class, batch_size, choices_ids=None):
    # create input token indices
    input_ids = []
    for text in texts:
        input_ids.append(tokenizer.encode(text, add_special_tokens=True))
    # input_ids = pad_sequences(input_ids, maxlen=maxlen, dtype='long', value=0, truncating='post', padding='post')
    input_ids = pad_sequences(input_ids, maxlen)
    # Create attention masks
    attention_masks = []
    for sent in input_ids:
        attention_masks.append([int(token_id > 0) for token_id in sent])

    if choices_ids is not None:
        dataset = TensorDataset(torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long), torch.tensor(choices_ids, dtype=torch.long))
    else:
        dataset = TensorDataset(torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long))
    sampler = sampler_class(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def calc_slope(y):
    n = len(y)
    if n == 1:
        raise ValueError('Can\'t compute slope for array of length=1')
    x_mean = (n + 1) / 2
    x2_mean = (n + 1) * (2 * n + 1) / 6
    xy_mean = np.average(y, weights=np.arange(1, n + 1))
    y_mean = np.mean(y)
    slope = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean * x_mean)
    return slope