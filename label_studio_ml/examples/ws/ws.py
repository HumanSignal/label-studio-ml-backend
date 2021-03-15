import json
import os
import glob
import spacy
import numpy as np
import re
import random

from functools import partial
from collections import defaultdict
from types import SimpleNamespace
from tqdm import tqdm

from nltk import ngrams
from snorkel.labeling.lf.nlp import NLPLabelingFunction
from snorkel.labeling import LFApplier, LFAnalysis, LabelingFunction

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from spacy import displacy


def match(spans_from, spans_to):
    def _iou(x, y):
        if x.get('labels') and y.get('labels') and x['labels'] != y['labels']:
            return 0
        s1, e1 = x['start'], x['end']
        s2, e2 = y['start'], y['end']
        if s2 > e1 or s1 > e2:
            return 0
        intersection = min(e1, e2) - max(s1, s2)
        union = max(e1, e2) - min(s1, s2)
        if union == 0:
            return 0
        iou = intersection / union
        return iou

    bests, argmaxes = [], []
    for span_from in spans_from:
        ious = []
        for span_to in spans_to:
            iou = _iou(span_from, span_to)
            ious.append(iou)
        argmax_iou = np.argmax(ious)
        best_iou = ious[argmax_iou]
        bests.append(best_iou)
        argmaxes.append(argmax_iou)
    return bests, argmaxes


def read_data():
    completions = os.path.expanduser('~/PycharmProjects/label-studio/tmp/gh/completions')
    texts, results = [], []
    for f in glob.glob(f'{completions}/*.json'):
        data = json.load(open(f))
        if data['completions'][0].get('skipped'):
            continue
        texts.append(data['data']['body'])
        results.append([r['value'] for r in data['completions'][0]['result'] if r['type'] == 'labels'])
    return texts, results


def read_data_test():
    tasks_json = os.path.expanduser('~/PycharmProjects/label-studio/tmp/gh/tasks.json')
    data = []
    with open(tasks_json) as f:
        tasks = json.load(f)
        for task in tasks.values():
            data.append(task['data']['body'])
    return data


# stanza.download('en')

nlp_spacy = spacy.load("en_core_web_sm")


class Factor(object):

    def match(self, results):
        out = defaultdict(list)
        for spans, result in zip(self.spans, results):
            scores = match(result, spans)
            for r, s in zip(result, scores):
                out[r['labels'][0]].append(s)
        return {label: float(np.mean(v)) for label, v in out.items()}


class SentenceFactor(Factor):

    def __init__(self, tasks):
        self.spans = []
        for task in tasks:
            text = task
            doc = nlp_spacy(text)
            spans = []
            for sent in doc.sents:
                spans.append({'start': sent.start_char, 'end': sent.end_char, 'text': sent.text})
            self.spans.append(spans)


class NounChunksFactor(Factor):

    def __init__(self, tasks):
        self.spans = []
        for task in tasks:
            text = task
            doc = nlp_spacy(text)
            spans = []
            for chunk in doc.noun_chunks:
                spans.append({'start': chunk.start_char, 'end': chunk.end_char, 'text': chunk.text})
            self.spans.append(spans)


class TokenFactor(Factor):

    def __init__(self, tasks):
        self.spans = []
        for task in tasks:
            text = task
            doc = nlp_spacy(text)
            # displacy.serve(doc, style="dep")
            spans = []
            for tok in doc:
                spans.append({'start': tok.idx, 'end': tok.idx + len(tok.text), 'text': tok.text})
            self.spans.append(spans)


def parse_kw(kw):
    parsed_kw = {}
    text = []

    for i, w in enumerate(kw.split()):
        m = re.match('\[([ABI])\](.*)', w)
        prefix, word = m.group(1), m.group(2)
        text.append(word)
        if 'words' not in parsed_kw:
            parsed_kw['words'] = {}
        parsed_kw['words'][word] = prefix
        if i == 0:
            parsed_kw['first'] = prefix
    parsed_kw['text'] = ' '.join(text)
    return parsed_kw


def sent_context_lf(x, label, parsed_kw):
    n = len(parsed_kw['text'].split())
    sent_I = x['I'].split()
    sent_B = x['B'].split() if 'B' in x else []
    sent_A = x['A'].split() if 'A' in x else []

    add_w = n - 1
    if add_w:
        head_I = x['I'].split()[:add_w]
        head_A = x.get('A', '').split()[:add_w]
        sent_Be = sent_B + head_I
        sent_Ie = sent_I + head_A
    else:
        sent_Be = sent_B
        sent_Ie = sent_I

    for grams in ngrams(sent_Be, n):
        if ' '.join(grams) == parsed_kw['text'] and parsed_kw['first'] == 'B':
            return label

    for grams in ngrams(sent_Ie, n):
        if ' '.join(grams) == parsed_kw['text'] and parsed_kw['first'] == 'I':
            return label

    for grams in ngrams(sent_A, n):
        if ' '.join(grams) == parsed_kw['text'] and parsed_kw['first'] == 'A':
            return label

    return -1


def get_lfs(keywords):
    lfs = []
    idx_label_map = {}
    for i, (label, kws) in enumerate(keywords.items()):
        idx_label_map[i] = label
        for kw in kws:
            parsed_kw = parse_kw(kw)
            if parsed_kw['first'] == 'B':
                name = parsed_kw['text'] + f'...[{label}]'
            elif parsed_kw['first'] == 'I':
                name = f'[{parsed_kw["text"]}...{label}]'
            elif parsed_kw['first'] == 'A':
                name = f'[{label}]...{parsed_kw["text"]}'
            lfs.append(LabelingFunction(
                name=name,
                f=sent_context_lf,
                resources=dict(label=i, parsed_kw=parsed_kw)
            ))
    return lfs, idx_label_map


class ContextSentenceFactor(Factor):

    def _get_ctx(self, doc, r):
        toks = []
        for tok in doc:
            if tok.is_space or tok.is_punct or tok.is_quote:
                continue
            if (tok.idx + len(tok.text)) < r['start'] - self.ctx_size or tok.idx > r['end'] + self.ctx_size:
                continue
            if tok.idx + len(tok.text) >= r['start'] - self.ctx_size and tok.idx < r['start']:
                toks.append((tok, '[B]'))
            elif r['start'] <= tok.idx <= r['end']:
                toks.append((tok, '[I]'))
            else:
                toks.append((tok, '[A]'))
        return toks

    def __init__(self, tasks, results):
        self.spans = []
        ctx = defaultdict(list)
        self.ctx_size = 100
        self.docs = []
        for task, result in zip(tasks, results):
            text = task
            doc = nlp_spacy(text)
            self.docs.append(doc)
            spans = []
            for sent in doc.sents:
                spans.append({
                    'start': sent.start_char,
                    'end': sent.end_char,
                    'text': sent.text
                })
            self.spans.append(spans)

            for r in result:
                toks = self._get_ctx(doc, r)
                text = []
                text_orig = []
                for tok, prefix in toks:
                    # text.append(tok.text)
                    text.append(prefix + tok.text)
                    text_orig.append(tok.text)
                text = ' '.join(text)
                text_orig = ' '.join(text_orig)
                label = r['labels'][0]
                ctx[label].append((text, text_orig))
        self._create_model(ctx)

    def _create_model(self, ctx):
        X, y = [], []
        texts = []
        n = None
        for label in sorted(ctx):
            for text, text_orig in sorted(ctx[label]):
                X.append(text)
                texts.append(text_orig)
                y.append(label)
        le = LabelEncoder()
        y_idx = le.fit_transform(y)
        m = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 3), max_features=n, tokenizer=lambda s: s.split(), lowercase=False),
            LogisticRegression()
        )
        m.fit(X, y_idx)

        voc = {}
        for word, idx in m.steps[0][1].vocabulary_.items():
            voc[idx] = word

        k = 3
        keywords = {}
        for label, weights in zip(le.classes_, m.steps[1][1].coef_):
            i = np.argsort(weights)[-k:]
            kw = [voc[ii] for ii in i]
            keywords[label] = kw

        k = defaultdict(list)
        for label, kws in keywords.items():
            other_kws = set(sum((v for l, v in keywords.items() if l != label), []))
            for kw in kws:
                if kw in other_kws:
                    continue
                k[label].append(kw)
        keywords = k

        self.lfs, self.idx_label_map = get_lfs(keywords)
        self.applier = LFApplier(self.lfs)

    def apply(self, tasks):
        print('Create regions...')
        random.shuffle(tasks)
        regions = self.create_regions(tasks[:100])
        print(f'Num regions: {len(regions)}')
        L_train = self.applier.apply(regions)
        lfa = LFAnalysis(L=L_train, lfs=self.lfs)
        confl = lfa.lf_conflicts()
        cov = lfa.lf_coverages()
        confli = np.argsort(confl)
        lfs_sorted = [self.lfs[i] for i in confli]
        out = []
        for lf, cf, cv in zip(lfs_sorted, confl[confli], cov[confli]):
            print(lf.name, cf, cv)
            out.append({'lop': lf.name, 'conflict': cf, 'coverage': cv})
        return out

    def create_regions(self, tasks):
        regions = []
        nskipped = 0
        for task in tqdm(tasks):
            if len(task) > 10000:
                nskipped += 1
                continue
            doc = nlp_spacy(task)
            for sent in doc.sents:
                ctx = self._get_ctx(doc, {'start': sent.start_char, 'end': sent.end_char})
                r = defaultdict(list)
                for tok, where in ctx:
                    r[where.lstrip('[').rstrip(']')].append(tok.text)
                region = {}
                for where in r:
                    region[where] = ' '.join(r[where])
                if 'I' not in region:
                    continue
                regions.append(region)
        print(f'Num skipped = {nskipped}')
        return regions

    def match(self, results):
        out = defaultdict(list)
        for spans, result in zip(self.spans, results):
            scores, argmaxes = match(result, spans)
            for r, s in zip(result, scores):
                out[r['labels'][0]].append(s)
        return {label: float(np.mean(v)) for label, v in out.items()}


from label_studio.ml import LabelStudioMLBase


class SimpleWS(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(SimpleWS, self).__init__(**kwargs)

        for tag_name, tag_info in self.parsed_label_config.items():
            if tag_info['type'] == 'TextArea':
                self.textarea_from_name = tag_name
                self.textarea_to_name = tag_info['to_name']
            elif tag_info['type'] == 'Choices':
                self.choices_from_name = tag_name
                self.choices_to_name = tag_info['to_name']
            elif tag_info['type'] == 'Labels':
                self.labels_from_name = tag_name
                self.labels_to_name = tag_info['to_name']

        self.data_key = 'body'

        self.model = None
        self.lops_str = 'LOps are not trained...'

    def predict(self, tasks, **kwargs):
        results = []
        if len(tasks) > 1:
            texts = [t['data'][self.data_key] for t in tasks]
            out = self.model.apply(texts)
            self.lops_str = json.dumps(out, indent=2)

        for task in tasks:
            results.append({
                'result': [{
                    'from_name': self.textarea_from_name,
                    'to_name': self.textarea_to_name,
                    'type': 'textarea',
                    'value': {
                        'textarea': self.lops_str
                    }
                }]
            })
        return results

    def fit(self, completions, workdir=None, **kwargs):

        tasks = []
        results = []
        selection = []
        for c in completions:
            tasks.append(c['data']['body'])
            result = []
            for r in c['completions'][0]['result']:
                if r['from_name'] == self.labels_from_name:
                    result.append(r['value'])
                if r['from_name'] == self.choices_from_name:
                    selection.append([int(i) for i in r['value']['choices']])
                else:
                    selection.append(None)
            results.append(result)
        self.model = ContextSentenceFactor(tasks, results)
        return {'train': True}


def main():
    tasks, results = read_data()
    tasks_test = read_data_test()
    f = ContextSentenceFactor(tasks, results)
    lops = f.apply(tasks_test)


if __name__ == "__main__":
    main()