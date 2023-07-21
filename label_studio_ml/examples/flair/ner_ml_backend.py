from flair.datasets import SentenceDataset
from flair.data import Corpus, Sentence

from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from label_studio_ml.model import LabelStudioMLBase

import os

#writing class with inheretance
class SequenceTaggerModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        #initialize base class
        super(SequenceTaggerModel, self).__init__(**kwargs)
       
        # you can load in information from your labelling interface
        # you need this information to load in annotations for fit or make predictions
        print("Full parsed labels: ", list(self.parsed_label_config.items()))
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name #this is the name of the tagset
        self.to_name = schema['to_name'][0] #this is the name of the raw data (f.e. "text")
        self.labels = schema['labels'] #these are the label names
        
        #for debugging
        print('from_name:', self.from_name) 
        print('to_name:', self.to_name)
        print('labels:', self.labels)
        print('current dir: ', os.getcwd())
        
        
        #if a model has been trained, load in for making predictions
        if self.train_output:
            self.model = self.load(self.train_output['base_path'])
        else:
            print("no model trained yet")
        
    def load(self, path):
        #helper function to load in flair based NER model
        return SequenceTagger.load(path+'/best-model.pt')
    
    def convert_to_flair_annotation(self, sentence, annotations):
        #convert label-studio annotation to flair annotation (BIO format)
        #Entities can contain multiple tokens in this format f.e. George Washington => Person
        for token in sentence:
            for tag in annotations[0]['result']:
                start = tag['value'].get('start')
                end = tag['value'].get('end')
                label = tag['value'].get('labels')[0]
                if (start == token.start_pos) and (end >= token.end_pos): #Begin token of entity
                    token.add_tag('ner', "B-"+label)
                    break
                elif (start < token.start_pos) and (end >= token.end_pos): #middle token or end of entity
                    token.add_tag('ner', "I-"+label)
                    break
                else:
                    pass
        
        return sentence
    
    def convert_to_ls_annotation(self, flair_sentences):
        #convert annotations in flair sentences object to labelstudio annotations
        results = []
        for sent in flair_sentences:
            sent_preds = [] #all predictions results for one sentence = one task
            tags = sent.to_dict('ner')
            scores_sent = []
            for ent in tags['entities']:
                sent_preds.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'labels',
                    "value": {
                    "start": ent['start_pos'],
                    "end": ent['end_pos'],
                    "text": ent['text'],
                    "labels": [ent['labels'][0].value]
                        }})
                #add score
                scores_sent.append(float(ent['labels'][0].score))
            
            #add minimum of certaincy scores of entities in sentence for active learning use
            score = min(scores_sent) if len(scores_sent) > 0 else float(2.0)
            results.append({'result': sent_preds,
                           'score': score}) 
        
        return results

    def _get_annotated_dataset(self, project_id):
        raise NotImplementedError('For this model, you need to implement data ingestion pipeline: '
                                  'go to ner.py > _get_annotated_dataset() and put your logic to retrieve'
                                  f'the list of annotated tasks from Label Studio project ID = {project_id}')

    
    def fit(self, event, data, **kwargs):
        #completions contain ALL the annotated samples.
        #train a model from scratch here.
        flair_sents = []
        completions = self._get_annotated_dataset(data['project_id'])
        for compl in completions:
            sent = Sentence(compl['data'][self.to_name]) #get raw sentence and convert to flair
            annotations = compl['annotations']
            sent = self.convert_to_flair_annotation(sent, annotations)
            
            #only add sentences that contain entities to dataset
            if len(sent.get_spans('ner')) != 0:
                flair_sents.append(sent)
        
        
        #make data ready for flair by making SentenceDataset and Corpus object
        data = SentenceDataset(flair_sents)
        corpus = Corpus(train=data, dev=None, test=None, name="ner-corpus", sample_missing_splits=True)
        
        #for debugging
        print("size train set: ", len(corpus.train))
        print("size dev set: ", len(corpus.dev))
        print("size test set: ", len(corpus.test))
        
        tag_type = 'ner'
        # 3. make the tag dictionary from the corpus
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
        print(tag_dictionary)

        # 4. initialize embeddings, here embeddings for dutch language
        embedding_types = [
            FlairEmbeddings('news-forward'),
            FlairEmbeddings('news-backward'),
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        # 5. initialize sequence tagger
        model: SequenceTagger = SequenceTagger(hidden_size=256//4,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type,
                                                use_crf=True)

        trainer: ModelTrainer = ModelTrainer(model, corpus)

        # 7. start training
        parameters = {
            'base_path': workdir, #workdir is set by label-studio
            'learning_rate':1e-1,
            'mini_batch_size':16,
            'max_epochs':20
        }
        
        #train and evaluate on test set
        trainer.train(**parameters)
        
        #print out evaluation for train set in console
        print("training evaluation:")
        print(model.evaluate(corpus.train, gold_label_type='ner')) 
        
        return parameters
    
    def predict(self, tasks, **kwargs):
        #make predictions with currently set model
        flair_sents = [Sentence(task['data']['text']) for task in tasks] #collect text data for each task in a list and make flair sent
        #predict with ner model for each flair sentence
        for sent in flair_sents:
            self.model.predict(sent)
        
        return self.convert_to_ls_annotation(flair_sents)
        
                    
                
    
