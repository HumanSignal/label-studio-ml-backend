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

    

    
    def predict(self, tasks, **kwargs):
        #make predictions with currently set model
        flair_sents = [Sentence(task['data']['text']) for task in tasks] #collect text data for each task in a list and make flair sent
        #predict with ner model for each flair sentence
        for sent in flair_sents:
            self.model.predict(sent)
        
        return self.convert_to_ls_annotation(flair_sents)
        
                    
                
    
