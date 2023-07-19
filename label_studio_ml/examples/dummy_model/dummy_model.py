import random

from label_studio_ml.model import LabelStudioMLBase


class DummyModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(DummyModel, self).__init__(**kwargs)
        
        # pre-initialize your variables here
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

    def predict(self, tasks, **kwargs):
        """ This is where inference happens: 
            model returns the list of predictions based on input list of tasks
            
            :param tasks: Label Studio tasks in JSON format
        """
        results = []
        for task in tasks:
            results.append({
                'result': [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'choices',
                    'value': {
                        'choices': [random.choice(self.labels)]
                    }
                }],
                'score': random.uniform(0, 1)
            })
        return results

    def fit(self, event, data, **kwargs):
        """ This is where training happens: train your model given list of completions,
            then returns dict with created links and resources

            :param completions: aka annotations, the labeling results from Label Studio 
            :param workdir: current working directory for ML backend
        """
        # save some training outputs to the job result
        return {'random': random.randint(1, 10)}
