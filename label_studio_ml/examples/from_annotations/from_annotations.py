import random
from copy import deepcopy

from label_studio_ml.model import LabelStudioMLBase


class AnnotationModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(AnnotationModel, self).__init__(**kwargs)
        # pre-initialize your variables here
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.schema = schema
        self.to_name = schema['to_name'][0]

    def predict(self, tasks, **kwargs):
        """ This is where inference happens:
            model returns the list of predictions based on input list of annotations
            :param tasks: Label Studio tasks in JSON format
            :return results: predictions in LS format
        """
        results = []
        for task in tasks:
            annotations = task['annotations']
            ann = random.choice(annotations)
            results.append({
                'result': deepcopy(ann['result']),
                'score': random.uniform(0, 1)
            })
        return results

    def fit(self, completions, workdir=None, **kwargs):
        """ This is where training happens: train your model given list of completions,
            then returns dict with created links and resources

            :param completions: aka annotations, the labeling results from Label Studio
            :param workdir: current working directory for ML backend
        """
        # save some training outputs to the job result
        return {'random': random.randint(1, 10)}
