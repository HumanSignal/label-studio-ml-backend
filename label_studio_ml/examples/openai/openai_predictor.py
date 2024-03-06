import os
import openai
import difflib
import logging

from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)

openai.api_key = os.environ['OPENAI_API_KEY']


class OpenAIPredictor(LabelStudioMLBase):
    DEFAULT_PROMPT = os.path.join(os.path.dirname(__file__), 'prompt.txt')

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(OpenAIPredictor, self).__init__(**kwargs)

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
        self.labels = self.info['labels']

        self.openai_model = kwargs.get('model', 'gpt-3.5-turbo')
        self.openai_max_tokens = int(kwargs.get('max_tokens', 40))
        self.openai_temperature = float(kwargs.get('temperature', 0.5))
        self.openai_prompt = kwargs.get('prompt', self.DEFAULT_PROMPT)
        if os.path.isfile(self.openai_prompt):
            with open(self.openai_prompt) as f:
                self.openai_prompt = f.read()

        logger.debug(
            f'Initialize OpenAI API with the following parameters:'
            f' model={self.openai_model}, max_tokens={self.openai_max_tokens}, temperature={self.openai_temperature},'
            f' prompt={self.openai_prompt}')

    def _get_prompt(self, task_data):
        if os.path.isfile(self.openai_prompt):
            # Read the prompt from the file
            # that allows changing the prompt without restarting the server
            # use it only for development
            with open(self.openai_prompt) as f:
                prompt = f.read()
        else:
            prompt = self.openai_prompt
        return prompt.format(labels=self.labels, **task_data)

    def _get_predicted_label(self, task_data):
        # Create a prompt for the OpenAI API
        prompt = self._get_prompt(task_data)
        # Call OpenAI's API to create a chat completion using the GPT-3 model
        response = openai.ChatCompletion.create(
            model=self.openai_model,
            messages=[
                {"role": "user", "content": prompt}  # The 'user' role is assigned to the prompt
            ],
            max_tokens=self.openai_max_tokens,  # Maximum number of tokens in the response is set to 40
            n=1,  # We only want one response
            stop=None,  # There are no specific stop sequences
            temperature=self.openai_temperature,  # The temperature parameter affects randomness in the output. Lower values (like 0.5) make the output more deterministic.
        )
        logger.debug(f'OpenAI response: {response}')
        # Extract the response text from the ChatCompletion response
        response_text = response.choices[0].message['content'].strip()
        
        # Extract the matched labels from the response text
        matched_labels = []
        for pred in response_text.split("\n"):
            scores = list(map(lambda l: difflib.SequenceMatcher(None, pred, l).ratio(), self.labels))
            matched_labels.append(self.labels[scores.index(max(scores))])

        # Return the input_text along with the identified sentiment
        return matched_labels

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            predicted_labels = self._get_predicted_label(task['data'])
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': predicted_labels}
            }]
            predictions.append({'result': result, 'score': 1.0})
        return predictions
