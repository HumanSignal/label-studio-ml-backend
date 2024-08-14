import logging
import os

from uuid import uuid4
from typing import List, Dict, Optional, Any
from label_studio_ml.model import LabelStudioMLBase
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from label_studio_ml.utils import match_labels

logger = logging.getLogger(__name__)

try:
    search = GoogleSearchAPIWrapper()
except Exception as e:
    logger.error(f'Error initializing GoogleSearchAPIWrapper: {e}. '
                 f'You will not be able to use the search tool.')
    search = None


class SearchResults(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.snippets = []

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        self.snippets = []

    def on_tool_end(self, output: str, **kwargs):
        """Run when tool ends running."""
        for snippet in output.split('...'):
            snippet = snippet.strip()
            if snippet:
                self.snippets.append(snippet)


class LangchainSearchAgent(LabelStudioMLBase):
    PROMPT_PREFIX = os.getenv('PROMPT_PREFIX', 'prompt')
    RESPONSE_PREFIX = os.getenv('RESPONSE_PREFIX', 'response')
    SNIPPETS_PREFIX = os.getenv('SNIPPETS_PREFIX', 'snippets')
    PROMPT_TEMPLATE = os.getenv('PROMPT_TEMPLATE', '{prompt}{text}')

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def get_prompt(self, annotation, prompt_from_name) -> str:
        result = annotation['result']
        for item in result:
            if item.get('from_name') != prompt_from_name:
                continue
            return '\n'.join(item['value']['text'])
        return ''

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        from_name, to_name, value = self.get_first_tag_occurence('Choices', 'Text')
        from_name_prompt, _, _ = self.get_first_tag_occurence(
            'TextArea', 'Text', name_filter=lambda s: s.startswith(self.PROMPT_PREFIX))
        from_name_response, _, _ = self.get_first_tag_occurence(
            'TextArea', 'Text', name_filter=lambda s: s.startswith(self.RESPONSE_PREFIX))
        from_name_snippets, _, _ = self.get_first_tag_occurence(
            'TextArea', 'Text', name_filter=lambda s: s.startswith(self.SNIPPETS_PREFIX))

        search_results = SearchResults()
        if not search:
            tools = []
        else:
            tools = [Tool(
                name="Google Search Snippets",
                description="Search Google for recent results.",
                func=search.run,
                callbacks=[search_results]
            )]
        llm = OpenAI(
            temperature=0,
            model_name='gpt-3.5-turbo-instruct'
        )
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate",
        )

        labels = self.parsed_label_config[from_name]['labels']
        predictions = []
        if context:
            prompt = self.get_prompt(context, from_name_prompt)
        else:
            prompt = self.get(from_name_prompt)

        if not prompt:
            return []

        logger.debug(f'Prompt: {prompt}')

        base_result = {
            'id': str(uuid4())[:4],
            'from_name': from_name_prompt,
            'to_name': to_name,
            'type': 'textarea',
            'value': {
                'text': [prompt]
            }
        }

        for task in tasks:
            text = self.preload_task_data(task, task['data'][value])
            full_prompt = self.PROMPT_TEMPLATE.format(prompt=prompt, text=text)
            logger.info(f'Full prompt: {full_prompt}')
            llm_result = agent.run(full_prompt)
            output_classes = match_labels(llm_result, labels)
            snippets = search_results.snippets
            logger.debug(f'LLM result: {llm_result}')
            logger.debug(f'Output classes: {output_classes}')
            logger.debug(f'Snippets: {snippets}')
            result = [base_result.copy()] + [{
                'from_name': from_name,
                'to_name': to_name,
                'type': 'choices',
                'value': {
                    'choices': output_classes
                }
            }, {
                'from_name': from_name_response,
                'to_name': to_name,
                'type': 'textarea',
                'value': {
                    'text': [llm_result]
                }
            }]
            if snippets:
                result.append({
                    'from_name': from_name_snippets,
                    'to_name': to_name,
                    'type': 'textarea',
                    'value': {
                        'text': snippets
                    }
                })
            predictions.append({
                'result': result,
                'model_version': self.get('model_version'),
            })

        return predictions

    def fit(self, event, data, **kwargs):
        logger.debug(f'Data received: {data}')
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED'):
            return

        prompt_from_name, prompt_to_name, value_key = self.get_first_tag_occurence(
            'TextArea', 'Text',
            name_filter=lambda s: s.startswith(self.PROMPT_PREFIX))
        prompt = self.get_prompt(data['annotation'], prompt_from_name)
        self.set(prompt_from_name, prompt)
