import logging
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from label_studio_ml.utils import match_labels

logger = logging.getLogger(__name__)

search = GoogleSearchAPIWrapper()


class SearchResults(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.snippets = []

    def on_tool_end(self, output: str, **kwargs):
        """Run when tool ends running."""
        self.snippets = [snippet.strip() for snippet in output.split(' ... ')]


class NewModel(LabelStudioMLBase):

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        from_name, to_name, value = self.get_first_tag_occurence('Choices', 'Text')
        search_results = SearchResults()
        tool = Tool(
            name="Google Search Snippets",
            description="Search Google for recent results.",
            func=search.results,
            callbacks=[search_results]
        )
        llm = OpenAI(temperature=0)
        agent = initialize_agent(
            [tool],
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        labels = self.parsed_label_config[from_name]['labels']
        predictions = []
        for task in tasks:
            text = task['data'][value]
            llm_result = agent.run(f'"{text}" - Is this a company or product? Answer only "Company" or "Product"')
            output_class = match_labels(llm_result, labels)
            snippets = search_results.snippets
            logger.debug(f'LLM result: {llm_result}')
            logger.debug(f'Output class: {output_class}')
            logger.debug(f'Snippets: {snippets}')
            predictions.append({
                'result': [{
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'choices',
                    'value': {
                        'choices': [output_class]
                    }
                }]
            })

        return predictions

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

