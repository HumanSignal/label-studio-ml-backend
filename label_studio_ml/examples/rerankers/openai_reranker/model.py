import os
import json

import openai
import logging
import prompts
import label_studio_sdk

from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase


POSITIVES = "positives"
HARD_NEGATIVES = "hard_negatives"
MODEL_VERSION = "reranker-openai-gpt4-ls"

logger = logging.getLogger(__name__)


class OpenAIReranker(LabelStudioMLBase):
    """OpenAI Reranker model helps to rerank the search results relatively to the user query."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set("model_version", MODEL_VERSION)

        # it's a standard name for API key in OpenAI
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if self.openai_api_key is None:
            raise Exception("Please set OPENAI_API_KEY environment variable.")
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        self.executor_openai = ThreadPoolExecutor(max_workers=10)
        logger.info("OpenAI initialized")

        # label studio sdk
        self.ls_api_key = os.environ.get("LABEL_STUDIO_API_KEY")
        self.ls_url = os.environ.get("LABEL_STUDIO_URL")
        if self.ls_api_key and self.ls_url:
            self.ls_client = label_studio_sdk.Client(
                url=self.ls_url, api_key=self.ls_api_key
            )
            self.ls_client.check_connection()
            logger.info("Label Studio initialized")
        else:
            raise Exception(
                "Please set LABEL_STUDIO_API_KEY and LABEL_STUDIO_URL environment variables. "
                "They are required to get batch predictions and annotations asynchronously. "
            )

    def setup(self):
        """Configure any parameters of your model here"""
        logger.info("OpenAIReranker setup completed successfully")

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> List[Dict]:
        """Rerank the documents based on the query.

        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
        :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(f"Reranker is going to predict {len(tasks)} tasks")
        project = self.ls_client.get_project(id=tasks[0]["project"])

        # run openai request in a separate thread to avoid timeouts for ml backend
        for task in tasks:
            logger.debug(f"OpenAI request starts in a thread for task {task['id']}")
            self.executor_openai.submit(self.threaded_openai_rerank, project, task)

        # return nothing because we process and save predictions
        # asynchronously in a separate thread using Label Studio SDK
        return []

    def openai_request(self, query, similar_docs):
        texts = [doc["page_content"] for doc in similar_docs]
        prompt = prompts.classification_prompt(query, texts)

        # Send request to LLM model
        logger.debug(prompt)
        response = self.openai_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4-0125-preview",
            n=1,
            stop=None,
            temperature=0.0,
        )

        # Extracting the optimized text from the response
        response = response.choices[0].message.content
        logger.debug("========================")
        logger.debug(response)
        try:
            output = json.loads(response)
        except json.JSONDecodeError:
            raise Exception(f"Can't parse JSON response from OpenAI: {response}")

        for item in output:
            item["id"] = int(item["id"])
            item["score"] = float(item["score"])

        return output

    def openai_rerank(self, query, similar_docs):
        output = self.openai_request(query, similar_docs)

        score = 0
        positives, hard_negatives = [], []
        for item in output:
            doc = similar_docs[item["id"]]
            doc["rerank"] = item["score"]
            score += item["score"]

            if item["label"] == "hard_negatives":
                hard_negatives.append(doc["id"])
            elif item["label"] == "positives":
                positives.append(doc["id"])

        score /= float(len(similar_docs))
        return self.create_prediction(MODEL_VERSION, score, positives, hard_negatives)

    def threaded_openai_rerank(self, project, task):
        prediction = self.openai_rerank(
            task['data'].get('query') or task['data'].get('question'), task["data"]["similar_docs"]
        )

        # Use Label Studio SDK to async save the prediction to the project
        project.create_prediction(task["id"], **prediction)
        logger.info(f"OpenAI prediction was created for task {task['id']}")

    @staticmethod
    def create_prediction(model_version, score, positives, hard_negatives):
        # self.set("model_version", model_version)
        return {
            "model_version": model_version,
            "score": score,
            "result": [
                {
                    "from_name": "rank",
                    "to_name": "results",
                    "type": "ranker",
                    "origin_model_version": model_version,
                    "value": {
                        "ranker": {
                            POSITIVES: list(positives),
                            HARD_NEGATIVES: list(hard_negatives),
                            "_": [],  # neutral
                        }
                    },
                }
            ],
        }

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """
        return None
