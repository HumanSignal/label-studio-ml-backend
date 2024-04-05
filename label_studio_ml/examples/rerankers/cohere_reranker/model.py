import os
import time

import cohere
import logging
import label_studio_sdk

from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase


POSITIVES = "positives"
HARD_NEGATIVES = "hard_negatives"

logger = logging.getLogger(__name__)


class CohereReranker(LabelStudioMLBase):
    """Cohere Reranker model helps to rerank the search results relatively to the user query."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set("model_version", "rerank-english-v2.0")

        # it's a standard name for API key in Cohere
        self.co_api_key = os.getenv("CO_API_KEY")
        self.co = cohere.Client(api_key=self.co_api_key)
        self.executor_cohere = ThreadPoolExecutor(max_workers=1)
        logger.info("CohereReranker initialized")

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
        logger.info("CohereReranker setup completed successfully")

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> List[Dict]:
        """Rerank the documents based on the query.

        :param tasks:
            [Label Studio tasks in JSON format]
            (https://labelstud.io/guide/task_format.html)
        :param context:
            [Label Studio context in JSON format]
            (https://labelstud.io/guide/ml_create#Implement-prediction-logic)
        :return predictions:
            if there is one task, it returns one prediction for this task, otherwise it returns [],
            because predictions are saved using Label Studio SDK asynchronously
            [Predictions array in JSON format]
            (https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(f"Reranker is going to predict {len(tasks)} tasks")
        project = self.ls_client.get_project(id=tasks[0]["project"])

        # Cohere: run for 1 task for labeling stream
        # because it requires showing the results immediately
        if len(tasks) == 1:
            task = tasks[0]
            prediction = self.cohere_rerank(
                task["data"]["query"], task["data"]["similar_docs"]
            )
            logger.info(f"Cohere prediction was created for task {task['id']}")
            return [prediction]

        # Cohere: run for all tasks in threads
        for task in tasks:
            logger.debug(f"Cohere request starts in a thread for task {task['id']}")
            self.executor_cohere.submit(self.threaded_cohere_rerank, project, task)

        # return nothing because we process and save predictions
        # asynchronously in a separate thread using Label Studio SDK
        return []

    def cohere_rerank(self, query, similar_docs):
        texts = [doc["page_content"] for doc in similar_docs]

        error, response, try_count = True, None, 0
        while error:
            try:
                response = self.co.rerank(
                    query=query,
                    documents=texts,
                    top_n=len(texts),
                    model=self.model_version,
                )
                error = False
            except cohere.errors.too_many_requests_error.TooManyRequestsError:
                try_count += 1
                logger.info(
                    f"Cohere API rate limit exceeded, retrying {try_count}. "
                    f"You can use paid plan to increase the rate limit. "
                    f"Sleeping for 60 seconds to recover  ..."
                )
                time.sleep(60)

        logger.info(f'Reranked query with {len(texts)} docs: "{query}"')

        score = 0
        positives, hard_negatives = [], []
        for result in response.results:
            doc = similar_docs[result.index]
            rerank_score = result.relevance_score
            doc["rerank"] = rerank_score
            score += rerank_score

            if rerank_score < 0.10:
                hard_negatives.append(doc["id"])
            elif rerank_score > 0.98:
                positives.append(doc["id"])

        score /= float(len(similar_docs))
        return self.create_prediction(
            "rerank-english-v2.0", score, positives, hard_negatives
        )

    def threaded_cohere_rerank(self, project, task):
        # create predictions using cohere
        prediction = self.cohere_rerank(
            task["data"]["query"], task["data"]["similar_docs"]
        )
        # Use Label Studio SDK to async save the prediction to the project
        project.create_prediction(task_id=task["id"], **prediction)
        logger.info(f"Cohere prediction was created for task {task['id']} using LS SDK")

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

    @staticmethod
    def compare_predictions(a: dict, b: dict) -> float:
        # compare two lists - how many elements are the same
        positives_a = a["result"][0]["value"]["ranker"][POSITIVES]
        positives_b = b["result"][0]["value"]["ranker"][POSITIVES]
        negatives_a = a["result"][0]["value"]["ranker"][HARD_NEGATIVES]
        negatives_b = b["result"][0]["value"]["ranker"][HARD_NEGATIVES]
        positive_score = len(set(positives_a) & set(positives_b)) / len(
            set(positives_a) | set(positives_b)
        )
        negative_score = len(set(negatives_a) & set(negatives_b)) / len(
            set(negatives_a) | set(negatives_b)
        )
        return (positive_score + negative_score) / 2

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
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        # store new data to the cache
        # self.set("my_data", "my_new_data_value")
        # self.set("model_version", "my_new_model_version")
        # print(f'New data: {self.get("my_data")}')
        # print(f'New model version: {self.get("model_version")}')

        print("fit() completed successfully.")
