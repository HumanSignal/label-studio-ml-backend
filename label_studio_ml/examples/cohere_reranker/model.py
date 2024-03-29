import os
import json
import openai
import cohere
import logging

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase

POSITIVES = "positives"
HARD_NEGATIVES = "hard_negatives"

logger = logging.getLogger(__name__)


class CohereReranker(LabelStudioMLBase):
    """Cohere Reranker model helps to rerank the search results relatively to the user query."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # it's a standard name for API key in Cohere
        self.co_api_key = os.getenv("CO_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

        self.co = cohere.Client(api_key=self.co_api_key)
        logger.info("CohereReranker initialized")

        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "rerank-english-v2.0")
        logger.info("CohereReranker setup completed successfully")

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> List[Dict]:
        """Rerank the documents based on the query.

        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
        :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(f"CohereReranker is going to predict {len(tasks)} tasks")
        predictions = []

        for task in tasks:
            query = task["data"]["query"]
            similar_docs = task["data"]["similar_docs"]

            if self.openai_api_key:
                prediction = self.openai_rerank(query, similar_docs)
            else:
                prediction = self.cohere_rerank(query, similar_docs)

            predictions.append(prediction)

        return predictions

    def cohere_rerank(self, query, similar_docs):
        texts = [doc["page_content"] for doc in similar_docs]

        response = self.co.rerank(
            query=query, documents=texts, top_n=len(texts), model=self.model_version
        )
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

    def openai_rerank(self, query, similar_docs):
        output = self.openai_request(query, similar_docs)

        score = 0
        positives, hard_negatives = [], []
        for item in output:
            doc = similar_docs[item["id"]]
            doc["rerank"] = item["score"]
            score += item["score"]

            if item["label"] == "hard_negative":
                hard_negatives.append(doc["id"])
            elif item["label"] == "positive":
                positives.append(doc["id"])

        score /= float(len(similar_docs))
        return self.create_prediction("openai-gpt4", score, positives, hard_negatives)

    def openai_request(self, query, similar_docs):
        texts = [doc["page_content"] for doc in similar_docs]

        prompt = (
            "1. Rerank the documents based on the QUERY relative to TEXTS.\n"
            "2. For each TEXT you should output a relevance score as the `score` field between 0.0 and 1.0, "
            "where 0.0 is the worse, 1.0 is the best.\n"
            "3. For each TEXT you should define if it is a POSITIVE or HARD NEGATIVE or NEUTRAL label "
            "based on the relevance and output it as `label` field.\n"
            "5. Each TEXT starts with `===>`, a number and a newline, then a text fragment follows (e.g. `\n===> 1.\n this is text one\n`), "
            "you should use this number in the JSON output as the `id` field.\n"
            "6. The output must be in JSON format only, never use ``` in the beginning of output, write JSON as is, "
            "e.g.:\n\n"
            "[\n"
            '{"id": 1, "score": 0.9, "label": "positive"},\n'
            '{"id": 2, "score": 0.1, "label": "hard_negative"},\n'
            '{"id": 3, "score": 0.5, "label": "neutral"},\n'
            "]\n\n"
            "---------------------------\n"
            "QUERY:\n"
            f"{query}\n"
            "---------------------------\n"
            "TEXTS:\n"
            + ("".join([f"\n\n===> {i}.\n{text}" for i, text in enumerate(texts)]))
        )

        # Send request to LLM model
        print(prompt)
        response = self.openai_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4-0125-preview",
            n=1,
            stop=None,
            temperature=0.0,
        )

        # Extracting the optimized text from the response
        response = response.choices[0].message.content
        print("========================")
        print(response)
        try:
            output = json.loads(response)
        except json.JSONDecodeError:
            raise Exception(f"Can't parse JSON response from OpenAI: {response}")

        for item in output:
            item["id"] = int(item["id"])
            item["score"] = float(item["score"])

        return output

    def create_prediction(self, model_version, score, positives, hard_negatives):
        return {
            "model_version": model_version,
            "score": score,
            "result": [
                {
                    "from_name": "rank",
                    "to_name": "results",
                    "type": "ranker",
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

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        # store new data to the cache
        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print("fit() completed successfully.")
