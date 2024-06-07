import json
import os
import time
import shutil
import traceback

import cohere
import logging
import label_studio_sdk

from cohere.finetuning import (
    BaseModel,
    BaseType,
    FinetunedModel,
    Settings,
)
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from types import SimpleNamespace

POSITIVES = "positives"
HARD_NEGATIVES = "hard_negatives"
DEFAUL_MODEL_VERSION = "reranker-cohere-english-v3.0"
CUSTOM_MODEL_VERSION = "reranker-cohere-custom"

TRAIN_DIR = "data/train"
TRAIN_FILE = 'train.jsonl'

logger = logging.getLogger(__name__)


class CohereReranker(LabelStudioMLBase):
    """Cohere Reranker model helps to rerank the search results relatively to the user query."""

    def __init__(self, *args, **kwargs):
        # it's a standard name for API key in Cohere
        self.co_api_key = os.getenv("CO_API_KEY")
        self.co = cohere.Client(api_key=self.co_api_key)
        self.executor_cohere = ThreadPoolExecutor(max_workers=2)

        super().__init__(*args, **kwargs)

        self.cohere_model = self.get_cohere_model()
        logger.info("CohereReranker initialized")

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
                "They are required to get annotations for training. "
            )

    def get_cohere_model(self):
        response = self.co.finetuning.list_finetuned_models()
        models = [
            model for model in response.finetuned_models
            if model.name.startswith(CUSTOM_MODEL_VERSION) and model.status in ["STATUS_READY", 'STATUS_PAUSED']
        ]

        # return the last model by created_at
        models = sorted(models, key=lambda x: x.created_at, reverse=True)
        self.cohere_model = (
            models[0] if models else SimpleNamespace(id=DEFAUL_MODEL_VERSION, name=DEFAUL_MODEL_VERSION)
        )
        self.set("model_version", self.cohere_model.name + "-" + self.cohere_model.id[0:4])
        return self.cohere_model

    def setup(self):
        """Configure any parameters of your model here"""
        self.get_cohere_model()
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
        self.get_cohere_model()
        logger.info(f"Reranker is going to predict {len(tasks)} tasks with model '{self.model_version}'")

        # Cohere: run for 1 task for labeling stream
        # because it requires showing the results immediately
        if len(tasks) == 1:
            task = tasks[0]
            prediction = self.cohere_rerank(
                task['data'].get('query') or task['data'].get('question'), task['data']['similar_docs']
            )
            logger.info(f"Cohere prediction was created for task {task['id']}")
            return [prediction]

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
                    model=self.cohere_model.id + '-ft',
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
            self.model_version, score, positives, hard_negatives
        )

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
        """ You can use it as the Custor Agreement Metric in
        Label Studio Enterprise to compare the annotations
        """
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
        :param data: the payload received from the event
        (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        if event in ["ANNOTATION_CREATED", "ANNOTATION_UPDATED"]:
            task = self.save_task(data)
            logger.info(f"Task {task['id']} with {len(task['annotations'])} was added successfully")

        if event == 'START_TRAINING':
            self.executor_cohere.submit(self.start_training, data)
            logger.info(f"Training event was received, starting training in background thread ...")

    @staticmethod
    def save_task(data):
        project_id = data["task"].get('project', 0)
        project_dir = os.path.join(TRAIN_DIR, str(project_id))
        os.makedirs(project_dir, exist_ok=True)

        # load task with annotations
        task_path = os.path.join(project_dir, str(data['task']['id']) + '.json')
        if os.path.exists(task_path):
            with open(task_path, 'r') as f:
                task = json.load(f)
        # or use a new task
        else:
            task = data["task"]
            task['annotations'] = []

        # update annotation if it already exists in the task
        for i, existing_annotation in enumerate(task["annotations"]):
            if existing_annotation["id"] == data["annotation"]["id"]:
                task['annotations'][i] = data["annotation"]
                break
        # or add a new annotation
        else:
            task['annotations'].append(data.get("annotation"))

        # save task with annotations
        with open(task_path, 'w') as f:
            json.dump(task, f)

        return task

    @staticmethod
    def annotation2cohere(task):
        """Convert the last updated annotation from a task
        to relevant passages and hard negatives in cohere format.
        """
        # get docs from task data and make mapping by ids
        docs = task['data']['similar_docs']
        docs = {doc['id']: doc for doc in docs}

        annotations = sorted(task['annotations'], key=lambda x: x['updated_at'])
        annotation = annotations[-1]  # take the last added annotation

        positives, negatives = [], []
        for doc_id in annotation['result'][0]['value']['ranker'][POSITIVES]:
            positives.append(docs[doc_id]['page_content'])
        for doc_id in annotation['result'][0]['value']['ranker'][HARD_NEGATIVES]:
            negatives.append(docs[doc_id]['page_content'])

        if not positives:
            return None

        # if the same item in positives and negatives, remove it from both
        intersection = set(positives) & set(negatives)
        for item in intersection:
            positives = [i for i in positives if i != item]
            negatives = [i for i in negatives if i != item]

        if not positives:
            return None

        # return relevant passages and hard negatives in cohere format
        return {
            "query": task['data'].get('query') or task['data'].get('question'),
            "relevant_passages": list(set(positives)),
            "hard_negatives": list(set(negatives))
        }

    def convert_dataset(self, project_id):
        count = 0
        project_dir = os.path.join(TRAIN_DIR, str(project_id))
        out_path = os.path.join(project_dir, TRAIN_FILE)

        with open(out_path, 'w') as fout:
            # read each task, get the last updated annotation and save it in jsonl
            for task_file in os.listdir(project_dir):
                if not task_file.endswith('.json'):
                    continue

                with open(os.path.join(project_dir, task_file), 'r') as f:
                    task = json.load(f)
                    annotations = task.get('annotations', [])
                    if not annotations:
                        continue

                    # save only the last annotation
                    record = self.annotation2cohere(task)
                    if record:
                        fout.write(json.dumps(record) + '\n')
                        count += 1

        logger.info(f"Dataset for project {project_id} with {count} annotations was saved to {out_path}")
        return out_path

    def download_snapshot(self, project_id):
        # download all tasks for the project
        project = self.ls_client.get_project(project_id)

        # create new export snapshot
        export_result = project.export_snapshot_create(
            title="Export SDK Snapshot",
        )
        assert "id" in export_result
        export_id = export_result["id"]

        # wait until snapshot is ready
        while project.export_snapshot_status(export_id).is_in_progress():
            time.sleep(1.0)

        # download snapshot file
        status, filename = project.export_snapshot_download(export_id, export_type="JSON")
        assert status == 200
        assert filename is not None

        logger.info(f"Status of the task export is {status}.\nFile name is {filename}")
        return filename

    def download_all_tasks(self, project_id):
        filename = CohereReranker.download_snapshot(self, project_id)

        # convert all tasks to separate files
        with open(filename, 'r') as f:
            tasks = json.load(f)

        # save each task in a separate file
        project_dir = os.path.join(TRAIN_DIR, str(project_id))
        # remove project_dir
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
        os.makedirs(project_dir, exist_ok=True)
        for task in tasks:
            task_path = os.path.join(project_dir, str(task['id']) + '.json')
            with open(task_path, 'w') as f:
                json.dump(task, f)

        logger.info(f"All {len(tasks)} tasks for project {project_id} were saved to {project_dir}")

    def cohere_train(self, project_id):
        name = self.get_last_model_name()

        project_dir = os.path.join(TRAIN_DIR, str(project_id))
        train_path = os.path.join(project_dir, TRAIN_FILE)
        dataset_name = name + '-dataset'
        logger.info(f"Training Cohere dataset: {dataset_name} with train file: {train_path}")

        rerank_dataset = self.co.datasets.create(
            name=dataset_name,
            data=open(train_path, "rb"),
            type="reranker-finetune-input"
        )
        dataset_response = self.co.wait(rerank_dataset)
        logger.info(f"Cohere dataset validation is done: {dataset_response}")

        # start the fine-tune job using this dataset
        finetuned_model = self.co.finetuning.create_finetuned_model(
            request=FinetunedModel(
                name=name,
                settings=Settings(
                    base_model=BaseModel(
                        name="english",
                        # version="3.0.0",
                        base_type="BASE_TYPE_RERANK",
                    ),
                    dataset_id=rerank_dataset.id,
                ),
            )
        )

        logger.info(
            "------------------------------------------------------------------------\n"
            f"Cohere Fine-tune created: {finetuned_model}\n"
            "------------------------------------------------------------------------\n"
        )

    def get_last_model_name(self):
        # get the latest cohere model
        response = self.co.finetuning.list_finetuned_models()
        models = sorted([
            model for model in response.finetuned_models if model.name.startswith(CUSTOM_MODEL_VERSION)
        ], key=lambda x: x.created_at, reverse=True)
        # return the last model by created_at
        if not models:
            # it's the first training
            name = CUSTOM_MODEL_VERSION + '-1'
        else:
            iteration = models[0].name.split('-')[-1]
            name = f"{CUSTOM_MODEL_VERSION}-{int(iteration) + 1}"
        return name

    def start_training(self, data):
        try:
            project_id = data['project']['id']
            logger.info('Downloading all tasks ...')
            self.download_all_tasks(project_id)
            logger.info('Convert annotations to cohere format ...')
            self.convert_dataset(project_id)
            logger.info('Cohere train ...')
            self.cohere_train(project_id)
        except Exception as exc:
            logger.error(f"Training failed with error: {exc}\n")
            logger.error(f"{traceback.format_exc()}")
            raise exc
