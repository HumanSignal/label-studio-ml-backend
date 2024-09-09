import os
import logging
import boto3
import io
import json

from typing import List, Dict
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (
    get_image_size,
    get_single_tag_keys,
    DATA_UNDEFINED_NAME,
)
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_data_dir, get_local_path
from botocore.exceptions import ClientError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
register_all_modules()

# init mmdetection model
# see docker-compose.yml for environment variables
config_file = os.environ.get("CONFIG_FILE", "yolov3_mobilenetv2_8xb24-320-300e_coco.py")
checkpoint_file = os.environ.get("CHECKPOINT_FILE", "yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth")
device = os.environ.get("DEVICE", "cpu")
logger.info(f"Load new model from: {config_file}, {checkpoint_file}")
model = init_detector(config_file, checkpoint_file, device=device)


class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(
        self,
        image_dir=None,
        labels_file=None,
        **kwargs,
    ):
        """
        Load MMDetection model from config and checkpoint into memory.
        (Check https://mmdetection.readthedocs.io/en/v1.2.0/GETTING_STARTED.html#high-level-apis-for-testing-images)

        Optionally set mappings from COCO classes to target labels
        :param config_file: Absolute path to MMDetection config file (e.g. /home/user/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x.py)
        :param checkpoint_file: Absolute path MMDetection checkpoint file (e.g. /home/user/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param labels_file: file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
        :param score_threshold: score threshold to wipe out noisy results
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs:
        """
        super(MMDetection, self).__init__(**kwargs)

        # default image upload folder
        upload_dir = os.path.join(get_data_dir(), "media", "upload")
        self.image_dir = image_dir or upload_dir
        logger.debug(f"{self.__class__.__name__} reads images from {self.image_dir}")

        # try to load label map from json file (optional)
        self.labels_file = labels_file
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        # try to parse labeling config to get image $value, from_name, to_name, labels
        params = get_single_tag_keys(
            self.parsed_label_config, "RectangleLabels", "Image"
        )
        self.from_name, self.to_name, self.value, self.labels_in_config = params
        self.labels_in_config = set(self.labels_in_config)
        self.score_threshold = float(os.environ.get("SCORE_THRESHOLD", 0.5))

        # try to build label map from mmdet labels to LS labels
        schema = list(self.parsed_label_config.values())[0]
        self.labels_attrs = schema.get("labels_attrs")
        self.build_labels_from_labeling_config(schema)

    def setup(self):
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

    def build_labels_from_labeling_config(self, schema):
        """
        Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tags,
        e.g. "airplane", "car" are label names from COCO dataset
        """
        mmdet_labels = model.dataset_meta.get("classes", [])
        mmdet_labels_lower = [label.lower() for label in mmdet_labels]
        print(
            "COCO dataset labels supported by this mmdet model:",
            model.dataset_meta,
        )

        # if labeling config has Label tags
        if self.labels_attrs:
            # try to find something like <Label value="Vehicle" predicted_values="airplane,car">
            for ls_label, label_attrs in self.labels_attrs.items():
                predicted_values = label_attrs.get("predicted_values", "").split(",")
                for predicted_value in predicted_values:
                    predicted_value = predicted_value.strip()  # remove spaces at the beginning and at the end
                    if predicted_value:  # it shouldn't be empty (like '')
                        if predicted_value not in mmdet_labels:
                            print(
                                f'Predicted value "{predicted_value}" is not in mmdet labels'
                            )
                        self.label_map[predicted_value] = ls_label

        # label map is still empty, not predicted_values found in Label tags,
        # try to build mapping automatically: map LS labels.lower() to mmdet_labels.lower() directly
        if not self.label_map:
            # try to find thin <Label value="Vehicle" predicted_values="airplane,car">
            for ls_label, _ in self.labels_attrs.items():
                try:
                    index = mmdet_labels_lower.index(ls_label.lower())
                except ValueError:
                    continue  # no label found in mmdet labels
                else:
                    self.label_map[mmdet_labels[index]] = ls_label

        print("MMDetection => Label Studio mapping of labels :\n", self.label_map)

    def _get_image_url(self, task: Dict) -> str:
        image_url = task["data"].get(self.value) or task["data"].get(
            DATA_UNDEFINED_NAME
        )

        # retrieve image from s3 bucket,
        # set env vars AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        # and AWS_SESSION_TOKEN to allow boto3 to access the bucket
        if image_url.startswith("s3://") and os.getenv('AWS_ACCESS_KEY_ID'):
            # pre-sign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip("/")
            client = boto3.client("s3")
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": bucket_name, "Key": key},
                )
            except ClientError as exc:
                logger.warning(
                    f"Can't generate pre-signed URL for {image_url}. Reason: {exc}"
                )

        return image_url

    def predict(self, tasks: List[Dict], **kwargs):
        if len(tasks) > 1:
            print(
                "==> Only the first task will be processed to avoid ML backend overloading"
            )
            tasks = [tasks[0]]

        predictions = []
        for task in tasks:
            prediction = self.predict_one_task(task)
            predictions.append(prediction)
        return predictions

    def predict_one_task(self, task: Dict):
        image_url = self._get_image_url(task)
        image_path = get_local_path(image_url, task_id=task.get('id'))
        model_results = inference_detector(model, image_path).pred_instances
        results = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)
        classes = model.dataset_meta.get("classes")
        # print(f">>> model_results: {model_results}")
        # print(f">>> label_map {self.label_map}")
        # print(f">>> self.model.dataset_meta: {self.model.dataset_meta}")

        for item in model_results:
            bboxes, label, scores = item["bboxes"], item["labels"][0], item["scores"]
            mmdet_label = classes[label]
            score = float(scores[-1])
            print("----------------------")
            print(f"task id > {task.get('id')}")
            print(f"bboxes > {bboxes}")
            print(f"label > {mmdet_label}")
            print(f"score > {score}")

            # bbox score is too low
            if score < self.score_threshold:
                continue

            # there is no mapping between MMDet label and LS label
            if mmdet_label not in self.label_map:
                continue

            output_label = self.label_map[
                mmdet_label
            ]  # map from MMDet label to LS label name
            print(f">>> LS output label: {output_label}")
            if output_label not in self.labels_in_config:
                print(output_label + " label not found in project config.")
                continue

            for bbox in bboxes:
                bbox = list(bbox)
                if not bbox:
                    continue

                x, y, xmax, ymax = bbox[:4]
                results.append(
                    {
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "type": "rectanglelabels",
                        "value": {
                            "rectanglelabels": [output_label],
                            "x": float(x) / img_width * 100,
                            "y": float(y) / img_height * 100,
                            "width": (float(xmax) - float(x)) / img_width * 100,
                            "height": (float(ymax) - float(y)) / img_height * 100,
                        },
                        "score": score,
                    }
                )
                all_scores.append(score)
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        print(f">>> RESULTS: {results}")
        return {"result": results, "score": avg_score, "model_version": self.get("model_version")}


def json_load(file, int_keys=False):
    with io.open(file, encoding="utf8") as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
