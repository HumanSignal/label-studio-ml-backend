import os
import logging

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from control_models.base import ControlModel
from control_models.choices import ChoicesModel
from control_models.rectangle_labels import RectangleLabelsModel
from control_models.rectangle_labels_obb import RectangleLabelsObbModel
from control_models.polygon_labels import PolygonLabelsModel
from control_models.keypoint_labels import KeypointLabelsModel
from control_models.video_rectangle import VideoRectangleModel
from control_models.timeline_labels import TimelineLabelsModel
from typing import List, Dict, Optional


logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)

# Register available model classes
available_model_classes = [
    ChoicesModel,
    RectangleLabelsModel,
    RectangleLabelsObbModel,
    PolygonLabelsModel,
    KeypointLabelsModel,
    VideoRectangleModel,
    TimelineLabelsModel,
]


class YOLO(LabelStudioMLBase):
    """Label Studio ML Backend based on Ultralytics YOLO"""

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "yolo")

    def detect_control_models(self) -> List[ControlModel]:
        """Detect control models based on the labeling config.
        Control models are used to predict regions for different control tags in the labeling config.
        """
        control_models = []

        for control in self.label_interface.controls:
            # skipping tags without toName
            if not control.to_name:
                logger.warning(
                    f'{control.tag} {control.name} has no "toName" attribute, skipping it'
                )
                continue

            # match control tag with available control models
            for model_class in available_model_classes:
                if model_class.is_control_matched(control):
                    instance = model_class.create(self, control)
                    if not instance:
                        logger.debug(
                            f"No instance created for {control.tag} {control.name}"
                        )
                        continue
                    if not instance.label_map:
                        logger.error(
                            f"No label map built for the '{control.tag}' control tag '{instance.from_name}'.\n"
                            f"This indicates that your Label Studio config labels do not match the model's labels.\n"
                            f"To fix this, ensure that the 'value' or 'predicted_values' attribute "
                            f"in your Label Studio config matches one or more of these model labels.\n"
                            f"If you don't want to use this control tag for predictions, "
                            f'add `model_skip="true"` to it.\n'
                            f"Examples:\n"
                            f'  <Label value="Car"/>\n'
                            f'  <Label value="YourLabel" predicted_values="label1,label2"/>\n'
                            f"Labels provided in your labeling config:\n"
                            f"  {str(control.labels_attrs)}\n"
                            f"Available '{instance.model_path}' model labels:\n"
                            f"  {list(instance.model.names.values())}"
                        )
                        continue

                    control_models.append(instance)
                    logger.debug(f"Control tag with model detected: {instance}")
                    break

        if not control_models:
            control_tags = ", ".join([c.type for c in available_model_classes])
            raise ValueError(
                f"No suitable control tags (e.g. {control_tags} connected to Image or Video object tags) "
                f"detected in the label config:\n{self.label_config}"
            )

        return control_models

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Run YOLO predictions on the tasks
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions [Predictions array in JSON format]
            (https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(
            f"Run prediction on {len(tasks)} tasks, project ID = {self.project_id}"
        )
        control_models = self.detect_control_models()

        predictions = []
        for task in tasks:

            regions = []
            for model in control_models:
                path = model.get_path(task)
                regions += model.predict_regions(path)

            # calculate final score
            all_scores = [region["score"] for region in regions if "score" in region]
            avg_score = sum(all_scores) / max(len(all_scores), 1)

            # compose final prediction
            prediction = {
                "result": regions,
                "score": avg_score,
                "model_version": self.model_version,
            }
            predictions.append(prediction)

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated.
        Or it's called when "Start training" clicked on the model in the project settings.
        """
        results = {}
        control_models = self.detect_control_models()
        for model in control_models:
            training_result = model.fit(event, data, **kwargs)
            results[model.from_name] = training_result

        return results
