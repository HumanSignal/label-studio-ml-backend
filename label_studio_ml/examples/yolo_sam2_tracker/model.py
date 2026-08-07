import logging
# from PIL import Image
# YOLO + SAM2 related imports
from typing import List, Dict
from typing import Optional

# YOLO imports:
from control_models.base import ControlModel
from control_models.video_rectangle_with_yolo_sam2_tracker import VideoRectangleWithYOLOSAM2TrackerModel
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

# Register available model classes
available_model_classes = [
    VideoRectangleWithYOLOSAM2TrackerModel,
]

logger = logging.getLogger(__name__)

class YoloSamMultiObjectTracking(LabelStudioMLBase):
    """
    YOLO_SAM model for object detection and tracking.
    Detection model based on YOLO and tracking based on Segment Anything 2.
    """

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "yolo_sam")

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

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> None:
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

        path = None
        regions = None # regions for the video
        predictions = []
        for task in tasks:
            logger.info(f"Processing task: {task}")
            logger.info(f"Processing control models: {len(control_models)}")
            for model in control_models:
                logger.info(f"Processing model: {model}")
                path = model.get_path(task)
                regions = model.predict_regions(path)
                break

            # calculate final score
            total_score = sum(region["score"] * len(region["value"]["sequence"]) for region in regions if "score" in region)
            total_frames = sum(len(region["value"]["sequence"]) for region in regions if "score" in region)
            avg_score = total_score / max(total_frames, 1)

            # compose final prediction
            prediction = {
                "result": regions,
                "score": avg_score,
                "model_version": self.model_version,
            }
            predictions.append(prediction)
            logger.debug(f"Generated {len(regions)} regions with avg_score {avg_score:.3f}")

        return ModelResponse(predictions=predictions)