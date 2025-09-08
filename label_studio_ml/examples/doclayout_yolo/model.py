import os
import logging

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

# --- MODIFIED IMPORTS ---
# Only import the relevant control model
from control_models.rectangle_labels import RectangleLabelsModel
# Remove imports for other control models
from typing import List, Dict, Optional

# --- END MODIFICATION ---

logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)

# --- MODIFIED ---
# Register available model classes (only RectangleLabels for DocLayout-YOLO)
available_model_classes = [
    RectangleLabelsModel,
]
# --- END MODIFICATION ---


# --- MODIFIED CLASS NAME ---
class DocLayoutYOLO(LabelStudioMLBase):
# --- END MODIFICATION ---
    """Label Studio ML Backend based on DocLayout-YOLO (YOLOv10)"""

    def setup(self):
        """Configure any parameters of your model here"""
        # --- MODIFIED VERSION ---
        self.set("model_version", "DocLayout-YOLO")
        # --- END MODIFICATION ---
        logger.info(f"Model root directory: {os.getenv('MODEL_ROOT', './models')}")
        logger.info(f"Default model name: {os.getenv('MODEL_NAME', 'doclayout_yolo_docstructbench_imgsz1024.pt')}")
        logger.info(f"Default image size: {os.getenv('DEFAULT_IMGSZ', 1024)}")
        logger.info(f"Default score threshold: {os.getenv('MODEL_SCORE_THRESHOLD', 0.5)}")
        logger.info(f"Allow custom model path: {os.getenv('ALLOW_CUSTOM_MODEL_PATH', 'true')}")


    def detect_control_models(self) -> List[RectangleLabelsModel]: # Modified return type hint
        """Detect control models based on the labeling config.
        Control models are used to predict regions for different control tags in the labeling config.
        """
        control_models = []

        if not hasattr(self, 'label_interface') or not self.label_interface:
             logger.error("Label interface is not initialized.")
             return []

        logger.debug(f"Scanning label config: {self.label_config}")
        logger.debug(f"Available control tags: {[c.tag for c in self.label_interface.controls]}")

        for control in self.label_interface.controls:
             logger.debug(f"Checking control: tag={control.tag}, name={control.name}, toName={control.to_name}")
             # skipping tags without toName
             if not control.to_name:
                 logger.warning(
                     f'{control.tag} {control.name} has no "toName" attribute, skipping it'
                 )
                 continue

             # match control tag with available control models
             for model_class in available_model_classes:
                 if model_class.is_control_matched(control):
                     logger.info(f"Control tag '{control.name}' matched by {model_class.__name__}. Creating instance...")
                     try:
                         instance = model_class.create(self, control)
                         if not instance:
                             logger.debug(
                                 f"No instance created for {control.tag} {control.name} (e.g., model_skip=true)"
                             )
                             continue

                         # Check label map after creation
                         if not instance.label_map:
                             # This check might need refinement depending on whether a model always has names
                             # or if labels come solely from the config.
                             logger.warning(
                                 f"No label map built for the '{control.tag}' control tag '{instance.from_name}'. "
                                 f"This might happen if the model's output classes (from model.names) "
                                 f"do not overlap with the labels defined in the Label Studio config for this tag, "
                                 f"or if the model lacks a `.names` attribute. "
                                 f"Predictions might be empty or incorrect for this tag."
                                 # f"Config labels:\n{control.labels_attrs}\n"
                                 # f"Model labels:\n{list(instance.model.names.values() if hasattr(instance.model, 'names') else [])}"
                             )
                             # Decide whether to proceed without a map or skip
                             # For object detection, a map is usually crucial. Let's log a warning but proceed.
                             # Consider raising an error if a map is strictly required.

                         control_models.append(instance)
                         logger.info(f"Successfully created and added control model instance: {instance}")
                         break # Stop checking other model classes for this control
                     except Exception as e:
                         logger.error(f"Error creating control model instance for {control.name}: {e}", exc_info=True)
                         # Optionally re-raise or continue to the next control
                         continue # Skip this control if creation fails

        if not control_models:
            control_tags = ", ".join([c.type for c in available_model_classes])
            logger.error(
                f"No suitable control tags (e.g., {control_tags} connected to Image object tag) "
                f"found or successfully initialized in the label config:\n{self.label_config}"
            )
            # raise ValueError(...) # Optionally raise error if no controls are found

        logger.info(f"Detected {len(control_models)} control model(s).")
        return control_models

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Run DocLayout-YOLO predictions on the tasks
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions [Predictions array in JSON format]
            (https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(
            f"Received {len(tasks)} tasks for prediction, project ID = {self.project_id}"
        )
        # Detect control models ONCE based on the label config passed in the request,
        # or fall back to the one set during setup.
        label_config_from_request = kwargs.get('label_config') or self.label_config
        if label_config_from_request != self.label_config:
             logger.warning("Label config from request differs from setup config. Re-initializing label interface for this request.")
             self.label_interface = LabelInterface(label_config_from_request) # Re-init if different
        elif not hasattr(self, 'label_interface') or not self.label_interface:
             logger.warning("Label interface not found, initializing from label config.")
             self.label_interface = LabelInterface(self.label_config)

        control_models = self.detect_control_models()
        if not control_models:
             logger.error("No control models detected or initialized. Cannot perform prediction.")
             return ModelResponse(predictions=[])

        predictions = []
        for task in tasks:
            task_id = task.get('id', 'unknown')
            logger.info(f"Processing task ID: {task_id}")
            regions = []
            try:
                # --- MODIFIED - Iterate through detected control models ---
                for model in control_models:
                    image_path = model.get_path(task) # Get image path based on control's toName->value
                    logger.info(f"Predicting regions for task {task_id} using control '{model.from_name}' on image '{os.path.basename(image_path)}'")
                    try:
                        task_regions = model.predict_regions(image_path)
                        regions.extend(task_regions)
                    except Exception as e:
                         logger.error(f"Error predicting regions for task {task_id} with control '{model.from_name}': {e}", exc_info=True)
                # --- END MODIFICATION ---

                # Calculate final score (average of region scores)
                all_scores = [region["score"] for region in regions if "score" in region]
                avg_score = sum(all_scores) / max(len(all_scores), 1) if all_scores else 0

                # Compose final prediction for the task
                prediction = {
                    "result": regions,
                    "score": avg_score,
                    "model_version": self.model_version,
                }
                predictions.append(prediction)
                logger.info(f"Finished processing task ID: {task_id}, found {len(regions)} regions.")

            except Exception as e:
                 logger.error(f"Failed to process task ID {task_id}: {e}", exc_info=True)
                 # Append an empty prediction or skip the task
                 predictions.append({
                    "result": [],
                    "score": 0,
                    "model_version": self.model_version,
                 })


        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated.
        DocLayout-YOLO example does not implement training.
        """
        logger.warning("Training (fit method) is not implemented for this DocLayout-YOLO backend.")
        return {} # Return empty dict to indicate no training occurred