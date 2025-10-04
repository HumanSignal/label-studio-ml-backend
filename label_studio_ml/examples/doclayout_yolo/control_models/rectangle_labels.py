import logging
import torch # Added
import os # Added

from .base import ControlModel, get_bool # Relative import
from typing import List, Dict
from label_studio_sdk.label_interface.control_tags import ControlTag


logger = logging.getLogger(__name__)


def is_obb(control: ControlTag) -> bool:
    """Check if the model should use oriented bounding boxes (OBB)
    based on the control tag attribute `model_obb` from the labeling config.
    DocLayout-YOLO likely doesn't support OBB by default, so this might always be false.
    """
    # DocLayout-YOLO is not specified to support OBB, assume false
    # return get_bool(control.attr, "model_obb", "false")
    return False


class RectangleLabelsModel(ControlModel):
    """
    Class representing a RectangleLabels (bounding boxes) control tag for DocLayout-YOLO model.
    """

    type = "RectangleLabels"
    # Default model path relative to MODEL_ROOT - user should change this or use env var/config attribute
    # model_path = "doclayout_yolo_docstructbench_imgsz1024.pt" # This is now handled in base.py

    @classmethod
    def is_control_matched(cls, control) -> bool:
        # check object tag type
        if control.objects[0].tag != "Image":
             logger.debug(f"Object tag is not Image for {control.name}")
             return False
        if is_obb(control):
             logger.debug(f"OBB is set for {control.name}, but not supported by this model.")
             return False
        if control.tag != cls.type:
             logger.debug(f"Control tag is not {cls.type} for {control.name}")
             return False
        logger.debug(f"Control tag {control.name} matched to {cls.__name__}")
        return True


    def predict_regions(self, image_path) -> List[Dict]:
        """Run DocLayout-YOLO prediction and return regions"""
        if self.model is None:
            logger.error("Model is not loaded.")
            return []

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        try:
            # --- MODIFIED PREDICTION CALL ---
            # Use parameters stored in the ControlModel instance
            logger.info(f"Running prediction on {os.path.basename(image_path)} with imgsz={self.model_imgsz}, conf={self.model_score_threshold}")
            results = self.model.predict(
                image_path,
                imgsz=self.model_imgsz,
                conf=self.model_score_threshold,
                device=device
                # Add other relevant prediction params if needed (e.g., iou, augment)
            )
            # --- END MODIFICATION ---

            # --- DEBUG PLOTTING ---
            # The result object itself might be plottable
            if results and len(results) > 0:
                 self.debug_plot(results[0])
            # --- END DEBUG ---

            # Check if results are in the expected format (list of result objects)
            if not results or not isinstance(results, list) or len(results) == 0:
                 logger.warning(f"Prediction yielded no results for {image_path}")
                 return []

            # Assume the first element contains the detections for the image
            det_result = results[0]

            # --- ADAPTED RESULT PROCESSING ---
            # IMPORTANT: Verify the structure of the result object returned by doclayout_yolo.YOLOv10.predict
            # We assume it's similar to ultralytics results and has `.boxes` and `.names` (via self.model.names)
            # It must have `.boxes` attribute which contains detections.
            if not hasattr(det_result, 'boxes') or det_result.boxes is None:
                 logger.warning(f"Result object for {image_path} has no '.boxes' attribute or it's None. Cannot extract detections.")
                 return []

            data = det_result.boxes # This holds the bounding box data
            # Need class names mapping from ID to string
            if not hasattr(self.model, 'names') or not self.model.names:
                 logger.error("Model does not have a 'names' attribute for class mapping. Cannot create regions.")
                 return []
            model_names = self.model.names # e.g., {0: 'text', 1: 'title', ...}

            regions = []
            # Check if boxes tensor is empty
            if data.shape[0] == 0:
                 logger.info(f"No bounding boxes detected in {image_path} above threshold {self.model_score_threshold}")
                 return []

            # Iterate through detected boxes
            # Expect data.conf, data.xywhn, data.cls to exist
            if not all(hasattr(data, attr) for attr in ['conf', 'xywhn', 'cls']):
                 logger.error("Detection result's '.boxes' attribute is missing one or more required attributes: 'conf', 'xywhn', 'cls'.")
                 return []

            image_width = det_result.orig_shape[1]
            image_height = det_result.orig_shape[0]

            for i in range(data.shape[0]):  # iterate over items
                 score = float(data.conf[i])  # Confidence score
                 class_id = int(data.cls[i])  # Class ID

                 # Map class ID to model label name
                 if class_id not in model_names:
                     logger.warning(f"Detected class ID {class_id} not found in model names map: {model_names}. Skipping.")
                     continue
                 model_label = model_names[class_id]

                 # Map model label to Label Studio label using the generated label_map
                 if model_label not in self.label_map:
                     # logger.warning(f"Model label '{model_label}' not found in Label Studio label map: {self.label_map}. Skipping.")
                     continue # Don't warn for every box, might be too verbose
                 output_label = self.label_map[model_label]

                 # Get normalized coordinates [x_center, y_center, width, height]
                 x, y, w, h = data.xywhn[i].tolist()

                 # Convert normalized xywh to LS format (top-left x, y, width, height in percentages)
                 ls_x = (x - w / 2) * 100
                 ls_y = (y - h / 2) * 100
                 ls_w = w * 100
                 ls_h = h * 100

                 # Ensure coordinates are within bounds [0, 100]
                 ls_x = max(0, min(ls_x, 100))
                 ls_y = max(0, min(ls_y, 100))
                 ls_w = max(0, min(ls_w, 100 - ls_x)) # Adjust width based on x
                 ls_h = max(0, min(ls_h, 100 - ls_y)) # Adjust height based on y


                 # Skip boxes below the specific threshold for this control tag
                 if score < self.model_score_threshold:
                      continue

                 regions.append({
                      "from_name": self.from_name,
                      "to_name": self.to_name,
                      "type": self.type.lower(), # "rectanglelabels"
                      "value": {
                          "rectanglelabels": [output_label],
                          "x": ls_x,
                          "y": ls_y,
                          "width": ls_w,
                          "height": ls_h,
                      },
                      "score": score,
                      "original_width": image_width, # Add original image size for context
                      "original_height": image_height
                 })
                 logger.debug(
                     f"Added region: label={output_label}, score={score:.2f}, "
                     f"xywh%=[{ls_x:.1f}, {ls_y:.1f}, {ls_w:.1f}, {ls_h:.1f}]"
                 )

            # --- END ADAPTATION ---
            logger.info(f"Predicted {len(regions)} regions for {os.path.basename(image_path)}")
            return regions

        except FileNotFoundError:
            logger.error(f"Image file not found at {image_path}")
            return []
        except Exception as e:
            logger.error(f"Error during prediction for {image_path}: {e}", exc_info=True)
            return []

# --- REMOVED ---
# # pre-load and cache default model at startup
# # This is now handled dynamically in base.py when needed
# RectangleLabelsModel.get_cached_model(RectangleLabelsModel.model_path)
# --- END REMOVED ---