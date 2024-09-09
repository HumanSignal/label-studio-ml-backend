import logging

from control_models.base import ControlModel
from control_models.rectangle_labels import is_obb
from typing import List, Dict
from label_studio_sdk.converter.utils import convert_yolo_obb_to_annotation


logger = logging.getLogger(__name__)


class RectangleLabelsObbModel(ControlModel):
    """
    Class representing a RectangleLabels OBB
    (oriented bounding boxes, rotated bounding boxes)
    control tag for YOLO model.
    """

    type = "RectangleLabels"
    model_path = "yolov8n-obb.pt"

    @classmethod
    def is_control_matched(cls, control) -> bool:
        # check object tag type
        if control.objects[0].tag != "Image":
            return False
        if not is_obb(control):
            return False
        return control.tag == cls.type

    def predict_regions(self, path) -> List[Dict]:
        results = self.model.predict(path)
        self.debug_plot(results[0].plot())

        # simple bounding boxes without rotation
        if results[0].obb is None:
            raise ValueError(
                "Simple bounding boxes are detected in the YOLO model results. "
                'However, `model_obb="true"` is set at the RectangleLabels tag '
                "in the labeling config. Set it to `false` to use simple bounding boxes."
            )

        # oriented bounding boxes with rotation (yolo obb model)
        return self.create_rotated_rectangles(results, path)

    def create_rotated_rectangles(self, results, path):
        """YOLO OBB: oriented bounding boxes"""
        logger.debug(f"create_rotated_rectangles: {self.from_name}")
        data = results[0].obb  # take bboxes from the first frame
        regions = []

        for i in range(data.shape[0]):  # iterate over items
            score = float(data.conf[i])  # tensor => float
            model_label = self.model.names[int(data.cls[i])]
            original_height, original_width = data.orig_shape
            value = convert_yolo_obb_to_annotation(
                data.xyxyxyxy[i].tolist(), original_width, original_height
            )

            logger.debug(
                "----------------------\n"
                f"task id > {path}\n"
                f"type: {self.control}\n"
                f"x, y, w, h, r > {value}\n"
                f"model label > {model_label}\n"
                f"score > {score}\n"
            )

            # bbox score is too low
            if score < self.model_score_threshold:
                continue

            # there is no mapping between model label and LS label
            if model_label not in self.label_map:
                continue
            output_label = self.label_map[model_label]
            value["rectanglelabels"] = [output_label]

            # add new region with rectangle
            region = {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "value": value,
                "score": score,
            }
            regions.append(region)
        return regions


# pre-load and cache default model at startup
RectangleLabelsObbModel.get_cached_model(RectangleLabelsObbModel.model_path)
