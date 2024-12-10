import logging

from control_models.base import ControlModel, get_bool
from typing import List, Dict
from label_studio_sdk.label_interface.control_tags import ControlTag


logger = logging.getLogger(__name__)


def is_obb(control: ControlTag) -> bool:
    """Check if the model should use oriented bounding boxes (OBB)
    based on the control tag attribute `model_obb` from the labeling config.
    """
    return get_bool(control.attr, "model_obb", "false")


class RectangleLabelsModel(ControlModel):
    """
    Class representing a RectangleLabels (bounding boxes) control tag for YOLO model.
    """

    type = "RectangleLabels"
    model_path = "yolov8m.pt"

    @classmethod
    def is_control_matched(cls, control) -> bool:
        # check object tag type
        if control.objects[0].tag != "Image":
            return False
        if is_obb(control):
            return False
        return control.tag == cls.type

    def predict_regions(self, path) -> List[Dict]:
        results = self.model.predict(path)
        self.debug_plot(results[0].plot())

        # oriented bounding boxes are detected, but it should be processed by RectangleLabelsObbModel
        if results[0].obb is not None and results[0].boxes is None:
            raise ValueError(
                "Oriented bounding boxes are detected in the YOLO model results. "
                'However, `model_obb="true"` is not set at the RectangleLabels tag '
                "in the labeling config."
            )

        # simple bounding boxes without rotation
        return self.create_rectangles(results, path)

    def create_rectangles(self, results, path):
        """Simple bounding boxes without rotation"""
        logger.debug(f"create_rectangles: {self.from_name}")
        data = results[0].boxes  # take bboxes from the first frame
        model_names = self.model.names
        regions = []

        for i in range(data.shape[0]):  # iterate over items
            score = float(data.conf[i])  # tensor => float
            x, y, w, h = data.xywhn[i].tolist()
            model_label = model_names[int(data.cls[i])]

            logger.debug(
                "----------------------\n"
                f"task id > {path}\n"
                f"type: {self.control}\n"
                f"x, y, w, h > {x, y, w, h}\n"
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

            # add new region with rectangle
            region = {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [output_label],
                    "x": (x - w / 2) * 100,
                    "y": (y - h / 2) * 100,
                    "width": w * 100,
                    "height": h * 100,
                },
                "score": score,
            }
            regions.append(region)
        return regions


# pre-load and cache default model at startup
RectangleLabelsModel.get_cached_model(RectangleLabelsModel.model_path)
