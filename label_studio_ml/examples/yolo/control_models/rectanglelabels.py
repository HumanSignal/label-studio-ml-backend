import math
import logging
import numpy as np

from control_models.base import ControlModel
from typing import List, Dict


logger = logging.getLogger(__name__)


class RectangleLabelsModel(ControlModel):
    """
    Class representing a RectangleLabels (bounding boxes) control tag for YOLO model.
    """
    type = 'RectangleLabels'
    model_path = 'yolov8n-obb.pt'

    @classmethod
    def is_control_matched(cls, control) -> bool:
        # check object tag type
        if control.objects[0].tag != 'Image':
            return False
        return control.tag == cls.type

    def predict_regions(self, path) -> List[Dict]:
        results = self.model.predict(path)
        self.debug_plot(results[0].plot())

        # simple bounding boxes without rotation
        if results[0].obb is None:
            return self.create_rectangles(results, path)

        # oriented bounding boxes with rotation (yolo obb model)
        else:
            return self.create_rotated_rectangles(results, path)
 
    def create_rectangles(self, results, path):
        """ Simple bounding boxes without rotation
        """
        logger.debug(f'create_rectangles: {self.from_name}')
        data = results[0].boxes  # take bboxes from the first frame
        regions = []

        for i in range(data.shape[0]):  # iterate over items
            score = float(data.conf[i])  # tensor => float
            x, y, w, h = data.xywhn[i].tolist()
            model_label = self.model.names[int(data.cls[i])]

            logger.debug(
                "----------------------\n"
                f"task id > {path}\n"
                f"type: {self.control}\n"
                f"x, y, w, h > {x, y, w, h}\n"
                f"model label > {model_label}\n"
                f"score > {score}\n"
            )

            # bbox score is too low
            if score < self.score_threshold:
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

    @staticmethod
    def convert_yolo_obb_to_annotation(coords, original_width, original_height):
        """
        Convert Yolo OBB format to LS annotation format.

        Args:
            coords (list of tuple): List of tuples containing the coordinates of the object in Yolo OBB format.
                Each tuple represents a corner of the bounding box in the order:
                (top-left, top-right, bottom-right, bottom-left).
            original_width (int): Original width of the image.
            original_height (int): Original height of the image.

        Returns:
            dict: Dictionary containing annotation information including:
                - x (float): X-coordinate of the top-left corner of the object in percentage of the original width.
                - y (float): Y-coordinate of the top-left corner of the object in percentage of the original height.
                - width (float): Width of the object in percentage of the original width.
                - height (float): Height of the object in percentage of the original height.
                - rotation (float): Rotation angle of the object in degrees.
        """
        coords = [(x * original_width, y * original_height) for x, y in coords]

        # Top-left corner (x, y)
        x, y = coords[0]

        # Width and height are distances between points
        w = math.sqrt((coords[1][0] - coords[0][0]) ** 2 + (coords[1][1] - coords[0][1]) ** 2)
        h = math.sqrt((coords[3][0] - coords[0][0]) ** 2 + (coords[3][1] - coords[0][1]) ** 2)

        # Calculate rotation angle
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        rotation = math.degrees(math.atan2(dy, dx))

        return {
            "original_width": original_width,
            "original_height": original_height,
            "x": (x - w/2)/ original_width * 100,
            "y": (y - h/2) / original_height * 100,
            "width": w / original_width * 100,
            "height": h / original_height * 100,
            "rotation": rotation,
        }

    # TODO: THIS FUNCTION IS STILL IN PROGRESS
    def create_rotated_rectangles(self, results, path):
        """ YOLO OBB: oriented bounding boxes
        """
        logger.debug(f'create_rotated_rectangles: {self.from_name}')
        data = results[0].obb  # take bboxes from the first frame
        regions = []

        for i in range(data.shape[0]):  # iterate over items
            score = float(data.conf[i])  # tensor => float
            x, y, w, h, r = data.xywhr[i].tolist()
            original_height, original_width = data.orig_shape
            # x, y, w, h = (
            #     x / data.orig_shape[1], y / data.orig_shape[0],
            #     w / data.orig_shape[1], h / data.orig_shape[0]
            # )
            value = self.convert_yolo_obb_to_annotation(
                data.xyxyxyxyn[i].tolist(), data.orig_shape[0], data.orig_shape[1]
            )
            model_label = self.model.names[int(data.cls[i])]

            logger.debug(
                "----------------------\n"
                f"task id > {path}\n"
                f"type: {self.control}\n"
                f"x, y, w, h, r > {value}\n"
                f"model label > {model_label}\n"
                f"score > {score}\n"
            )

            # bbox score is too low
            if score < self.score_threshold:
                continue

            # there is no mapping between model label and LS label
            if model_label not in self.label_map:
                continue
            output_label = self.label_map[model_label]
            value = {
                "original_width": original_width,
                "original_height": original_height,
                "x": (x - w/2) / original_width * 100,
                "y": (y - h/2) / original_height * 100,
                "width": w / original_width * 100,
                "height": h / original_height * 100,
                "rotation": r * 180 / np.pi,
            }
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
RectangleLabelsModel.get_cached_model(RectangleLabelsModel.model_path)
