import logging
from typing import List, Dict
from uuid import uuid4

import cv2
from label_studio_sdk.converter.brush import mask2rle

from control_models.base import ControlModel

logger = logging.getLogger(__name__)


class BrushLabelsModel(ControlModel):
    """
    Class representing a BrushLabels control tag for YOLO model.
    """

    type = "BrushLabels"
    model_path = "yolov8n-seg.pt"

    @classmethod
    def is_control_matched(cls, control) -> bool:
        # check object tag type
        if control.objects[0].tag != "Image":
            return False
        return control.tag == cls.type

    def predict_regions(self, path) -> List[Dict]:
        results = self.model.predict(path)
        return self.create_brush(results, path)

    def create_brush(self, results, path):
        logger.debug(f"create_brush: {self.from_name}")
        data = results[0].masks
        model_names = self.model.names
        height, width = data.orig_shape

        regions = []
        for i in range(len(data)):
            label_id = str(uuid4())[:9]
            score = float(results[0].boxes.conf[i])
            mask = (
                cv2.resize(data.data[i].numpy(), results[0].orig_shape[::-1]) > 0
            ).astype("uint8") * 255
            rle = mask2rle(mask)
            model_label = model_names[int(results[0].boxes.cls[i])]

            logger.debug(
                "----------------------\n"
                f"task id > {path}\n"
                f"type: {self.control}\n"
                f"rle > {rle}\n"
                f"model label > {model_label}\n"
                f"score > {score}\n"
            )

            if score < self.model_score_threshold:
                continue

            if model_label not in self.label_map:
                continue
            output_label = self.label_map[model_label]

            region = {
                "id": label_id,
                "from_name": self.from_name,
                "to_name": self.to_name,
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": [output_label],
                },
                "score": score,
                "type": "brushlabels",
            }
            regions.append(region)
        return regions


# pre-load and cache default model at startup
BrushLabelsModel.get_cached_model(BrushLabelsModel.model_path)
