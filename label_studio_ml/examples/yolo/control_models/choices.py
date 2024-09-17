import logging
import numpy as np

from control_models.base import ControlModel
from typing import List, Dict


logger = logging.getLogger(__name__)


class ChoicesModel(ControlModel):
    """
    Class representing a Choices (classes) control tag for YOLO model.
    """

    type = "Choices"
    model_path = "yolov8n-cls.pt"

    @classmethod
    def is_control_matched(cls, control) -> bool:
        # check object tag type
        if control.objects[0].tag != "Image":
            return False
        # support both Choices and Taxonomy because of their similarity
        return control.tag in [cls.type, "Taxonomy"]

    def predict_regions(self, path) -> List[Dict]:
        results = self.model.predict(path)
        self.debug_plot(results[0].plot())
        return self.create_choices(results, path)

    def create_choices(self, results, path):
        logger.debug(f"create_choices: {self.from_name}")
        mode = self.control.attr.get("choice", "single")
        data = results[0].probs.data.cpu().numpy()

        # single
        if mode in ["single", "single-radio"]:
            # we must keep data items that matches label_map only, because we need to search among label_map only
            indexes = [
                i for i, name in self.model.names.items() if name in self.label_map
            ]
            data = data[indexes]
            model_names = [self.model.names[i] for i in indexes]
            # find the best choice
            index = np.argmax(data)
            probs = [data[index]]
            names = [model_names[index]]
        # multi
        else:
            # get indexes of data where data >= self.model_score_threshold
            indexes = np.where(data >= self.model_score_threshold)
            probs = data[indexes].tolist()
            names = [self.model.names[int(i)] for i in indexes[0]]

        if not probs:
            logger.debug("No choices found")
            return []

        score = np.mean(probs)
        logger.debug(
            "----------------------\n"
            f"task id > {path}\n"
            f"control: {self.control}\n"
            f"probs > {probs}\n"
            f"score > {score}\n"
            f"names > {names}\n"
        )

        if score < self.model_score_threshold:
            logger.debug(f"Score is too low for single choice: {names[0]} = {probs[0]}")
            return []

        # map to Label Studio labels
        output_labels = [
            self.label_map[name] for name in names if name in self.label_map
        ]

        # add new region with rectangle
        return [
            {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "choices",
                "value": {"choices": output_labels},
                "score": float(score),
            }
        ]


# pre-load and cache default model at startup
ChoicesModel.get_cached_model(ChoicesModel.model_path)
