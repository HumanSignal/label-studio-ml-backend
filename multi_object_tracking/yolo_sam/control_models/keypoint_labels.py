import logging
from control_models.base import ControlModel, get_bool
from typing import List, Dict

logger = logging.getLogger(__name__)


class KeypointLabelsModel(ControlModel):
    """
    Class representing a KeypointLabels control tag for YOLO model.
    """

    type = "KeyPointLabels"
    model_path = (
        "yolov8n-pose.pt"  # Adjust the model path to your keypoint detection model
    )
    add_bboxes: bool = True
    point_size: float = 1
    point_threshold: float = 0
    point_map: Dict = {}

    def __init__(self, **data):
        super().__init__(**data)

        self.add_bboxes = get_bool(self.control.attr, "model_add_bboxes", "true")
        self.point_size = float(self.control.attr.get("model_point_size", 1))
        self.point_threshold = float(self.control.attr.get("model_point_threshold", 0))
        self.point_map = self.build_point_mapping()

    @classmethod
    def is_control_matched(cls, control) -> bool:
        # Check object tag type
        if control.objects[0].tag != "Image":
            return False
        return control.tag == cls.type

    def build_point_mapping(self):
        """Build a mapping between points and Label Studio labels, e.g.
        <Label value="nose" predicted_values="person" model_index="0" /> => {"person::0": "nose"}
        """
        mapping = {}
        for value, label_tag in self.control.labels_attrs.items():
            model_name = label_tag.attr.get("predicted_values")
            model_index = label_tag.attr.get("model_index")
            if model_name and not model_index:
                logger.warning(
                    f"`model_index` is not provided for Label tag: {label_tag}"
                )
            if not model_name and model_index:
                logger.warning(
                    f"`predicted_values` is not provided for Label tag: {label_tag}"
                )
            if model_name and model_index:
                mapping[f"{model_name}::{model_index}"] = value

        if not mapping:
            logger.error(
                f"No point to label mapping found for control tag: {self.control}"
            )
        return mapping

    def predict_regions(self, path) -> List[Dict]:
        results = self.model.predict(path)
        return self.create_keypoints(results, path)

    def create_keypoints(self, results, path):
        logger.debug(f"create_keypoints: {self.from_name}")
        keypoints_data = results[0].keypoints  # Get keypoints from the first frame
        bbox_data = results[0].boxes
        image_width = results[0].orig_shape[1]
        model_names = self.model.names
        regions = []

        for bbox_index in range(
            keypoints_data.shape[0]
        ):  # Iterate over detected bboxes
            bbox_conf = bbox_data.conf[bbox_index]
            point_xyn = (
                keypoints_data.xyn[bbox_index] * 100
            )  # Convert normalized keypoints to percentages
            model_label = model_names[int(results[0].boxes.cls[bbox_index])]

            point_logs = "\n".join(
                [f' model_index="{i}", xy={xyn}' for i, xyn in enumerate(point_xyn)]
            )
            logger.debug(
                "----------------------\n"
                f"task id > {path}\n"
                f"type: {self.control}\n"
                f"model label > {model_label}\n"
                f"keypoints >\n{point_logs}\n"
                f"confidences > {bbox_conf}\n"
            )

            # bbox score is too low
            if bbox_conf < self.model_score_threshold:
                continue

            # There is no mapping between model label and LS label
            if model_label not in self.label_map:
                continue

            # Add parent bbox that contains all keypoints
            if self.add_bboxes:
                region = self.create_bounding_box(
                    bbox_conf, bbox_data, bbox_index, model_label
                )
                regions.append(region)

            for point_index, xyn in enumerate(point_xyn):
                point_conf = keypoints_data.conf[bbox_index][point_index]
                if point_conf < self.point_threshold:
                    continue

                x, y = xyn.tolist()
                index_name = f"{model_label}::{point_index}"
                if index_name not in self.point_map:
                    logger.warning(
                        f"Point {index_name} not found in point map, "
                        f"you have to define it in the labeling config, e.g.:\n"
                        f'<Label value="nose" predicted_values="person" model_index="0" />'
                    )
                    continue
                point_label = self.point_map[index_name]

                # Add new region with keypoint
                region = {
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "keypointlabels",
                    "value": {
                        # point label
                        "keypointlabels": [point_label],
                        # point width, just visual styling
                        "width": self.point_size / image_width * 100,
                        "x": x,
                        "y": y,
                    },
                    "meta": {
                        "text": [f"bbox-{bbox_index}"]  # Group keypoints by bbox index
                    },
                    "score": float(point_conf),
                }
                # If bboxes are used, group keypoints by bbox
                if self.add_bboxes:
                    region["parentID"] = f"bbox-{bbox_index}"
                regions.append(region)
        return regions

    def create_bounding_box(self, bbox_conf, bbox_data, bbox_index, model_label):
        # Add parent bbox that contains all keypoints
        x, y, w, h = bbox_data.xywhn[bbox_index].tolist()
        region = {
            "id": f"bbox-{bbox_index}",
            "from_name": self.from_name + "_bbox",
            "to_name": self.to_name,
            "type": "rectanglelabels",
            "value": {
                "rectanglelabels": [model_label],
                "x": (x - w / 2) * 100,
                "y": (y - h / 2) * 100,
                "width": w * 100,
                "height": h * 100,
            },
            "meta": {"text": [f"bbox-{bbox_index}"]},  # Group keypoints by bbox index
            "score": float(bbox_conf),
            "hidden": True,
        }
        return region


# Pre-load and cache default model at startup
KeypointLabelsModel.get_cached_model(KeypointLabelsModel.model_path)
