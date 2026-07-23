import os
import logging

from pydantic import BaseModel
from typing import Optional, List, Dict, ClassVar
from ultralytics import YOLO

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.label_interface.control_tags import ControlTag
from label_studio_sdk.label_interface import LabelInterface


# use matplotlib plots for debug
DEBUG_PLOT = os.getenv("DEBUG_PLOT", "false").lower() in ["1", "true"]
MODEL_SCORE_THRESHOLD = float(os.getenv("MODEL_SCORE_THRESHOLD", 0.5))
DEFAULT_MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_ROOT = os.getenv("MODEL_ROOT", DEFAULT_MODEL_ROOT)
os.makedirs(MODEL_ROOT, exist_ok=True)
# if true, allow to use custom model path from the control tag in the labeling config
ALLOW_CUSTOM_MODEL_PATH = os.getenv("ALLOW_CUSTOM_MODEL_PATH", "true").lower() in [
    "1",
    "true",
]

# Global cache for YOLO models
_model_cache = {}
logger = logging.getLogger(__name__)


def get_bool(attr, attr_name, default="false"):
    return attr.get(attr_name, default).lower() in ["1", "true", "yes"]


class ControlModel(BaseModel):
    """
    Represents a control tag in Label Studio, which is associated with a specific type of labeling task
    and is used to generate predictions using a YOLO model.

    Attributes:
        type (str): Type of the control, e.g., RectangleLabels, Choices, etc.
        control (ControlTag): The actual control element from the Label Studio configuration.
        from_name (str): The name of the control tag, used to link the control to the data.
        to_name (str): The name of the data field that this control is associated with.
        value (str): The value name from the object that this control operates on, e.g., an image or text field.
        model (object): The model instance (e.g., YOLO) used to generate predictions for this control.
        model_path (str): Path to the YOLO model file.
        model_score_threshold (float): Threshold for prediction scores; predictions below this value will be ignored.
        label_map (Optional[Dict[str, str]]): A mapping of model labels to Label Studio labels.
    """

    type: ClassVar[str]
    control: ControlTag
    from_name: str
    to_name: str
    value: str
    model: YOLO
    model_path: ClassVar[str]
    model_score_threshold: float = 0.5
    label_map: Optional[Dict[str, str]] = {}
    label_studio_ml_backend: LabelStudioMLBase
    project_id: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def is_control_matched(cls, control) -> bool:
        """Check if the control tag matches the model type.
        Args:
            control (ControlTag): The control tag from the Label Studio Interface.
        """
        raise NotImplementedError("This method should be overridden in derived classes")

    @staticmethod
    def get_from_name_for_label_map(
        label_interface: LabelInterface, target_name: str
    ) -> str:
        """Get the 'from_name' attribute for the label map building."""
        return target_name

    @classmethod
    def create(cls, mlbackend: LabelStudioMLBase, control: ControlTag):
        """Factory method to create an instance of a specific control model class.
        Args:
            mlbackend (LabelStudioMLBase): The ML backend instance.
            control (ControlTag): The control tag from the Label Studio Interface.
        """
        from_name = control.name
        to_name = control.to_name[0]
        value = control.objects[0].value_name

        # if skip is true, don't process this control
        if get_bool(control.attr, "model_skip", "false"):
            logger.info(
                f"Skipping control tag '{control.tag}' with name '{from_name}', model_skip=true found"
            )
            return None
        # read threshold attribute from the control tag, e.g.: <RectangleLabels model_score_threshold="0.5">
        model_score_threshold = float(
            control.attr.get("model_score_threshold")
            or control.attr.get(
                "score_threshold"
            )  # not recommended option, use `model_score_threshold`
            or MODEL_SCORE_THRESHOLD
        )
        # read `model_path` attribute from the control tag
        model_path = (
            ALLOW_CUSTOM_MODEL_PATH and control.attr.get("model_path")
        ) or cls.model_path

        model = cls.get_cached_model(model_path)
        model_names = model.names.values()  # class names from the model
        # from_name for label mapping can be differed from control.name (e.g. VideoRectangle)
        label_map_from_name = cls.get_from_name_for_label_map(
            mlbackend.label_interface, from_name
        )
        label_map = mlbackend.build_label_map(label_map_from_name, model_names)

        return cls(
            control=control,
            from_name=from_name,
            to_name=to_name,
            value=value,
            model=model,
            model_score_threshold=model_score_threshold,
            label_map=label_map,
            label_studio_ml_backend=mlbackend,
            project_id=mlbackend.project_id,
        )

    @classmethod
    def load_yolo_model(cls, filename) -> YOLO:
        """Load YOLO model from the file."""
        path = os.path.join(MODEL_ROOT, filename)
        logger.info(f"Loading yolo model: {path}")
        model = YOLO(path)
        logger.info(f"Model {path} names:\n{model.names}")
        return model

    @classmethod
    def get_cached_model(cls, path: str) -> YOLO:
        if path not in _model_cache:
            _model_cache[path] = cls.load_yolo_model(path)
        return _model_cache[path]

    def debug_plot(self, image):
        if not DEBUG_PLOT:
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.imshow(image[..., ::-1])
        plt.axis("off")
        plt.title(self.type)
        plt.show()

    def predict_regions(self, path) -> List[Dict]:
        """Predict regions in the image using the YOLO model.
        Args:
            path (str): Path to the file with media
        """
        raise NotImplementedError("This method should be overridden in derived classes")

    def fit(self, event, data, **kwargs):
        """Fit the model."""
        logger.warning("The fit method is not implemented for this control model")
        return False

    def get_path(self, task):
        task_path = task["data"].get(self.value) or task["data"].get(
            DATA_UNDEFINED_NAME
        )
        if task_path is None:
            raise ValueError(
                f"Can't load path using key '{self.value}' from task {task}"
            )
        if not isinstance(task_path, str):
            raise ValueError(f"Path should be a string, but got {task_path}")

        # try path as local file or try to load it from Label Studio instance/download via http
        path = (
            task_path
            if os.path.exists(task_path)
            else get_local_path(task_path, task_id=task.get("id"))
        )
        logger.debug(f"load_image: {task_path} => {path}")
        return path

    def __str__(self):
        """Return a string with full representation of the control tag."""
        return (
            f"{self.type} from_name={self.from_name}, "
            f"label_map={self.label_map}, model_score_threshold={self.model_score_threshold}"
        )

    class Config:
        arbitrary_types_allowed = True
        protected_namespaces = ("__.*__", "_.*")  # Excludes 'model_'
