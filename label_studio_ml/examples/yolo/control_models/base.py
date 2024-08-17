import os
import matplotlib.pyplot as plt

from pydantic import BaseModel
from typing import Optional, List, Dict, ClassVar
from ultralytics import YOLO

from label_studio_ml.model import LabelStudioMLBase
from label_studio_sdk.label_interface.control_tags import ControlTag


# use matplotlib plots for debug
DEBUG_PLOT = os.getenv('DEBUG_PLOT', "false").lower() in ["1", "true"]
SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', 0.5))


class ControlModel(BaseModel):
    """
    Represents a control tag in Label Studio, which is associated with a specific type of data
    and is used to generate predictions using a YOLO model.

    Attributes:
        type (str): Type of the control, e.g., RectangleLabels, Choices, etc.
        control (ControlTag): The actual control element from the Label Studio configuration.
        from_name (str): The name of the control tag, used to link the control to the data.
        to_name (str): The name of the data field that this control is associated with.
        value (str): The value name from the object that this control operates on, e.g., an image or text field.
        model (object): The model instance (e.g., YOLO) used to generate predictions for this control.
        score_threshold (float): Threshold for prediction scores; predictions below this value will be ignored.
        label_map (Optional[Dict[str, str]]): A mapping of model labels to Label Studio labels.
    """
    type: ClassVar[str]
    control: ControlTag
    from_name: str
    to_name: str
    value: str
    model: YOLO
    score_threshold: float = 0.5
    label_map: Optional[Dict[str, str]] = {}
    label_studio_ml_backend: LabelStudioMLBase

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def is_control_matched(cls, control) -> bool:
        """ Check if the control tag matches the model type.
        Args:
            control (ControlTag): The control tag from the Label Studio Interface.
        """
        raise NotImplementedError("This method should be overridden in derived classes")

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

        model = cls.load_yolo_model()
        model_names = model.names.values()
        label_map = mlbackend.build_label_map(from_name, model_names)
        
        # read `score_threshold` attribute from the control tag, e.g.: <RectangleLabels score_threshold="0.5">
        score_threshold = float(control.attr.get('score_threshold') or SCORE_THRESHOLD)

        return cls(
            control=control,
            from_name=from_name,
            to_name=to_name,
            value=value,
            model=model,
            score_threshold=score_threshold,
            label_map=label_map,
            label_studio_ml_backend=mlbackend
        )

    @classmethod
    def load_yolo_model(cls) -> YOLO:
        """This method should be overridden in child classes to load the specific YOLO model."""
        raise NotImplementedError("This method should be overridden in derived classes")

    def debug_plot(self, image):
        if not DEBUG_PLOT:
            return

        plt.figure(figsize=(10, 10))
        plt.imshow(image[..., ::-1])
        plt.axis('off')
        plt.title(self.type)
        plt.show()

    def predict_regions(self, path) -> List[Dict]:
        """Predict regions in the image using the YOLO model.
        Args:
            path (str): Path to the file with media
        """
        raise NotImplementedError("This method should be overridden in derived classes")

    def __str__(self):
        """Return a string with full representation of the control tag."""
        return (
            f"{self.type} from_name={self.from_name}, "
            f"label_map={self.label_map}, score_threshold={self.score_threshold}"
        )

    class Config:
        arbitrary_types_allowed = True
