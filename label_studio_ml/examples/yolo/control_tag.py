from pydantic import BaseModel
from typing import List, Optional, Dict

from label_studio_sdk.label_interface.control_tags import ControlTag
from ultralytics import YOLO


class ControlTag(BaseModel):
    """
    Represents a control tag in Label Studio, which is associated with a specific type of data
    and is used to generate predictions using a model like YOLO.

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

    type: str
    control: ControlTag
    from_name: str
    to_name: str
    value: str
    model: YOLO
    score_threshold: float = 0.5
    label_map: Optional[Dict[str, str]] = {}

    def __init__(self, **data):
        super().__init__(**data)

    def __str__(self):
        """Return a string with full representation of the control tag.
        """
        return (
            f"{self.type} from_name={self.from_name}, "
            f"label_map={self.label_map}, score_threshold={self.score_threshold}"
        )



    class Config:
        arbitrary_types_allowed = True
