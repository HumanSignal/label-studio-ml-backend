import os
import logging

from pydantic import BaseModel
from typing import Optional, List, Dict, ClassVar
# --- MODIFIED IMPORT ---
# from ultralytics import YOLO
from doclayout_yolo import YOLOv10 as DocLayoutModel # Renamed for clarity
# --- END MODIFICATION ---

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
# --- ADDED ---
# Default model name, can be overridden by environment variable or label config attribute
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "doclayout_yolo_docstructbench_imgsz1024.pt")
# Default image size, can be overridden by label config attribute
DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", 1024))
# --- END ADDITION ---

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
    and is used to generate predictions using a DocLayout-YOLO model.

    Attributes:
        type (str): Type of the control, e.g., RectangleLabels, Choices, etc.
        control (ControlTag): The actual control element from the Label Studio configuration.
        from_name (str): The name of the control tag, used to link the control to the data.
        to_name (str): The name of the data field that this control is associated with.
        value (str): The value name from the object that this control operates on, e.g., an image or text field.
        # --- MODIFIED TYPE HINT ---
        model (DocLayoutModel): The model instance (DocLayout-YOLO) used to generate predictions for this control.
        # --- END MODIFICATION ---
        model_path (str): Path/name of the DocLayout-YOLO model file relative to MODEL_ROOT.
        model_score_threshold (float): Threshold for prediction scores; predictions below this value will be ignored.
        # --- ADDED ---
        model_imgsz (int): Image size used for prediction.
        # --- END ADDITION ---
        label_map (Optional[Dict[str, str]]): A mapping of model labels to Label Studio labels.
    """

    type: ClassVar[str]
    control: ControlTag
    from_name: str
    to_name: str
    value: str
    # --- MODIFIED TYPE HINT ---
    model: DocLayoutModel
    # --- END MODIFICATION ---
    model_path: str # Changed from ClassVar to instance variable
    model_score_threshold: float = 0.5
    # --- ADDED ---
    model_imgsz: int = 1024
    # --- END ADDITION ---
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
                "score_threshold" # not recommended option, use `model_score_threshold`
            )
            or MODEL_SCORE_THRESHOLD
        )
        # --- ADDED ---
        # read imgsz attribute from control tag
        model_imgsz = int(
            control.attr.get("model_imgsz")
            or control.attr.get("imgsz") # alternative name
            or DEFAULT_IMGSZ
        )
        # --- END ADDITION ---

        # read `model_path` attribute from the control tag or use default
        model_name = (
            ALLOW_CUSTOM_MODEL_PATH and control.attr.get("model_path")
        ) or DEFAULT_MODEL_NAME # Use default name if not specified

        # --- MODIFIED MODEL LOADING ---
        model_full_path = os.path.join(MODEL_ROOT, model_name)
        if not os.path.exists(model_full_path):
             # Try loading from Hugging Face Hub if path doesn't exist locally
             logger.warning(f"Local model '{model_full_path}' not found.")
             logger.info(f"Attempting to load '{model_name}' from Hugging Face Hub...")
             # We assume model_name is a valid HF repo ID or contains one like 'user/repo'
             # If the model needs specific args for from_pretrained, they'd need to be handled
             try:
                 # Use the class method from DocLayout-YOLO README if available
                 if hasattr(DocLayoutModel, 'from_pretrained'):
                      model = DocLayoutModel.from_pretrained(model_name)
                      # Cache the loaded model using its identifier
                      _model_cache[model_name] = model
                      logger.info(f"Successfully loaded '{model_name}' from Hugging Face Hub.")
                 else:
                      # Fallback or specific logic if from_pretrained isn't standard
                      raise NotImplementedError(f"Cannot load '{model_name}' automatically. Please download it to the '{MODEL_ROOT}' directory.")
             except Exception as e:
                 logger.error(f"Failed to load model '{model_name}' from Hugging Face Hub: {e}")
                 raise FileNotFoundError(
                    f"Model file '{model_name}' not found locally in '{MODEL_ROOT}' "
                    f"and could not be loaded from Hugging Face Hub. "
                    f"Please ensure the model exists or `model_path` is set correctly."
                 ) from e
        else:
            # Load from local cache/file
            model = cls.get_cached_model(model_full_path) # Use full path for caching local files

        # Check if model loading was successful
        if model is None:
             raise ValueError(f"Failed to load model '{model_name}'")

        # --- Get model class names (assuming compatible API) ---
        # IMPORTANT: Verify that DocLayoutModel instance has a `names` attribute like ultralytics.YOLO
        if not hasattr(model, 'names') or not isinstance(model.names, dict):
             logger.warning(f"Model '{model_name}' does not have a standard `.names` attribute (dictionary of class IDs to names). Label mapping might be incorrect.")
             # Fallback: Try to get names from the label config itself if model doesn't provide them?
             # Or raise an error? For now, let's proceed but label map will likely be empty/wrong.
             model_names = []
        else:
             model_names = list(model.names.values()) # class names from the model
        # --- END MODIFIED MODEL LOADING & CLASS NAMES ---

        # from_name for label mapping can be differed from control.name (e.g. VideoRectangle)
        label_map_from_name = cls.get_from_name_for_label_map(
            mlbackend.label_interface, from_name
        )
        label_map = mlbackend.build_label_map(label_map_from_name, model_names)

        # --- MODIFIED RETURN VALUE ---
        return cls(
            control=control,
            from_name=from_name,
            to_name=to_name,
            value=value,
            model=model,
            model_path=model_name, # Store the potentially relative path/name used
            model_score_threshold=model_score_threshold,
            model_imgsz=model_imgsz, # Pass imgsz
            label_map=label_map,
            label_studio_ml_backend=mlbackend,
            project_id=mlbackend.project_id,
        )
        # --- END MODIFICATION ---

    @classmethod
    def load_yolo_model(cls, full_path) -> DocLayoutModel: # Changed param name
        """Load DocLayout-YOLO model from the file."""
        logger.info(f"Loading DocLayout-YOLO model: {full_path}")
        # --- MODIFIED MODEL INSTANTIATION ---
        try:
            model = DocLayoutModel(full_path)
            # IMPORTANT: Verify .names attribute existence and type
            if not hasattr(model, 'names') or not isinstance(model.names, dict):
                 logger.warning(f"Loaded model from '{full_path}' lacks a standard `.names` dictionary.")
            else:
                 logger.info(f"Model {full_path} names:\n{model.names}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {full_path}: {e}")
            raise
        # --- END MODIFICATION ---

    @classmethod
    def get_cached_model(cls, path: str) -> DocLayoutModel: # Path is now full path for local
        """Gets a model from cache or loads it."""
        if path not in _model_cache:
            _model_cache[path] = cls.load_yolo_model(path)
        return _model_cache[path]

    def debug_plot(self, image_or_result): # Argument might be result object now
        if not DEBUG_PLOT:
            return

        try:
            import matplotlib.pyplot as plt
            # --- MODIFIED ---
            # The yolov8 example plotted the *result* object. Check if DocLayout-YOLO results have plot()
            if hasattr(image_or_result, 'plot'):
                 # Assuming plot returns an image array (like numpy)
                 plot_img = image_or_result.plot(pil=True) # Use PIL=True based on README example
                 # Convert PIL to numpy array for matplotlib if needed
                 import numpy as np
                 plot_img_np = np.array(plot_img)
                 plt.imshow(plot_img_np) # Display the annotated image
                 plt.axis("off")
                 plt.title(f"{self.type} - {self.model_path}")
                 plt.show()
            # Fallback if no plot method or plotting raw image
            # elif isinstance(image_or_result, np.ndarray):
            #      plt.imshow(image_or_result[..., ::-1]) # Assuming BGR->RGB if needed
            #      plt.axis("off")
            #      plt.title(self.type)
            #      plt.show()
            else:
                 logger.warning("Debug plot failed: Result object has no standard plot() method.")
            # --- END MODIFIED ---
        except ImportError:
            logger.warning("matplotlib not installed, skipping debug plot.")
        except Exception as e:
            logger.error(f"Error during debug plot: {e}")


    def predict_regions(self, path) -> List[Dict]:
        """Predict regions in the image using the DocLayout-YOLO model.
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
        try:
            path = get_local_path(task_path, task_id=task.get("id"))
            logger.debug(f"Resolved path: {task_path} => {path}")
            if not os.path.exists(path):
                 logger.error(f"Resolved path does not exist: {path}")
                 raise FileNotFoundError(f"Resolved path does not exist: {path}")
        except Exception as e:
            logger.error(f"Error resolving path {task_path}: {e}")
            raise ValueError(f"Cannot access file at {task_path}. Check ML backend logs for details.") from e

        return path

    def __str__(self):
        """Return a string with full representation of the control tag."""
        return (
            f"{self.type} from_name={self.from_name}, "
            f"model={self.model_path}, imgsz={self.model_imgsz}, threshold={self.model_score_threshold}, "
            f"label_map={self.label_map}"
        )

    class Config:
        arbitrary_types_allowed = True
        protected_namespaces = ("__.*__", "_.*")  # Excludes 'model_'