import os
import logging
from typing import List, Dict, Optional

import torch
import torchvision.transforms as T
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import get_local_path, DATA_UNDEFINED_NAME

# --- D-FINE specific imports ---
DFINE_CODE_DIR = os.getenv('DFINE_CODE_DIR', '/app/d-fine-code')
if DFINE_CODE_DIR not in os.sys.path:
    os.sys.path.insert(0, DFINE_CODE_DIR)
try:
    from src.core import YAMLConfig
except ImportError as e:
    raise ImportError(
        f"Failed to import D-FINE components. "
        f"Ensure DFINE_CODE_DIR ('{DFINE_CODE_DIR}') is set correctly and contains D-FINE's 'src' directory, "
        f"and that PYTHONPATH is configured if running locally. Original error: {e}"
    )
# --- End D-FINE specific imports ---

logger = logging.getLogger(__name__)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class DFINEModel(LabelStudioMLBase):
    DEFAULT_MODEL_CONFIG_FILE = 'dfine_hgnetv2_l_coco.yml'
    DEFAULT_MODEL_WEIGHTS_FILE = 'dfine_l_coco.pth'

    def __init__(self, **kwargs):
        # The project_id and label_config are passed by LSML and handled by super().__init__
        # self.setup() will be called by super().__init__()
        super(DFINEModel, self).__init__(**kwargs)

    def setup(self):
        """
        Configure any parameters of your model here.
        This is called only once by LabelStudioMLBase.__init__(),
        so all D-FINE specific initializations should go here.
        """
        self.device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # MODEL_DIR is usually set by the _wsgi.py or docker-compose.yml
        self.model_dir = os.getenv('MODEL_DIR', '/data/models')
        self.dfine_config_root_dir = os.path.join(DFINE_CODE_DIR, 'configs')

        model_config_filename = os.getenv('DFINE_CONFIG_FILE', self.DEFAULT_MODEL_CONFIG_FILE)
        model_weights_filename = os.getenv('DFINE_MODEL_WEIGHTS', self.DEFAULT_MODEL_WEIGHTS_FILE)

        # Define these attributes on self *before* using them in self.set("model_version", ...)
        self.model_config_path = os.path.join(self.dfine_config_root_dir, 'dfine', model_config_filename)
        self.model_weights_path = os.path.join(self.model_dir, model_weights_filename)
        
        logger.info(f"D-FINE Code Directory: {DFINE_CODE_DIR}")
        logger.info(f"Model Weights Directory (for .pth): {self.model_dir}")
        logger.info(f"D-FINE Config Root Directory (for .yml): {self.dfine_config_root_dir}")
        logger.info(f"Resolved D-FINE Config Path: {self.model_config_path}")
        logger.info(f"Resolved Model Weights Path: {self.model_weights_path}")

        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(f"D-FINE config file not found: {self.model_config_path}")
        if not os.path.exists(self.model_weights_path):
            raise FileNotFoundError(f"D-FINE model weights not found: {self.model_weights_path}")

        # Set model version using the cache's set method
        model_version_name = f"DFINE-{os.path.basename(self.model_config_path)}-{os.path.basename(self.model_weights_path)}"
        self.set("model_version", model_version_name) # `set` is inherited from LabelStudioMLBase

        # Load D-FINE configuration and model
        self.d_fine_cfg = YAMLConfig(self.model_config_path, resume=self.model_weights_path)
        if "HGNetv2" in self.d_fine_cfg.yaml_cfg and self.model_weights_path:
            self.d_fine_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        
        logger.info(f"Loading checkpoint from {self.model_weights_path}")
        checkpoint = torch.load(self.model_weights_path, map_location="cpu")
        state = checkpoint.get("ema", {}).get("module", checkpoint.get("model"))
        if state is None:
             raise ValueError(f"Could not find model state in checkpoint: {self.model_weights_path}. Checked 'ema.module' and 'model'.")
        
        self.d_fine_cfg.model.load_state_dict(state)
        logger.info("Model state loaded successfully.")

        self.model = self.d_fine_cfg.model.deploy().to(self.device).eval()
        self.postprocessor = self.d_fine_cfg.postprocessor.deploy().to(self.device).eval()
        logger.info("D-FINE model and postprocessor loaded and set to eval mode.")

        eval_size = self.d_fine_cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
        self.image_height, self.image_width = eval_size[0], eval_size[1]
        self.transform = T.Compose([
            T.Resize((self.image_height, self.image_width)),
            T.ToTensor(),
        ])

        self.model_class_names = COCO_CLASSES
        
        # Initialize score_thresh_from_config and label_map
        # self.label_interface is available here because LabelStudioMLBase.__init__ calls use_label_config
        # *before* calling self.setup(), provided label_config was passed to __init__.
        # (label_config is passed during /setup API call)
        self.score_thresh_from_config = float(os.getenv('MODEL_SCORE_THRESHOLD', 0.5))
        self.label_map = {} # Will be populated based on the specific control tag in predict or if label_interface is ready

        if self.label_interface: # Check if label_interface is initialized (it should be if label_config was passed)
            try:
                # self.from_name, self.to_name, self.value are set by get_first_tag_occurence
                from_name_for_labels, _, _ = self.get_first_tag_occurence('RectangleLabels', 'Image')
                # self.build_label_map uses self.from_name internally
                self.label_map = self.build_label_map(from_name_for_labels, self.model_class_names)
                logger.info(f"Label map for '{from_name_for_labels}' loaded in setup: {self.label_map}")
                if not self.label_map:
                     logger.warning(f"No label map built for control tag '{from_name_for_labels}' during setup. "
                                   f"Ensure 'predicted_values' in your Label Studio config match model labels: {self.model_class_names}")
                
                control_tag_attrs = self.label_interface.get_control(from_name_for_labels).attr
                self.score_thresh_from_config = float(control_tag_attrs.get('model_score_threshold', self.score_thresh_from_config))
            except Exception as e:
                logger.warning(f"Could not find 'RectangleLabels' or parse score_threshold from label_config during setup: {e}. "
                               "Will attempt in predict() or use default/env_var for score_threshold.")
        else:
            logger.warning("self.label_interface not available during setup. Score_threshold will use default/env_var and label_map will be empty. This might happen if model is initialized without a label_config initially.")
        
        logger.info(f"Using score threshold (from setup): {self.score_thresh_from_config}")


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        predictions = []
        
        from_name, to_name, value_key = self.get_first_tag_occurence('RectangleLabels', 'Image')
        
        # Ensure label_map is built using the current from_name from the labeling config
        # self.from_name is an attribute of LabelStudioMLBase, set by get_first_tag_occurence
        if not self.label_map or self.from_name != from_name:
            current_label_map = self.build_label_map(from_name, self.model_class_names)
            if not current_label_map:
                logger.warning(f"No label map built for control tag '{from_name}' in predict. "
                               f"Ensure 'predicted_values' in your Label Studio config match model labels: {self.model_class_names}")
            self.label_map = current_label_map
        
        # Get score threshold from current control tag attributes
        control_tag_attrs = self.label_interface.get_control(from_name).attr
        current_score_threshold = float(control_tag_attrs.get('model_score_threshold', self.score_thresh_from_config))


        for task in tasks:
            image_url = task['data'].get(value_key) or task['data'].get(DATA_UNDEFINED_NAME)
            if not image_url:
                logger.warning(f"Image URL not found in task: {task.get('id', 'N/A')}. Skipping.")
                predictions.append({"result": [], "score": 0, "model_version": self.get("model_version")})
                continue

            try:
                image_path = get_local_path(image_url, task_id=task.get('id'))
                image_pil = Image.open(image_path).convert("RGB")
                original_width, original_height = image_pil.size

                image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
                original_size_tensor = torch.tensor([[original_width, original_height]], dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    outputs_raw = self.model(image_tensor)
                    processed_predictions = self.postprocessor(outputs_raw, original_size_tensor)
                
                results_for_task = []
                avg_score = 0
                num_detections = 0

                if processed_predictions and isinstance(processed_predictions, list) and len(processed_predictions) > 0:
                    pred_item = processed_predictions[0] 
                    
                    labels_raw = pred_item.get('labels')
                    boxes_raw = pred_item.get('boxes') 
                    scores_raw = pred_item.get('scores')

                    if labels_raw is not None and boxes_raw is not None and scores_raw is not None:
                        for label_idx, box_coords, score in zip(labels_raw, boxes_raw, scores_raw):
                            score_val = score.item()
                            if score_val < current_score_threshold:
                                continue
                            
                            model_label_name_idx = label_idx.item()
                            if not (0 <= model_label_name_idx < len(self.model_class_names)):
                                logger.warning(f"Predicted label index {model_label_name_idx} is out of bounds for COCO_CLASSES (len {len(self.model_class_names)}). Skipping.")
                                continue
                            model_label_name = self.model_class_names[model_label_name_idx]
                            
                            ls_label = self.label_map.get(model_label_name)

                            if not ls_label:
                                logger.debug(f"Label '{model_label_name}' (idx {label_idx.item()}) not found in LS label map {self.label_map}. Skipping.")
                                continue

                            x1, y1, x2, y2 = box_coords.tolist()

                            results_for_task.append({
                                "from_name": from_name,
                                "to_name": to_name,
                                "type": "rectanglelabels",
                                "value": {
                                    "rectanglelabels": [ls_label],
                                    "x": (x1 / original_width) * 100,
                                    "y": (y1 / original_height) * 100,
                                    "width": ((x2 - x1) / original_width) * 100,
                                    "height": ((y2 - y1) / original_height) * 100,
                                },
                                "score": score_val,
                            })
                            avg_score += score_val
                            num_detections += 1
                
                final_avg_score = avg_score / num_detections if num_detections > 0 else 0
                predictions.append({
                    "result": results_for_task,
                    "score": final_avg_score,
                    "model_version": self.get("model_version")
                })

            except Exception as e:
                logger.error(f"Error processing task {task.get('id', 'N/A')}: {e}", exc_info=True)
                predictions.append({"result": [], "score": 0, "model_version": self.get("model_version")})
        
        return ModelResponse(predictions=predictions) # Removed model_version from ModelResponse init, it will use self.get("model_version") internally

    def fit(self, event, data, **kwargs):
        logger.info("fit() method called, but D-FINE training is not implemented in this ML backend.")
        pass