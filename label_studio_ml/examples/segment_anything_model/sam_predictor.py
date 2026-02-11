import os
import logging
import torch
import cv2
import pathlib
import numpy as np

from typing import List, Dict, Optional
from label_studio_ml.utils import InMemoryLRUDictCache

# Monkey-patch torch.as_tensor to handle numpy 2.x compatibility
_original_as_tensor = torch.as_tensor
def _patched_as_tensor(data, dtype=None, device=None):
    """Patched version of torch.as_tensor that handles numpy 2.x compatibility"""
    if isinstance(data, np.ndarray):
        # For numpy 2.x compatibility, ensure arrays are properly converted
        if dtype is None and data.dtype == np.uint8:
            # Explicitly convert uint8 arrays
            return _original_as_tensor(data.copy(), dtype=torch.uint8, device=device)
        elif dtype is not None:
            # If dtype is specified, ensure the array is compatible
            if data.dtype == np.float32 and dtype == torch.int:
                # Convert float32 to int properly
                return _original_as_tensor(data.astype(np.int32), dtype=dtype, device=device)
    return _original_as_tensor(data, dtype=dtype, device=device)
torch.as_tensor = _patched_as_tensor

# Also patch tensor.numpy() to handle numpy 2.x compatibility
_original_tensor_numpy = torch.Tensor.numpy
def _patched_tensor_numpy(self, *args, **kwargs):
    """Patched version of tensor.numpy() that handles numpy 2.x compatibility"""
    try:
        return _original_tensor_numpy(self, *args, **kwargs)
    except RuntimeError as e:
        if "Numpy is not available" in str(e):
            # Fallback: manually convert tensor to numpy array
            # This is a workaround for numpy 2.x compatibility issues
            arr = self.detach().cpu().contiguous()
            # Convert to list first, then to numpy array
            if arr.dim() == 0:
                return np.array(arr.item())
            else:
                # Map torch dtypes to numpy dtypes
                dtype_map = {
                    torch.float32: np.float32,
                    torch.float64: np.float64,
                    torch.int32: np.int32,
                    torch.int64: np.int64,
                    torch.uint8: np.uint8,
                    torch.bool: np.bool_,
                }
                np_dtype = dtype_map.get(arr.dtype, None)
                return np.array(arr.tolist(), dtype=np_dtype)
        raise
torch.Tensor.numpy = _patched_tensor_numpy

logger = logging.getLogger(__name__)
_MODELS_DIR = pathlib.Path(__file__).parent / "models"

VITH_CHECKPOINT = os.environ.get("VITH_CHECKPOINT", _MODELS_DIR / "sam_vit_h_4b8939.pth")
ONNX_CHECKPOINT = os.environ.get("ONNX_CHECKPOINT", _MODELS_DIR / "sam_onnx_quantized_example.onnx")
MOBILESAM_CHECKPOINT = os.environ.get("MOBILESAM_CHECKPOINT", _MODELS_DIR / "mobile_sam.pt")


class SAMPredictor(object):

    def __init__(self, model_choice):
        self.model_choice = model_choice

        # cache for embeddings
        # TODO: currently it supports only one image in cache,
        #   since predictor.set_image() should be called each time the new image comes
        #   before making predictions
        #   to extend it to >1 image, we need to store the "active image" state in the cache
        self.cache = InMemoryLRUDictCache(1)

        # if you're not using CUDA, use "cpu" instead .... good luck not burning your computer lol
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device {self.device}")

        if model_choice == 'ONNX':
            import onnxruntime
            from segment_anything import sam_model_registry, SamPredictor

            self.model_checkpoint = VITH_CHECKPOINT
            if self.model_checkpoint is None:
                raise FileNotFoundError("VITH_CHECKPOINT is not set: please set it to the path to the SAM checkpoint")
            if ONNX_CHECKPOINT is None:
                raise FileNotFoundError("ONNX_CHECKPOINT is not set: please set it to the path to the ONNX checkpoint")
            logger.info(f"Using ONNX checkpoint {ONNX_CHECKPOINT} and SAM checkpoint {self.model_checkpoint}")

            self.ort = onnxruntime.InferenceSession(ONNX_CHECKPOINT)
            reg_key = "vit_h"

        elif model_choice == 'SAM':
            from segment_anything import SamPredictor, sam_model_registry

            self.model_checkpoint = VITH_CHECKPOINT
            if self.model_checkpoint is None:
                raise FileNotFoundError("VITH_CHECKPOINT is not set: please set it to the path to the SAM checkpoint")

            logger.info(f"Using SAM checkpoint {self.model_checkpoint}")
            reg_key = "vit_h"

        elif model_choice == 'MobileSAM':
            from mobile_sam import SamPredictor, sam_model_registry

            self.model_checkpoint = MOBILESAM_CHECKPOINT
            if not self.model_checkpoint:
                raise FileNotFoundError("MOBILE_CHECKPOINT is not set: please set it to the path to the MobileSAM checkpoint")
            logger.info(f"Using MobileSAM checkpoint {self.model_checkpoint}")
            reg_key = 'vit_t'
        else:
            raise ValueError(f"Invalid model choice {model_choice}")

        sam = sam_model_registry[reg_key](checkpoint=self.model_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    @property
    def model_name(self):
        return f'{self.model_choice}:{self.model_checkpoint}:{self.device}'

    def set_image(self, img_path, calculate_embeddings=True, task=None, path_resolver=None):
        payload = self.cache.get(img_path)
        if payload is None:
            # Get image and embeddings
            logger.debug(f'Payload not found for {img_path} in `IN_MEM_CACHE`: calculating from scratch')
            if path_resolver is None:
                raise ValueError("path_resolver is required for resolving local paths")
            image_path = path_resolver(img_path, task_id=task.get('id'))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Ensure image is contiguous and properly typed for numpy 2.x compatibility
            image = np.ascontiguousarray(image, dtype=np.uint8)
            self.predictor.set_image(image)
            payload = {'image_shape': image.shape[:2]}
            logger.debug(f'Finished set_image({img_path}) in `IN_MEM_CACHE`: image shape {image.shape[:2]}')
            if calculate_embeddings:
                image_embedding = self.predictor.get_image_embedding().cpu().numpy()
                payload['image_embedding'] = image_embedding
                logger.debug(f'Finished storing embeddings for {img_path} in `IN_MEM_CACHE`: '
                             f'embedding shape {image_embedding.shape}')
            self.cache.put(img_path, payload)
        else:
            logger.debug(f"Using embeddings for {img_path} from `IN_MEM_CACHE`")
        return payload

    def predict_onnx(
        self,
        img_path,
        point_coords: Optional[List[List]] = None,
        point_labels: Optional[List] = None,
        input_box: Optional[List] = None,
        task: Optional[Dict] = None,
        path_resolver=None
    ):
        # calculate embeddings
        payload = self.set_image(
            img_path,
            calculate_embeddings=True,
            task=task,
            path_resolver=path_resolver
        )
        image_shape = payload['image_shape']
        image_embedding = payload['image_embedding']

        onnx_point_coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        onnx_point_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        onnx_box_coords = np.array(input_box, dtype=np.float32).reshape(2, 2) if input_box else None

        onnx_coords, onnx_labels = None, None
        if onnx_point_coords is not None and onnx_box_coords is not None:
            # both keypoints and boxes are present
            onnx_coords = np.concatenate([onnx_point_coords, onnx_box_coords], axis=0)[None, :, :]
            onnx_labels = np.concatenate([onnx_point_labels, np.array([2, 3])], axis=0)[None, :].astype(np.float32)

        elif onnx_point_coords is not None:
            # only keypoints are present
            onnx_coords = np.concatenate([onnx_point_coords, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_labels = np.concatenate([onnx_point_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)

        elif onnx_box_coords is not None:
            # only boxes are present
            raise NotImplementedError("Boxes without keypoints are not supported yet")

        onnx_coords = self.predictor.transform.apply_coords(onnx_coords, image_shape).astype(np.float32)

        # TODO: support mask inputs
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)

        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coords,
            "point_labels": onnx_labels,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image_shape, dtype=np.float32)
        }

        masks, prob, low_res_logits = self.ort.run(None, ort_inputs)
        masks = masks > self.predictor.model.mask_threshold
        mask = masks[0, 0, :, :].astype(np.uint8)  # each mask has shape [H, W]
        prob = float(prob[0][0])
        # TODO: support the real multimask output as in https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
        return {
            'masks': [mask],
            'probs': [prob]
        }

    def predict_sam(
        self,
        img_path,
        point_coords: Optional[List[List]] = None,
        point_labels: Optional[List] = None,
        input_box: Optional[List] = None,
        task: Optional[Dict] = None,
        path_resolver=None
    ):
        self.set_image(
            img_path,
            calculate_embeddings=False,
            task=task,
            path_resolver=path_resolver
        )
        point_coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        point_labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        input_box = np.array(input_box, dtype=np.float32) if input_box else None

        masks, probs, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box,
            # TODO: support multimask output
            multimask_output=False
        )
        mask = masks[0, :, :].astype(np.uint8)  # each mask has shape [H, W]
        prob = float(probs[0])
        return {
            'masks': [mask],
            'probs': [prob]
        }

    def predict(
        self, img_path: str,
        point_coords: Optional[List[List]] = None,
        point_labels: Optional[List] = None,
        input_box: Optional[List] = None,
        task: Optional[Dict] = None,
        path_resolver=None
    ):
        if self.model_choice == 'ONNX':
            return self.predict_onnx(
                img_path, point_coords, point_labels, input_box, task, path_resolver
            )
        elif self.model_choice in ('SAM', 'MobileSAM'):
            return self.predict_sam(
                img_path, point_coords, point_labels, input_box, task, path_resolver
            )
        else:
            raise NotImplementedError(f"Model choice {self.model_choice} is not supported yet")

