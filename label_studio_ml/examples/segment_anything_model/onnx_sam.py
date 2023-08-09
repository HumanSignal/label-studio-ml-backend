import numpy as np
import cv2
import os
import string
import random
import torch
import onnxruntime
import logging

from typing import List, Dict, Optional
from segment_anything import sam_model_registry, SamPredictor
from label_studio_ml.model import LabelStudioMLBase
from label_studio_converter import brush
from label_studio_ml.utils import get_image_local_path, InMemoryLRUDictCache

VITH_CHECKPOINT = os.environ.get("VITH_CHECKPOINT", "sam_vit_h_4b8939.pth")
ONNX_CHECKPOINT = os.environ.get("ONNX_CHECKPOINT", "sam_onnx_quantized_example.onnx")

logger = logging.getLogger(__name__)


def load_my_model():
    """
    Loads the Segment Anything model on initializing Label studio, so if you call it outside MyModel it doesn't load every time you try to make a prediction
    Returns the predictor object. For more, look at Facebook's SAM docs
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"  # if you're not using CUDA, use "cpu" instead .... good luck not burning your computer lol
    logger.debug(f"Using device {device}")
    sam = sam_model_registry["vit_h"](
        VITH_CHECKPOINT)  # Note: YOU MUST HAVE THE MODEL SAVED IN THE SAME DIRECTORY AS YOUR BACKEND
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor


PREDICTOR = load_my_model()

# empty mask and indicator for no mask
onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

ORT = onnxruntime.InferenceSession(ONNX_CHECKPOINT)

IN_MEM_CACHE = InMemoryLRUDictCache(int(os.getenv("IMAGE_CACHE_SIZE", 100)))


class SamModel(LabelStudioMLBase):

    def get_img_size(self, context):
        height = context['result'][0]['original_height']
        width = context['result'][0]['original_width']
        logger.debug(f'height is {height} and width is {width}')
        return height, width

    def store_embeddings_in_cache(self, predictor, img_path):
        # Get image and embeddings
        logger.debug(f"Storing embeddings for {img_path} in `IN_MEM_CACHE`")
        image_path = get_image_local_path(img_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        payload = {
            'image': image,
            'image_embedding': image_embedding
        }
        IN_MEM_CACHE.put(img_path, payload)
        logger.debug(f'Finished storing embeddings for {img_path} in `IN_MEM_CACHE`: '
                     f'embedding shape {image_embedding.shape}')
        return payload

    def get_image_embeddings(self, predictor, img_path):
        payload = IN_MEM_CACHE.get(img_path)
        if payload is None:
            payload = self.store_embeddings_in_cache(predictor, img_path)
        else:
            logger.debug(f"Using embeddings for {img_path} from `IN_MEM_CACHE`")
        return payload

    def _predict_mask(self, img_path, onnx_coord, onnx_label):

        payload = self.get_image_embeddings(PREDICTOR, img_path)

        # loading the image you are annotating
        image = payload['image']
        image_embedding = payload['image_embedding']
        onnx_coord = PREDICTOR.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

        # Package to run in onnx
        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
        }
        logger.debug(
            f"ORT inputs: "
            f"point_coords={ort_inputs['point_coords']},"
            f"point_labels={ort_inputs['point_labels']}"
        )
        # Predict and threshold mask
        masks, _, low_res_logits = ORT.run(None, ort_inputs)
        masks = masks > PREDICTOR.model.mask_threshold

        mask = masks[0, 0, :, :].astype(np.uint8)  # each mask has shape [H, W]
        return mask

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs):
        """ Returns the predicted mask for a smart keypoint that has been placed."""

        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')
        img_path = tasks[0]['data'][value]

        if not context:
            # if there is no context, no interaction has happened yet
            # however, if it has just opened the task, we can warm the cache with the image
            # for the efficiency for the consecutive calls
            if img_path not in IN_MEM_CACHE:
                print(f'Create cache for {img_path}')
                self.store_embeddings_in_cache(PREDICTOR, img_path)
            return []

        logger.debug(f"the context is {context}")
        logger.debug(f"the tasks are {tasks}")
        logger.debug(f"the kwargs are {kwargs}")

        # smart annotation
        smart_annotation_type = context['result'][0]['type']
        if smart_annotation_type not in ['rectanglelabels', 'keypointlabels']:
            # only rectangle and keypoint labels are supported
            return []

        # get the smart annotation label
        label = context['result'][0]['value'][smart_annotation_type][0]

        # getting the height and width of the image that you are annotating real-time
        height, width = self.get_img_size(context)

        if smart_annotation_type == 'rectanglelabels':
            # getting coordinates of the box
            x = context['result'][0]['value']['x'] * width / 100
            y = context['result'][0]['value']['y'] * height / 100
            box_width = context['result'][0]['value']['width'] * width / 100
            box_height = context['result'][0]['value']['height'] * height / 100

            input_box = np.array([int(x), int(y), int(box_width + x), int(box_height + y)])
            onnx_box_coords = input_box.reshape(2, 2)
            onnx_box_labels = np.array([2, 3])
            # onnx_coord = np.concatenate([None, onnx_box_coords], axis=0)[None, :, :]
            onnx_coord = np.concatenate([onnx_box_coords, np.array([[0, 0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([onnx_box_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)

        else:
            # getting coordinates of the keypoint
            x = context['result'][0]['value']['x'] * width / 100
            y = context['result'][0]['value']['y'] * height / 100

            input_point = np.array([[int(x), int(y)]])
            input_label = np.array([1])

            onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        # Run inference using the onnx model
        predicted_mask = self._predict_mask(img_path, onnx_coord, onnx_label)

        results = []
        label_id = ''.join(random.SystemRandom().choice(
            string.ascii_uppercase + string.ascii_lowercase + string.digits)),  # creates a random ID for your label everytime so no chance for errors

        # converting the mask from the model to RLE format which is usable in Label Studio
        mask = predicted_mask * 255
        rle = brush.mask2rle(mask)
        results.append({
            "from_name": from_name,
            "to_name": to_name,
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "value": {
                "format": "rle",
                "rle": rle,
                "brushlabels": [label],
            },
            "type": "brushlabels",
            "id": label_id,
            "readonly": False,
        })

        # returning the result from the prediction and passing it to show on the front-end
        return [{'result': results, 'model_version': 'vit_h'}]


if __name__ == '__main__':
    # test the model
    model = SamModel()
    model.use_label_config('''
    <View>
        <Image name="image" value="$image" zoom="true"/>
        <BrushLabels name="tag" toName="image">
            <Label value="Banana" background="#FF0000"/>
            <Label value="Orange" background="#0d14d3"/>
        </BrushLabels>
        <KeyPointLabels name="tag2" toName="image" smart="true" >
            <Label value="Banana" background="#000000" showInline="true"/>
            <Label value="Orange" background="#000000" showInline="true"/>
        </KeyPointLabels>
        <RectangleLabels name="tag3" toName="image"  >
            <Label value="Banana" background="#000000" showInline="true"/>
            <Label value="Orange" background="#000000" showInline="true"/>
        </RectangleLabels>
    </View>
    ''')
    results = model.predict(
        tasks=[{
            'data': {
                'image': 'https://s3.amazonaws.com/htx-pub/datasets/images/125245483_152578129892066_7843809718842085333_n.jpg'
            }}],
        context={
            'result': [{
                'original_width': 1080,
                'original_height': 1080,
                'image_rotation': 0,
                'value': {
                    'x': 49.441786283891545,
                    'y': 59.96810207336522,
                    'width': 0.3189792663476874,
                    'labels': ['Banana'],
                    'keypointlabels': ['Banana']
                },
                'is_positive': True,
                'id': 'fBWv1t0S2L',
                'from_name': 'tag2',
                'to_name': 'image',
                'type': 'keypointlabels',
                'origin': 'manual'
            }]}
    )
    import json
    results[0]['result'][0]['value']['rle'] = f'...{len(results[0]["result"][0]["value"]["rle"])} integers...'
    print(json.dumps(results, indent=2))
