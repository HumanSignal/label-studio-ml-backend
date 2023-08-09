from label_studio_converter import brush
from label_studio_ml.utils import InMemoryLRUDictCache
import numpy as np
import os
import string
import random
import torch
from typing import List, Dict, Optional
from onnx_sam import SamModel

SAM_CHOICE = os.environ.get("SAM_CHOICE", "MobileSAM")  # other option is just SAM

if SAM_CHOICE == "MobileSAM":
    from mobile_sam import SamPredictor, sam_model_registry

    CHECKPOINT = os.environ.get("MOBILESAM_CHECKPOINT", "mobile_sam.pt")
    model_type = "vit_t"
elif SAM_CHOICE == "SAM":
    from segment_anything import sam_model_registry, SamPredictor

    CHECKPOINT = os.environ.get("VITH_CHECKPOINT", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"


def load_my_model():
    """
    Loads the Segment Anything model on initializing Label studio, so if you call it outside AdvancedSAM it doesn't load every time you try to make a prediction
    Returns the predictor object. For more, look at Facebook's SAM docs
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](
        checkpoint=CHECKPOINT)  # Note: YOU MUST HAVE THE MODEL SAVED IN THE SAME DIRECTORY AS YOUR BACKEND
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor


PREDICTOR = load_my_model()
PREV_IMG_PATH = ""
PREV_IMG = 0
IMAGE_EMBEDDING = 0
previous_id = 0

IN_MEM_CACHE = InMemoryLRUDictCache(int(os.getenv("IMAGE_CACHE_SIZE", 100)))


class AdvancedSamModel(SamModel):

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
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

            # getting the height and width of the image that you are annotating
        height, width = self.get_img_size(context)

        points, labels, input_box, alias, box_width, box_height = self.get_tasks(tasks, height, width, **kwargs)

        x, y, box_width, box_height, smart_annotation = self.get_smart_position(width=width, height=height,
                                                                                box_width=box_width, box_height=box_height, **kwargs)

        payload = self.get_image_embeddings(PREDICTOR, img_path)

        points, labels, input_box = self.get_smart(points=points, x=x, y=y, box_width=box_width, box_height=box_height,
                                                   labels=labels, smart_annotation=smart_annotation,
                                                   input_box=input_box, **kwargs)

        print(f"point coords are {None if points.size == 0 else points}")
        print(f"labels are {np.array(labels)}")
        print(f"box is {input_box}")
        masks, _, _ = PREDICTOR.predict(
            point_coords=None if points.size == 0 else points,
            point_labels=np.array(labels),
            box=input_box,
            multimask_output=False
        )

        predictions = self.get_results(masks=masks, width=width, height=height, alias=alias, from_name=from_name,to_name=to_name)

        return predictions

    def get_tasks(self, tasks, height, width, context, **kwargs):
        # declaring what will be used in the for annotations
        points = np.empty((0, 2), dtype=int)
        labels = []
        input_box = None
        alias = None  # helps find which while label you are doing
        box_width = None
        box_height = None

        # if there are previous tasks, try those
        try:
            tasks = tasks[0]["drafts"][0]["result"]

            # for each task, get the corresponding information for the current label
            # make sure that the beginning of the label matches the number of the brush
            smart_type = context['result'][0]['type']
            current_label = int(context['result'][0]['value'][smart_type][0])
            alias = abs(current_label)

            print(f'Alias is {alias}, current label is {current_label}')

            for task in tasks:
                type = task["type"]
                if type == 'brush_labels' or type == "brushlabels":
                    continue
                try:
                    if task["value"]["format"] == "rle":
                        continue
                except:
                    print("")
                value = task["value"]
                task_val = int(value[type][0])

                # check if the current task matches the labels for the keypoint generated one
                if abs(int(value[type][0])) == alias:
                    x = value["x"] * width / 100
                    y = value["y"] * height / 100
                    if type == "keypointlabels":
                        new_point = np.array([[int(x), int(y)]])
                        points = np.vstack((points, new_point))
                        if task_val < 0:
                            labels.append(0)
                        elif task_val > 0:
                            labels.append(1)

                    if type == "rectanglelabels":
                        box_width = value["width"] * width / 100
                        box_height = value["height"] * height / 100
                        input_box = np.array([int(x), int(y), int(box_width + x), int(box_height + y)])
                        input_box = input_box[None, :]
        except Exception as e:
            print(e)
            print("No previous tasks")

        return points, labels, input_box, alias, box_width, box_height

    def get_smart(self, points, x, y, box_width, box_height, labels, smart_annotation, input_box, **kwargs):
        # for bounding boxes
        if smart_annotation == "rectanglelabels":
            input_box = np.array([int(x), int(y), int(box_width + x), int(box_height + y)])
            input_box = input_box[None, :]
            print(f"the x and y is {x}, {y} and the box width and height are {box_width + x}, {box_height + y}")

        # for keypoints
        if smart_annotation == "keypointlabels":
            new_point = np.array([[int(x), int(y)]])
            points = np.vstack((points, new_point))
            label = int(kwargs['context']['result'][0]['value']['labels'][0])
            if label > 0:
                labels.append(1)
            if label < 0:
                labels.append(0)

        return points, labels, input_box

    def get_smart_position(self, width, height, box_width, box_height, **kwargs):
        # now, getting information for the smart keypoint or rectangle label
        smart_annotation = kwargs['context']['result'][0]['type']
        if smart_annotation == "rectanglelabels":
            box_width = kwargs['context']['result'][0]['value']['width'] * width / 100
            box_height = kwargs['context']['result'][0]['value']['height'] * height / 100

        # getting x and y coordinates of the keypoint or bounding box starting position
        x = kwargs['context']['result'][0]['value']['x'] * width / 100
        y = kwargs['context']['result'][0]['value']['y'] * height / 100

        return x, y, box_width, box_height, smart_annotation

    def get_results(self, masks, width, height, alias, from_name, to_name):
        mask = masks[0].astype(np.uint8)
        results = []
        predictions = []

        label_id = ''.join(random.SystemRandom().choice(
            string.ascii_uppercase + string.ascii_lowercase + string.digits)),  # creates a random ID for your label everytime so no chance for errors

        # converting the mask from the model to RLE format which is usable in Label Studio
        mask = mask * 255
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
                "brushlabels": [f'{alias}_brush'],
            },
            "type": "brushlabels",
            "id": label_id,
            "readonly": False,
        })

        # returning the result from the prediction and passing it to show on the front-end
        predictions.append({"result": results,
                            "model_version": "vit_h"
                            })

        return predictions
