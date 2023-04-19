from label_studio_ml.model import LabelStudioMLBase
from label_studio_converter import brush
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from label_studio_ml.utils import get_image_local_path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import string
import datetime
import os

def load_my_model():
        """
        Loads the Segment Anything model on initializing Label studio, so if you call it outside MyModel it doesn't load every time you try to make a prediction
        Returns the predictor object. For more, look at Facebook's SAM docs
        """
        # if you're not using CUDA, use "cpu" instead
        device = "cuda"

        # Note: YOU MUST HAVE THE MODEL SAVED IN THE SAME DIRECTORY AS YOUR BACKEND
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")

        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor

PREDICTOR = load_my_model()

class MyModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
    
    def predict(self, tasks, **kwargs):
        """ Returns the predicted mask for a smart keypoint that has been placed."""

        results = []
        predictions = []
        predictor = PREDICTOR

        image_url = tasks[0]['data']['image']


        # getting the height and width of the image that you are annotating real-time 
        height = kwargs['context']['result'][0]['original_height']
        width = kwargs['context']['result'][0]['original_width']

        # getting x and y coordinates of the keypoint
        x = kwargs['context']['result'][0]['value']['x'] * width / 100
        y = kwargs['context']['result'][0]['value']['y'] * height / 100

        # label that you selected with the keypoint. If this is running into error, use the second line of code instead
        label = kwargs['context']['result'][0]['value']['labels'][0]
        # keypointlabel = kwargs['context']['result'][0]['value']['keypointlabels'][0]


        # loading the image you are annotating in local. The image_dir MUST MATCH THE DIRECTORY WHERE YOU IMPORTED YOUR ANNOTATIONS INTO LABEL STUDIO FROM
        image_path = get_image_local_path(image_url, image_dir=os.getcwd())
        split = image_path.split('-')[-1]
        image = cv2.imread(f"./{split}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # retriving predictions from SAM. For more info, look at Facebook's SAM docs
        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords=np.array([[int(x), int(y)]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )

        mask = masks[0].astype(np.uint8) # each mask has shape [H, W]

        # converting the mask from the model to RLE format which is usable in Label Studio
        mask = mask * 255
        rle = brush.mask2rle(mask)

        # returning the result from the prediction and passing it to show on the front-end
        results.append({
            "from_name": self.from_name,
            "to_name": self.to_name,
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "value": {
                "format": "rle",
                "rle": rle,
                "brushlabels": [label],
            },
            "type": "brushlabels",
            "id": ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)), # creates a random ID for your label every time
            "readonly": False,
        })
        predictions.append({"result": results,
                            "model_version": "vit_h"
        })

        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        return {'random': random.randint(1, 10)}