from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from PIL import Image
import os
from label_studio_ml.utils import (get_choice, get_env, get_local_path,
                                   get_single_tag_keys, is_skipped)
from io import BytesIO
import requests

MODEL_DIR = os.getenv('MODEL_DIR')
HOSTNAME = get_env('HOSTNAME', 'http://localhost:8080')
API_KEY = ""

print('=> LABEL STUDIO HOSTNAME = ', HOSTNAME)
print('=> MODEL_DIR = ', MODEL_DIR)

class Yolov8Backend(LabelStudioMLBase):
    def __init__(self, project_id=None,**kwargs):
        # Call base class constructor
        super(Yolov8Backend, self).__init__(**kwargs)
        # Initialize self variables
        self.conifg=self.use_label_config('''
        <View>
            <Image name="image" value="$image" zoom="false"/>
            <RectangleLabels name="label" toName="image">
                <Label value="single_door" background="#FFA39E"/>
                <Label value="double_door" background="##25d30d"/>
            </RectangleLabels>
        </View>
            ''')
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        self.labels = ['single_door', 'double_door']

        # Load model
        #model_path=os.path.join(MODEL_DIR,"best.pt")
        current_path = os.getcwd()
        print(current_path)
        model_path='./my_ml_backend/best.pt'
        self.model = YOLO(model_path)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:

        task = tasks[0]

        # Getting URL of the image
        image_url = task['data'][self.value]
        full_url = HOSTNAME + image_url
        print("FULL URL: ", full_url)

        # Header to get request
        header = {
            "Authorization": "Token " + API_KEY}
        
        # Getting URL and loading image
        image = Image.open(BytesIO(requests.get(
            full_url, headers=header).content))
        # Height and width of image
        original_width, original_height = image.size
        
        # Creating list for predictions and variable for scores
        predictions = []
        score = 0

    # Getting prediction using model
        results = self.model.predict(image)

        # Getting mask segments, boxes from model prediction
        for result in results:
            for i, box in enumerate(result.boxes):
                
                # 2D array with box xyxy posistions
                box_pos = box.xyxy.cpu().tolist()
                box_conf = box.conf.cpu()
                box_cls = box.cls.cpu()
                x1=box_pos[0][0]
                y1=box_pos[0][1]
                x2=box_pos[0][2]
                y2=box_pos[0][3]
                # x= x1 
                # y= y1
                # width=x2-x1
                # height=y2-y1
                x= x1 / original_width * 100.0
                y= y1 / original_height * 100.0
                width=(x2-x1)/ original_width * 100.0
                height=(y2-y1)/ original_height *100.0

                # Adding dict to prediction
                predictions.append({
                    "from_name" : self.from_name,
                    "to_name" : self.to_name,
                    "id": str(i),
                    "type": "rectanglelabels",
                    "score": box_conf.item(),
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "x":x,
                        "y":y,
                        "width":width,
                        "height":height,
                        "rectanglelabels": [self.labels[int(box_cls.item())]]
                    }})

                # Calculating score
                score += box.conf.item()


        print(10*"#", "Returned Prediction", 10*"#")

        # Dict with final dicts with predictions
        if predictions:
            final_prediction = [{
                "result": predictions,
                "score": score / (i + 1),
                "model_version": "v8x"
            }]
        else:
            final_prediction = [{
                "result": predictions,
                "model_version": "v8x"
            }]

        return final_prediction


        




# if __name__ == '__main__':
#     # test the model
#     model = Yolov8Backend()
#     model.use_label_config('''
# <View>
#     <Image name="image" value="$image" zoom="false"/>
#     <RectangleLabels name="label" toName="image">
#         <Label value="single_door" background="#FFA39E"/>
#         <Label value="double_door" background="#D4380D"/>
#     </RectangleLabels>
# </View>
#     ''')
#     results = model.predict(
#         tasks=[{
#             'data': {
#                 'image': 'https://s3.amazonaws.com/htx-pub/datasets/images/125245483_152578129892066_7843809718842085333_n.jpg'
#             }}]
#     )
#     import json
#     print(json.dumps(results, indent=2))