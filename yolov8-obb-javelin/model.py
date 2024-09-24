from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO

class YOLOv8OBBModel(LabelStudioMLBase):
    """Custom ML Backend model for YOLOv8 OBB"""

    def setup(self):
        """Configure any parameters of your model here, including loading the YOLO model"""
        self.set("model_version", "0.0.1")
        
        # Path to the trained YOLOv8 OBB model (best.pt)
        model_path = 'C:/Users/sinja/OneDrive/Dokumenti/Sinja/indigolabs/Athletics/athletics_project/project/runs/obb/train2/weights/best.pt'
        self.model = YOLO(model_path)
        print("Model loaded successfully.")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Inference logic using the YOLOv8 OBB model."""
        
        print(f"Run prediction on {tasks}")
        print(f"Received context: {context}")
        print(f"Project ID: {self.project_id}")
        print(f"Label config: {self.label_config}")
        print(f"Parsed JSON Label config: {self.parsed_label_config}")
        print(f"Extra params: {self.extra_params}")

        predictions = []

        for task in tasks:
            # Get the image URL from the task
            image_url = task['data']['image']
            image = self.load_image(image_url)

            # Run YOLO model to get the results
            results = self.model(image)

            # Process results (YOLOv8 returns xyxy format, OBB requires angle calculation)
            obb_predictions = self.extract_obbs(results)

            predictions.append({
                "result": obb_predictions
            })

        # Return predictions in Label Studio format
        return ModelResponse(predictions=predictions)

    def load_image(self, image_url):
        """Load image from Label Studio's image URL"""
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return img

    def extract_obbs(self, results):
        """Convert YOLOv8 results to OBB format for Label Studio"""
        obbs = []
        for result in results.xyxy[0]:  # YOLOv8 result in xyxy format
            x_min, y_min, x_max, y_max = result[:4]
            confidence = result[4]
            label = int(result[5])  # Class label

            # Calculate the bounding box properties and angle (for OBB)
            obb = {
                "from_name": "label",  # Match this with your Label Studio config
                "to_name": "image",    # Match this with your Label Studio config
                "type": "rectanglex",  # Rectanglex is for rotated bounding boxes
                "value": {
                    "x": float(x_min),
                    "y": float(y_min),
                    "width": float(x_max - x_min),
                    "height": float(y_max - y_min),
                    "rotation": 0.0  # Adjust this if necessary for OBB
                },
                "score": float(confidence),
            }

            obbs.append(obb)

        return obbs

    def fit(self, event, data, **kwargs):
        """Optional: handle model updates or annotations"""
        
        # Retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # Store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')
