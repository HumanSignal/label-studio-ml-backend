from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
from label_studio_tools.core.label_config import parse_config

import os
from PIL import Image
from uuid import uuid4
from ultralytics import YOLO
import torch
import os
import yaml
import shutil


LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")


with open("label_to_coco.yml", "r") as file:
    ls_config = yaml.safe_load(file)

label_to_COCO = ls_config["labels_to_coco"]

JUST_CUSTOM = True if len(label_to_COCO) == 0 else False

# checks if you have already built a custom model
# if you want to do it for a new task, move this model out of the directory
NEW_START = os.path.isfile('yolov8n(custom).pt')

if not JUST_CUSTOM: 
    pretrained_model = YOLO('yolov8n-oiv7.pt')

if NEW_START:
    shutil.copyfile('./yolov8n.pt', 'yolov8n(custom).pt')
    custom_model = YOLO('yolov8n(custom).pt')
    FIRST_USE = True
else:
    custom_model = YOLO('yolov8n(custom).pt')
    FIRST_USE = False


class YOLO(LabelStudioMLBase):

    def __init__(self, project_id, **kwargs):
        super(YOLO, self).__init__(**kwargs)

        print(f"initializing teh model the kwargs are {kwargs}")
        self.device = "cuda" if torch.cuda.is_available else "cpu" # can to mps


        self.custom_parsed_label_config = parse_config(self.label_config)

        # print(self.label_config)
        # print(self.custom_parsed_label_config)

        self.first_use = FIRST_USE

        print(f"can it gather the config? {self.custom_parsed_label_config} and {self.label_config}")
        parsed = self.custom_parsed_label_config
        classes = parsed['label']['labels']

        print(f"the classes are {classes}")

        self.NEW_START = NEW_START
        self.JUST_CUSTOM = JUST_CUSTOM        

        self.COCO_to_label = {v:k for k, v in label_to_COCO.items()}

        first_label_classes = list(label_to_COCO.keys()) # raw labels from labelling config
        second_label_classes = [x for x in classes if x not in set(first_label_classes)] # raw labels from labelling config


        # if the user changes the labelling config, it shouldn't automatically destroy everything
        # so only change it if we are starting brand new

        input_file = "custom_config.yml"
        with open(input_file, "r") as file:
            data = yaml.safe_load(file)
        
        if self.NEW_START: 

            self.custom_num_to_name = {i:v for i,v in enumerate(second_label_classes)}
            
            data["names"] = self.custom_num_to_name

            with open(input_file, "w") as file:
                yaml.dump(data, file, default_flow_style=False)
        else:
            self.custom_num_to_name = data["names"]
        
        print(f"self class to name is {self.custom_num_to_name}")
        self.custom_name_to_num = {v:k for k, v in self.custom_num_to_name.items()}



        print(classes)
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Inference logic for YOLO model """

        print("getting predictions")

        self.from_name, self.to_name, self.value = self.get_first_tag_occurence('RectangleLabels', 'Image')

        imgs = []
        lengths = []

        # loading all images into lists
        for task in tasks:

            raw_img_path = task['data']['image']

            try:
                img_path = get_image_local_path(
                    raw_img_path,
                    label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                    label_studio_host=LABEL_STUDIO_HOST
                )
                print(f"the real image path is {img_path}")
            except:
                img_path = raw_img_path
            
            img = Image.open(img_path)

            imgs.append(img)

            W, H = img.size
            lengths.append((H, W))

        # predicting from PIL loaded images
        if not self.JUST_CUSTOM:
            results_1 = pretrained_model.predict(source=imgs) # define model earlier
        else:
            results_1 = None


        # we don't want the predictions from the pretrained version of the custom model
        # because it hasn't reshaped to the new classes yet
        if not self.first_use:
            results_2 = custom_model.predict(source=imgs)
        else:
            results_2 = None

        # each item will be the predictions for a task
        predictions = []

        # basically, running this loop for each task
        for res_num, results in enumerate([results_1, results_2]):
            if results == None:
                continue
            
            for (result, len) in zip(results, lengths):
                boxes = result.boxes.cpu().numpy()

                print(result.names) # gives dict matching num to names ex. {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4

                print(f"the confidences are {boxes.conf}")

                pretrained = True if res_num == 0 else False
                # results names
                predictions.append(self.get_results(boxes.xywh, boxes.cls, len, boxes.conf, result.names, pretrained=pretrained))

        return predictions

    def get_results(self, boxes, classes, length, confidences, num_to_names_dict, pretrained=True):
        """This method returns annotation results that will be packaged and sent to Label Studio frontend"""
        
        results = []

        print(f"the to and from names are {self.from_name} and {self.to_name}")

        for box, class_num, conf in zip(boxes, classes, confidences):

            label_id = str(uuid4())[:9]

            x, y, w, h = box

            height, width = length

            if pretrained:
                name = num_to_names_dict[int(class_num)]
                label = self.COCO_to_label.get(name)
            else: # then, we are using the custom model
                label = num_to_names_dict[int(class_num)]
                
            if label==None:
                continue
            
            results.append({
                'id': label_id,
                'from_name': self.from_name,
                'to_name': self.to_name,
                'original_width': int(width),
                'original_height': int(height),
                'image_rotation': 0,
                'value': {
                    'rotation': 0,
                    'rectanglelabels': [label],
                    'width': w / width * 100,
                    'height': h / height * 100,
                    'x': (x - 0.5*w) / width * 100,
                    'y': (y-0.5*h) / height * 100
                },
                'score': conf.item(),
                'type': 'rectanglelabels',
                'readonly': False
            })
        
        return {
            'result': results
        }

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model       
        """

        results = data['annotation']['result']
        data = data['task']['data']
        image_path = data['image']
        image_paths = [image_path]
        all_new_paths = []

        true_img_paths = []
        for raw_img_path in image_paths:
            try:
                img_path = get_image_local_path(
                    raw_img_path,
                    label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                    label_studio_host=LABEL_STUDIO_HOST
                )
            except:
                img_path = raw_img_path
            
            img = Image.open(img_path)

            name = raw_img_path.split("/")[-1]

            true_img_paths.append(img_path)
            
        sample_img_path = true_img_paths[0]

        img = Image.open(sample_img_path)

        project_path = sample_img_path.split("/")[:-1]
        image_name = sample_img_path.split("/")[-1]

        print(f"image name is {image_name}")

        img1 = img.save(f"./datasets/temp/images/{image_name}")
        img2 = img.save(f"./datasets/temp/images/(2){image_name}")

        all_new_paths.append(f"./datasets/temp/images/{image_name}")
        all_new_paths.append(f"./datasets/temp/images/(2){image_name}")

        # now saving text file labels
        txt_name = (image_path.split('/')[-1]).rsplit('.', 1)[0]

        with open(f'./datasets/temp/labels/{txt_name}.txt', 'w') as f:
            f.write("")
        with open(f'./datasets/temp/labels/(2){txt_name}.txt', 'w') as f:
            f.write("")

        all_new_paths.append(f'./datasets/temp/labels/{txt_name}.txt')
        all_new_paths.append(f'./datasets/temp/labels/(2){txt_name}.txt')


        for result in results:

            value = result['value']
            label = value['rectanglelabels'][0]

            if label in self.custom_name_to_num:
            
                # these are out of 100, so you need to convert them back
                x = value['x']
                y = value['y']
                width = value['width']
                height = value['height']

                orig_width = result['original_width']
                orig_height = result['original_height']

                w = width / 100
                h = height / 100
                trans_x = (x / 100) + (0.5 * w)
                trans_y = (y / 100) + (0.5 * h)

                # now getting the class label 
                label_num = self.custom_name_to_num.get(label)

                with open(f'./datasets/temp/labels/{txt_name}.txt', 'a') as f:
                    f.write(f"{label_num} {trans_x} {trans_y} {w} {h}\n")
                with open(f'./datasets/temp/labels/(2){txt_name}.txt', 'a') as f:
                    f.write(f"{label_num} {trans_x} {trans_y} {w} {h}\n")
        

        results = custom_model.train(data='custom_config.yml', epochs = 1, imgsz=640)

        self.first_use = False
        
        # remove all these files so train starts from nothing next time
        self.remove_train_files(all_new_paths)
    
    def remove_train_files(self, file_paths):
        """This cleans the dataset directory"""
        for path in file_paths:
            os.remove(path)