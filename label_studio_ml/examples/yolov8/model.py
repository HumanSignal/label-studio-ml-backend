from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path

import os
from PIL import Image
from uuid import uuid4

from ultralytics import YOLO
import torch

import os
import yaml


LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")


# change config file depending on how many classes there are in the saved model

NEW_START = False

if NEW_START:
    model = YOLO('yolov8n.pt')
else:
    model = YOLO('./yolov8n.yml')
    # model = torch.hub.load('ultralytics/yolov8', 'yolov8n', classes=2)
    model.load_state_dict(torch.load('yolov8n(testing).pt'))
    # model.eval()

# TODO:
# figure out how to integrate class names for things not predicted


class YOLO(LabelStudioMLBase):

    def __init__(self, project_id, **kwargs):
        super(YOLO, self).__init__(**kwargs)
        self.device = "cuda" if torch.cuda.is_available else "cpu" # can to mps


        # read this from the config file
        self.class_to_name = {
            0: "cats",
            1: "dog"
        }

        # print(self.label_config)
        # print(self.parsed_label_config)


        # TODO: this should all be done before loading the model


        # create a new YAML file for training
        parsed = self.parsed_label_config
        classes = parsed['label']['labels']

        self.class_to_name = {i:v for i,v in enumerate(classes)}
        self.name_to_class = {v:k for k, v in self.class_to_name.items()}

        input_file = "train_config.yml"
        with open(input_file, "r") as file:
            data = yaml.safe_load(file)
        
        data["names"] = self.class_to_name

        with open(input_file, "w") as file:
            yaml.dump(data, file, default_flow_style=False)

        # TODO: adjust num_classes in the yolov8 yaml file as well
        weights_file = "yolov8.yml"
        with open(weights_file, "r") as file:
            weights = yaml.safe_load(file)
        
        weights["nc"] = len(self.class_to_name)

        with open(weights_file, "w") as file:
            yaml.dump(weights, file, default_flow_style=False)
        
        # model = YOLO('./yolov8n.yml')
        # # model = torch.hub.load('ultralytics/yolov8', 'yolov8n', classes=2)
        # model.load_state_dict(torch.load('yolov8n(testing).pt'))



        print(classes)
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        # print(f'''\
        # Run prediction on {tasks}
        # Received context: {context}
        # Label config: {self.label_config}
        # Parsed JSON Label config: {self.parsed_label_config}''')

                # Project ID: {self.project_id}
        # model.eval()

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
        results = model.predict(source=imgs) # define model earlier

        # each item will be the predictions for a task
        predictions = []

        # basically, running this loop for each task
        for (result, len) in zip(results, lengths):
            boxes = result.boxes.cpu().numpy()

            print(result.names)

            print(f"the confidences are {boxes.conf}")

            predictions.append(self.get_results(boxes.xywh, boxes.cls, len, boxes.conf, result.names))
        


        # # TODO: here figure out what type of prediction we are looking for -> classification, segmentation, bounding boxes, etc.
        # context = "classification"

        # if context=="classification":
        #     model = YOLO('yolov8n-cls.pt')
        # img = 'https://ultralytics.com/images/bus.jpg'
        # results = model(img)

        # find how to get images and labels from different places YOLO

        return predictions

    def get_results(self, boxes, classes, length, confidences, names_dict):
        results = []

        for box, name, conf in zip(boxes, classes, confidences):

            label_id = str(uuid4())[:9]

            x, y, w, h = box

            height, width = length


            results.append({
                'id': label_id,
                'from_name': self.from_name,
                'to_name': self.to_name,
                'original_width': int(width),
                'original_height': int(height),
                'image_rotation': 0,
                'value': {
                    'rotation': 0,
                    # 'rectanglelabels': [self.class_to_name[f"{int(name)}"]],
                    'rectanglelabels': [names_dict[int(name)]],
                    'width': w / width * 100, # this is correcrt
                    'height': h / height * 100, # this is also correct
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
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # model.train()

        print(f"the fit is {data}")
        # results = data["annotation"]["result"]
        # ^ this will be a list of all the rectangles you are fine tuning

        # figure out how to do this with multiple images at once

        results = data['annotation']['result']
        data = data['task']['data']
        image_path = data['image']
        image_paths = [image_path]

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
            
            # im_save = img.save(f"dataset/images/{name}")

        sample_img_path = true_img_paths[0]

        img = Image.open(sample_img_path)

        project_path = sample_img_path.split("/")[:-1]
        image_name = sample_img_path.split("/")[-1]

        print(f"image name is {image_name}")

        img1 = img.save(f"./datasets/temp/images/{image_name}")
        img2 = img.save(f"./datasets/temp/images/(2){image_name}")

        # these rename the directories for label studio format
        # os.rename(project_path.split("/"), project_path.replace((f"/{project_path.split('/')[-2]}/"), "images"))
            
        # # making the labels directory
        # os.mkdir(img_path.split("/")[:-1], "labels")

        # now saving text file labels
        txt_name = (image_path.split('/')[-1]).split('.')[0]

        with open(f'./datasets/temp/labels/{txt_name}.txt', 'w') as f:
                f.write("")
        with open(f'./datasets/temp/labels/(2){txt_name}.txt', 'w') as f:
            f.write("")


        for result in results:
            value = result['value']
            label = value['rectanglelabels'][0]
            
            # these are out of 100, so you need to convert them back
            x = value['x']
            y = value['y']
            width = value['width']
            height = value['height']

            orig_width = result['original_width']
            orig_height = result['original_height']


            # doing the inverse of these operation, but keeping it normalized
            # 'width': w / width * 100, # this is correcrt
            # 'height': h / height * 100, # this is also correct
            # 'x': (x - 0.5*w) / width * 100,
            # 'y': (y-0.5*h) / height * 100

            # so, in YOLO format, we just need to to be normalize to 1

            w = width / 100
            h = height / 100
            trans_x = (x / 100) + 0.5 * w
            trans_y = (y / 100) + 0.5 * h

            # now getting the class label 
            label = self.name_to_class.get(label)

            with open(f'./datasets/temp/labels/{txt_name}.txt', 'a') as f:
                f.write(f"{label} {trans_x} {trans_y} {w} {h}\n")
            with open(f'./datasets/temp/labels/(2){txt_name}.txt', 'a') as f:
                f.write(f"{label} {trans_x} {trans_y} {w} {h}\n")
        

        results = model.train(data='train_config.yml', epochs = 1, imgsz=640)
        # indexing error if there is only one image
        # do two images or more images for no error



        # you can send a list of images into the YAML file
        # so we can just save thelabels int eh data upload directory?
        # for now let's just work on saving all the images in a new directory and then using that?

        # this is assuming all images are in a list


        # TODO: make sure this rewrites whatever images were already there
        # having so many images rewritten is a time consuming process - think of a way to mitigate this

        """Here is the process
        
        - whatever project you are in, rename that project to images
        - create a labels directory there as well 
        - make sure the above doesn't break label studio 
        - put the label text files in there
        - create a images.txt file that contains only the paths of the images that have been chosen by the user
        - remove the txt files when done and labels and RENAME back the labels directory
        """ 

        # setting the new model
        # self.set("new_model", model)
        # set a new model version


        print(f"the event is {event}") # ANNOTATION CREATED
        print(f"kwargs are {kwargs}")

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')


        # setting the new model -> this is only key settting so that we can retrieve it later
        # self.set("new_model", model)
        
        # save the model to the directory
        torch.save(model.state_dict(), 'yolov8n(testing).pt')

        # return {'model_file': 'yolov8n(testing).pt'}

