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

import shutil


"""USE THIS TO UPDATE WHICH MODEL YOU ARE USING"""
# TODO: use the best.pt saved to load nstead
# https://github.com/ultralytics/ultralytics/issues/2750#issuecomment-1556847848


LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")


"""TODO

2. let user give overlaps in the docker file

## ^ after the above, send a PR


"""


class YOLO(LabelStudioMLBase):

    def __init__(self, project_id, **kwargs):
        super(YOLO, self).__init__(**kwargs)
        self.device = "cuda" if torch.cuda.is_available else "cpu" # can to mps

        print(self.label_config)
        # print(self.parsed_label_config)



        # parsed = self.parsed_label_config
        classes = parsed['label']['labels']


        with open("ls_config.yml", "r") as file:
            ls_config = yaml.safe_load(file)


        label_to_COCO = ls_config["labels_to_coco"]
        self.NEW_START = True if label_to_COCO['NEW_START']=='True' else False
        self.JUST_CUSTOM = True if label_to_COCO['JUST_CUSTOM']=='True' else False

        print(f"{self.NEW_START} and {self.JUST_CUSTOM}")


        # TODO: get from docker
        # label_to_COCO = {
        #     "cats": "Cat",
        #     "lights": "Traffic light",
        #     "cars": "Car",
        # }
        


        # defining model start

        if not self.JUST_CUSTOM: 
            self.pretrained_model = YOLO('yolov8n-oiv7.pt')

        # add logic that creates it from regular here
        if self.NEW_START:
            shutil.copyfile('./yolov8n.pt', 'yolov8n(custom).pt')
            self.custom_model = YOLO('yolov8n(custom).pt')
            FIRST_USE = True
        else:
            self.custom_model = YOLO('yolov8n(custom).pt')


        self.COCO_to_label = {v:k for k, v in label_to_COCO.items()}

        first_label_classes = list(label_to_COCO.keys()) # raw labels from labelling config
        second_label_classes = [x for x in classes if x not in set(first_label_classes)] # raw labels from labelling config



        # if they change the labelling config, it shouldn't automatically destroy everything
        input_file = "custom_config.yml"
        with open(input_file, "r") as file:
            data = yaml.safe_load(file)
        
        # obious way to toggle
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
        if not self.JUST_CUSTOM:
            results_1 = self.pretrained_model.predict(source=imgs) # define model earlier
        else:
            results_1 = None

        if not self.FIRST_USE:
            results_2 = self.custom_model.predict(source=imgs)
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
        results = []

        print(f"the to and from names are {self.from_name} and {self.to_name}")

        for box, class_num, conf in zip(boxes, classes, confidences):

            label_id = str(uuid4())[:9]

            x, y, w, h = box

            height, width = length

            if pretrained:
                name = num_to_names_dict[int(class_num)]
                label = self.COCO_to_label.get(name)
                print(f"class num is {class_num} and name is {name}")
            else: # then, we are using the custom model
                label = num_to_names_dict[int(class_num)]
                
            print(f"the labellllllll is {label}")

            if label==None:
                print(f"it's none {label}")
                continue
            
            print("but we're still going")
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
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
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


                # doing the inverse of these operation, but keeping it normalized
                # 'width': w / width * 100, # this is correcrt
                # 'height': h / height * 100, # this is also correct
                # 'x': (x - 0.5*w) / width * 100,
                # 'y': (y-0.5*h) / height * 100

                # so, in YOLO format, we just need to to be normalize to 1

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
        

        results = self.custom_model.train(data='custom_config.yml', epochs = 1, imgsz=640)

        FIRST_USE = False

        # indexing error if there is only one image
        # do two images or more images for no error
        
        # remove all these files so train starts from nothing next time
        # self.remove_train_files(all_new_paths)



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
        # torch.save(model.state_dict(), 'yolov8n(testing).pt')
    
    def remove_train_files(self, file_paths):
        for path in file_paths:
            os.remove(path)


