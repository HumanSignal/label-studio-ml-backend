This project integrates the YOLOv8 model with Label Studio.


## How The Project Works

This project helps you detect objects in Label Studio by doing two things.

1 - Uses a pretrained YOLOv8 model on Google's Open Images V7 (OIV7) to provide a pretrained model on 600 classes!
2 - Use a custom model for classes in cases that don't fit under the 600 classes in the OIV7 dataset

While annotating in label studio, you predefine which one of your labels overlap with the first pretrained model and custom labels that don't fit under the 600 classes are automatically used in the second custom model for predictions that is trained as you submit annotations in Label Studio. 

Predictions are then gathered using the OIV7 pretrained model and the custom model in Label Studio in milliseconds, where you can adjust annotations and fine tune your custom model for even more precise predictions. 


## Setup

1. Defining Classes for Pretrained and Custom Models

Edit your labeling config to something like the following

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="cats" background="red"/>
    <Label value="cars" background="purple"/>
    <Label value="taxi" background="orange"/>
    <Label value="lights" background="green"/>
  </RectangleLabels>
</View>
```

In the `label_to_coco.yml` edit the dictionary to where the keys are the exact names of your rectangular labels in label studio and the values are the exact names of the same classes in [open-images-v7.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/open-images-v7.yaml).

Any classes in your labeling config that you do not add to `label_to_coco.yml` will be trained using the second, custom model.

Note: if you leave this YAML file empty with no keys and values, only the custom model will be trained and then used for predictions. In such a case, the model trained on 600 classes will not be used at all.

2. Editing `docker-compose.yml`

Set `LABEL_STUDIO_HOST` to your private IP address (which starts with 192 so ex. 192.168.1.1) with the port that label studio is running on. For example, your docker compose may look like `LABEL_STUDIO_HOST=192.168.1.1:8080`

Set `LABEL_STUDIO_ACCESS_TOKEN` by going to your Label Studio Accounts & Settings, and then copying the Access Token. Paste it into the docker file. Ex. `LABEL_STUDIO_ACCESS_TOKEN=cjneskn2keoqpejleed8d8frje9992jdjdasvbfnwe2jsx`

3. Running the backend

Run `docker compose up` to start the backend. Under the `Machine Learning` settings in your project in Label Studio enter the following URL while adding the model: `http://{your_private_ip}:9090`. Note: if you changed the port before running the backend, you will have to change it here as well. 



## Notes

If you would like to save your model inside of your docker container or move it into your local machine, you will need to access the terminal of your docker container. See how to do this [here](https://stackoverflow.com/a/30173220).

If you want to train a new custom model, move the `yolov8n(custom).pt` out of your container's directory. It will automatically realize there is no custom model, and will create a new one from scratch to use when training custom models. 