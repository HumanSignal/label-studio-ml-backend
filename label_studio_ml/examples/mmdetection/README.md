---
title:
type: blog
order: 40
---

This [Machine Learning backend](https://labelstud.io/guide/ml.html) allows you to automatically prelabel your images with bounding boxes. It is powered by amazing [OpenMMLab MMDetection library](https://github.com/open-mmlab/mmdetection), which gives you access to many existing state-of-the-art models like FasterRCNN, RetinaNet, YOLO and others. 

Check this installation guide and then play around them, pick up the best model that suits your current dataset!


## Start using it

1. [Install the model locally](#Installation)

2. Run Label Studio, then go to the **Model** page. Paste the selected ML backend URL then click on **Add Backend**.

3. Go to **Setup** page, use `COCO annotation` template or `Bbox object detection`. O
   Optionally you can modify label config with `predicted_values` attribute. It provides a list of COCO labels separated by comma. If object detector outputs any of these labels, they will be translated to the actual label name from `value` attribute.

    For example:
    
    ```xml
    <Label value="Airplane" predicted_values="airplane"/>
    <Label value="Car" predicted_values="car,truck"/>
    ```
   
means that
- if COCO object detector predicts bbox with label `"airplane"`, you'll finally see the label `"Airplane"`.
- if it predicts `"car"` or `"truck"` - they will be squashed to `"Car"` label.

[Here is](#The-full-list-of-COCO-labels) the full list of COCO labels for convenience.


## Installation

1. Setup MMDetection environment following [this installation guide](https://mmdetection.readthedocs.io/en/v1.2.0/INSTALL.html). Depending on your OS, some of the dependencies could be missed (gcc-c++, mesa-libGL) - install them using your package manager.

2. Create and initialize directory `./coco-detector`:

    ```bash
    label-studio-ml init coco-detector --from label_studio_ml/examples/mmdetection/mmdetection.py
    ```

3. Download `config_file` and `checkpoint_file` from MMDetection model zoo (use [recommended Faster RCNN for quickstarting](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#inference-with-existing-models)).
Note that `config_file` should be located within cloned [mmdetection repo](https://github.com/open-mmlab/mmdetection).


Now depending on your specific use case, there are different settings:

## Run ML backend on the same machine as Label Studio

#### Images are uploaded in Label Studio UI
   ```bash
   label-studio-ml start coco-detector --with \
   config_file=mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
   checkpoint_file=/absolute/path/to/downloaded/checkpoint.pth
   ```
In this case, ML backend reads images from the default data upload folder.
If you change the default upload folder (e.g. by setting `LABEL_STUDIO_BASE_DATA_DIR=/custom/data/path`), you need to explicitly specify upload folder, e.g.:

   ```bash
   label-studio-ml start coco-detector --with \
   config_file=mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
   checkpoint_file=/absolute/path/to/downloaded/checkpoint.pth \
   image_dir=/custom/data/path/media/upload
   ```

#### Images are specified as remote URLs

   ```bash
   label-studio-ml start coco-detector --with \
   config_file=mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
   checkpoint_file=/absolute/path/to/downloaded/checkpoint.pth
   ```

## Run ML backend server on the different machine as Label Studio

When running ML backend on a separate server instance and connecting it to Label Studio app via remote hostname URL, one thing you need to keep in mind dealing with upload images is that your ML backend server needs to get the full image URLs.
For this case, you need to provide Label Studio hostname before running ML backend:

   ```bash
   LABEL_STUDIO_HOSTNAME=http://my.label-studio.com:8080 label-studio-ml start coco-detector --with \
   config_file=mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
   checkpoint_file=/absolute/path/to/downloaded/checkpoint.pth
   ```


## Other parameters

#### GPU support
It's highly recommended to use device=gpu:0 if you have a GPU available - it will significantly speed up image prelabeling. For example:
   ```bash
   label-studio-ml start coco-detector --with \
   config_file=mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
   checkpoint_file=/absolute/path/to/downloaded/checkpoint.pth \
   device=gpu:0
   ```

#### Bounding box thresholding

Feel free to tune `score_threshold` - lower values increase sensitivity but produce more noise.

   ```bash
   label-studio-ml start coco-detector --with \
   config_file=mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
   checkpoint_file=/absolute/path/to/downloaded/checkpoint.pth \
   score_threshold=0.5
   ```
     

### The full list of COCO labels
```text
airplane
apple
backpack
banana
baseball_bat
baseball_glove
bear
bed
bench
bicycle
bird
boat
book
bottle
bowl
broccoli
bus
cake
car
carrot
cat
cell_phone
chair
clock
couch
cow
cup
dining_table
dog
donut
elephant
fire_hydrant
fork
frisbee
giraffe
hair_drier
handbag
horse
hot_dog
keyboard
kite
knife
laptop
microwave
motorcycle
mouse
orange
oven
parking_meter
person
pizza
potted_plant
refrigerator
remote
sandwich
scissors
sheep
sink
skateboard
skis
snowboard
spoon
sports_ball
stop_sign
suitcase
surfboard
teddy_bear
tennis_racket
tie
toaster
toilet
toothbrush
traffic_light
train
truck
tv
umbrella
vase
wine_glass
zebra
```
