# Quick usage

For quick usage run docker-compose in your working directory:

```bash
docker-compose up -d
```

# Reference to tutorial

See the tutorial in the documentation for building your own image and advanced usage:

https://github.com/heartexlabs/label-studio/blob/master/docs/source/tutorials/object-detector.md


# Labeling Config
There are two possible variants of labeling configurations that can be used:

1. In this example, you can provide labels "as is" and they will be automatically mapped to MMDetection model's labels.
This will work for simple use cases. For example, Label Studio `Airplane` maps to MMDetection `airplane`.

```
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Airplane" background="green"/>
    <Label value="Car" background="blue"/>
  </RectangleLabels>
</View>
```

2. More complex labeling config with `predicted_values`:

```
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Vehicle" predicted_values="airplane,car" background="green"/>
  </RectangleLabels>
</View>
```

In this example, you can combine multiple labels into one Label Studio label. For example, Label Studio Vehicle maps to MMDetection "airplane" and "car".


# Run without docker

> These steps provided by @raash1d [in this issue](https://github.com/heartexlabs/label-studio-ml-backend/issues/167#issuecomment-1495061050). Note: the patch from the comment is already applied, except hardcoding of label_config into kwargs.

**It's highly recomended to use docker, it allows to avoid lots of dependency problems!**

1. Clone the Label Studio ML Backend repository in your directory of choice
```
git clone https://github.com/heartexlabs/label-studio-ml-backend
cd label-studio-ml-backend/label_studio_ml/examples/mmdetection-3
```

2. Create a virtual environment using venv and install all dependencies using pip
```
python -m venv ml-backend
source ml-backend/bin/activate # assuming you're on bash or zsh
pip install -r requirements.txt
```

3. Install and Download mmdet related dependencies in the virtual environment
```
mim install mmengine
mim download mmdet --config yolov3_mobilenetv2_8xb24-320-300e_coco --dest .
mim install mmcv==2.0.0rc3
```

4. Export required variables
```
export checkpoint_file=yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth
export config_file=yolov3_mobilenetv2_8xb24-320-300e_coco.py
```

5. Run the following command to start your ML backend
```
label-studio-ml start --root-dir .. mmdetection-3 --kwargs hostname=http://<ip.of.your.label.studio.server>:8080 access_token=<access_token> [score_threshold=0.5] [--check]
```

* Use this guide to find out your access token: https://labelstud.io/guide/api.html
* If you're running label studio and the ML backend on the same machine, then the IP of your label studio server could also be "localhost". Try it, I haven't tested this.
* `score_threshold` and `--check` are optional.
* You can use and increased value of `score_threshold` parameter when you see a lot of unwanted detections or lower its value if you don't see any detections.
* The `--check` parameter allows you to check if all requirements to run the backend have been fulfilled
