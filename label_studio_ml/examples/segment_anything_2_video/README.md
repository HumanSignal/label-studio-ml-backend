<!--
---
title: SAM2 with Videos
type: guide
tier: all
order: 15
hide_menu: true
hide_frontmatter_title: true
meta_title: Using SAM2 with Label Studio for Video Annotation
categories:
    - Computer Vision
    - Video Annotation
    - Object Detection
    - Segment Anything Model
image: "/tutorials/sam2-video.png"
---
-->

# Using SAM2 with Label Studio for Video Annotation

This guide describes the simplest way to start using **SegmentAnything 2** with Label Studio.

This repository is specifically for working with object tracking in videos. For working with images, 
see the [segment_anything_2_image repository](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image)

![sam2](./Sam2Video.gif)

## Before you begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart). 

This tutorial uses the [`segment_anything_2_video` example](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_video). 

## Running from source

1. To run the ML backend without Docker, you have to clone the repository and install all dependencies using pip:

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend
pip install -e .
cd label_studio_ml/examples/segment_anything_2_video
pip install -r requirements.txt
```

2. Download [`segment-anything-2` repo](https://github.com/facebookresearch/segment-anything-2) into the root directory. Install SegmentAnything model and download checkpoints using [the official Meta documentation](https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#installation). Make sure that you complete the steps for downloadingn the checkpoint files! 
   If you want to install segment-anything-2 repo in a different directory, you must the SEGMENT_ANYTHING_2_PATH environment variable to the path of the segment-anything-2 directory.

3. Export the following environment variables (fill them in with your credentials!):
- LABEL_STUDIO_URL: the http:// or https:// link to your label studio instance (include the prefix!) 
- LABEL_STUDIO_API_KEY: your api key for label studio, available in your profile. 
- MAX_FRAMES_TO_TRACK: the maximum number of frames to track in a video each time the model is called.
- PROMPT_TYPE: the type of prompt you want to use: "box" or "point". By choosing "box", you will be able to draw a box around the object you want to track. By choosing "point", five key points will be automatically selected within the box to help identify the object.
- SEGMENT_ANYTHING_2_REPO_PATH: the path to the segment-anything-2 repo. Default is `segment-anything-2`, this means that the segment-anything-2 repo is in the same directory as the label-studio-ml-backend example folder.

4. Then you can start the ML backend on the default port `9090`:

```bash
cd ../
label-studio-ml start ./segment_anything_2_video
```
Note that if you're running in a cloud server, you'll need to run on an exposed port. To change the port, add `-p <port number>` to the end of the start command above.
5. Connect running ML backend server to Label Studio: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL. Read more in the official [Label Studio documentation](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio).
 Again, if you're running in the cloud, you'll need to replace this localhost location with whatever the external ip address is of your container, along with the exposed port.

# Labeling Config
For your project, you can use any labeling config with video properties. Here's a basic one to get you started!

```xml     
<View>
    <Labels name="videoLabels" toName="video" allowEmpty="true">
        <Label value="Player" background="#11A39E"/>
        <Label value="Ball" background="#D4380D"/>
    </Labels>

    <!-- Please specify FPS carefully, it will be used for all project videos -->
    <Video name="video" value="$video" framerate="25.0"/>
    <VideoRectangle name="box" toName="video" smart="true"/>
</View>
```

## Known limitations
- As of 8/11/2024, SAM2 only runs on GPU servers. 
- Currently, we do not support video segmentation. 
- Multi-object tracking is enabled, but due to a bug (https://github.com/HumanSignal/label-studio-ml-backend/issues/664), the UI shows the same label for all objects even though the predictions have different labels.
- Be cautious of the UI, as due to a bug (https://github.com/HumanSignal/label-studio/issues/6593), frames are not displayed properly, causing a misalignment between labels and frames. Under the hood, the labels are still being applied to the correct frames but the UI may not display this correctly.

If you want to contribute to this repository to help with some of these limitations, you can submit a PR. 

## Customization

The ML backend can be customized by adding your own models and logic inside the `./segment_anything_2_video` directory. 
