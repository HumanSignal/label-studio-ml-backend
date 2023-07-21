# Interactive Annotation in Label Studio with Segment Anything Model

<img src="https://user-images.githubusercontent.com/106922533/234322629-e583c838-11eb-4261-aaa1-872f1695720c.gif" width="500" />

<img src="https://user-images.githubusercontent.com/106922533/234322576-a24643f8-aeb6-421c-984e-d0d2e2233cd4.gif" width="500" />

Use Facebook's Segment Anything Model with Label Studio!

# Intro

There are two models in this repo that you can use.
1. Advanced Segment Anything Model
2. ONNX Segment Anything Model

The Advanced Segment Anything Model introduces the ability to combined a multitude of different prompts to achieve a prediction, and the ability to use MobileSAM.
- Mix one rectangle label with multiple positive keypoints to refine your predictions! Use negative keypoints to take away area from predictions for increased control.
- Use MobileSAM, and extremely lightweight alternative to the heavy original SegmentAnythingModel, to retrieve predictions. This can run inference within less than a second solely using a laptop without external compute!

The ONNX Segment Anything Model gives the ability to use either a single keypoint or single rectangle label to prompt the original SAM.
- This offers a much faster prediction using the original Segment Anything Model due to using the ONNX version.
- Downside: image size must be specified before using the ONNX model, and cannot be generalized to other image sizes while labelling. Also, does not yet offer the mixed labelling and refinement that AdvancedSAM does.

## Your Choices
**Using AdvancedSAM**
1. *Use with MobileSAM architecture*
-  Pros: very lightweight can be run on laptops, mix many different combinations of input prompts to fine-tune prediction
-  Cons: Little less accuracy than regular SAM
3. _Use with original SAM architecture_
- Pros: higher accuracy than MobileSAM, mix many different combinations of input prompts to fine-tune prediction
- Cons: takes long to gather predictions (~2s to create embedding of an image), heavy and requires access to good GPUs

**Using ONNXSAM**
1. _Use with original SAM architecture_
- Pros: much faster than when you use it in Advanced SAM
- Cons: can only use one smart label per predictions, image size must be defined before generating the ONNX model, cannot label images with different size without running into issues

In addition, AdvancedSAM gives the ability to use MobileSAM, a lightweight version of Segment Anything Model than can be run without much computational power needed. 

# Setup

## Setting Up the Backend

### 1. Clone this repo

Place the images you want to annotate in this project's folder. If you want to use the version of the code that uses slower individual inference times, but has a faster rate on the first label only (not using ONNX), then refer to [this commit instead](https://github.com/shondle/label-studio-ml-backend/tree/4367b18a52a7a494125874467c5e980a6068eca5/label_studio_ml/examples/segment_anything_model)

### 2. Retrieve Label Studio Code

```
git clone https://github.com/heartexlabs/label-studio-ml-backend
cd label-studio-ml-backend

# Install label-studio-ml and its dependencies
pip install -U -e .
```

- [Label Studio Installation Instructions](https://labelstud.io/guide/install.html#Install-with-Anaconda)

### 3. Download SAM

Follow [SAM installation instructions with pip](https://github.com/facebookresearch/segment-anything). 
Then, install the [ViT-H SAM model](https://github.com/facebookresearch/segment-anything)
Then use the SAM installation instructions from above to convert to ONNX and place *into this project's directory*

### 4. Add to your bashrc
```
nano ~/.bashrc
# add the bottom of your bashrc
export ML_TIMEOUT_SETUP=120
```

### 5. Installations
```
pip install label-studio numpy opencv-python label-studio-converter
```

### 6. Start the Backend
```
# change into this project folder from where you are
cd segment_anything_model
python _wsgi.py -p 4243
```

### 7. Run Label Studio
```
label-studio start
```

## Settings on the frontend

1. Create a project and go to settings.
2. Under "Machine Learning" click "Add Model"<br>
3. Under "URL" paste the URL of where the model backend is running (you can find this in the terminal where you started the backend)<br>
4. Switch on "Use for interactive preannotations"<br>
5. Click "Validate and Save"<br>

6. Next -> go to "Labelling Interface". This is on the same side where you chose the "Machine Learning" tab.<br>
7. Choose the code option and paste in the following template-
```
<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="tag" toName="image">
  	<Label value="Banana" background="#FF0000"/>
  	<Label value="Orange" background="#0d14d3"/>
  </BrushLabels>
  <KeyPointLabels name="tag2" toName="image">
    <Label value="Banana" smart="true" background="#000000" showInline="true"/>
    <Label value="Orange" smart="true" background="#000000" showInline="true"/>
    <Label value="Orange Eraser" smart="true" background="#000000" showInline="true"/>
  </KeyPointLabels>
  <RectangleLabels name="tag3" toName="image">
    <Label value="Banana" smart="true" background="#000000" showInline="true"/>
    <Label value="Orange" smart= "true" background="#000000" showInline="true"/>
  </RectangleLabels>
</View>
```

Notes when you change for your use case - 
- Label values must be the same for KeyPointLabels and BrushLabels
- "smart" should be set to the label values for the Keypoints
- You must format the Eraser string the exact same way, mirroring one of the other labels, in order to use this feature. 


# Creating the Annotation

1. After finishing the above, import an image into your project.<br/>
2. Click into the labelling interface. <br>
3. Check *"Auto-Annotation"* in the upper right hand corner<br>
4. (Optional, but recommended) Check *"Auto accept annotation suggestions"*<br>
5. Click the smart tool icon and make sure it is set to the keypoint option<br>
6. Choose the smart keypoint box on the bottom of the image. <br>
- If you set your labels the same as under *"Settings on the frontend"*, this should be the label with number 3 or 4
- (the first two are brush labels. These are not smart)

7. Click on the image where you want SAM to return the auto-segmentation for. <br>

> NOTE: The first time you retrieve a prediction after starting the frontend, it will take a while due to the way Label Studio works with loading models. There is a workaround in this code so that **AFTER THE FIRST RUN, THE PREDICTIONS WILL BE RECIEVED QUICKER.** On top of this, this commit allows for faster individual inference times overall, but has a slower first label so that a map of the image can be generated. If you would prefer to have overall slower individual inference times, but a faster first inference, then refer to [this commit](https://github.com/shondle/label-studio-ml-backend/tree/4367b18a52a7a494125874467c5e980a6068eca5/label_studio_ml/examples/segment_anything_model).

8. Click the generated prediction on the left side<br>
- Click the eraser on the icon tab and erase away
- Or, add to the brush prediction by choosing the one of the brush labels under the images and drawing on the object you want to label.
- *Use the eraser label to use SAM's inference to erase from an annotation by cutting off edges in the background*
- Or, do nothing if it predicted perfectly :)

9. Create more predictions by following step 6-8, then press submit!<br>
