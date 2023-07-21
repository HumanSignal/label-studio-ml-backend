# Interactive Annotation in Label Studio with Segment Anything Model

<img src="https://user-images.githubusercontent.com/106922533/234322629-e583c838-11eb-4261-aaa1-872f1695720c.gif" width="500" />

<img src="https://user-images.githubusercontent.com/106922533/234322576-a24643f8-aeb6-421c-984e-d0d2e2233cd4.gif" width="500" />

Use Facebook's Segment Anything Model with Label Studio!

# Intro

There are two models in this repo that you can use.
- **1. Advanced Segment Anything Model**
- **2. ONNX Segment Anything Model**

The **Advanced Segment Anything Model** introduces the ability to combined a multitude of different prompts to achieve a prediction, and the ability to use MobileSAM.
- Mix one rectangle label with multiple positive keypoints to refine your predictions! Use negative keypoints to take away area from predictions for increased control.
- Use MobileSAM, and extremely lightweight alternative to the heavy original SegmentAnythingModel, to retrieve predictions. This can run inference within less than a second solely using a laptop without external compute!

The **ONNX Segment Anything Model** gives the ability to use either a single keypoint or single rectangle label to prompt the original SAM.
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

### 2. Download Model Weights

For using MobileSAM-
Install the weights using [this link](https://cdn.githubraw.com/ChaoningZhang/MobileSAM/01ea8d0f/weights/mobile_sam.pt) and place in folder (along with the advanced_sam.py and onnx_sam.py files)

For using regular SAM and/or ONNX-
Follow [SAM installation instructions with pip](https://github.com/facebookresearch/segment-anything). 
Then, install the [ViT-H SAM model](https://github.com/facebookresearch/segment-anything)

For the ONNX model-
Then use the SAM installation instructions from above to convert to ONNX and place *into this project's directory*

### 5. Install Requirements
Change your directory into this folder and then install all requirements.

- [Label Studio Installation Instructions](https://labelstud.io/guide/install.html#Install-with-Anaconda)

```
pip install requirements.txt
```

### 6. Start the Backend and Run Label Studio
```
# change into this project folder from where you are
cd segment_anything_model
python _wsgi.py -p 4243
label-studio start
```

## Settings on the frontend

1. Create a project and go to settings.
2. Under "Machine Learning" click "Add Model"<br>
3. Under "URL" paste the URL of where the model backend is running (you can find this in the terminal where you started the backend)<br>
4. Switch on "Use for interactive preannotations"<br>
5. Click "Validate and Save"<br>

6. Next -> go to "Labelling Interface". This is on the same side where you chose the "Machine Learning" tab.<br>
7. Choose the code option and choose your template-

# Creating the Annotation

See the following video tutorial for annotating -> 

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

9. Create more predictions by following step 6-8, then press submit!<br>

# Labelling Configs
Default configs are provided below.

## When using the AdvancedSAM-
- Give one brush label per class you want to annotate.
- Give each brush and alias. 
- For each class, create two keypoints. The first keypoint is for gaining predictions from the model where a keypoint is placed. The second can be referred to as a 'negative keypoint' telling the model to avoid predictions in the area where it is placed.
  - You MUST give each keypoint an alias. The first alias will be the index (starting from one) of your class labels. The alias for the second keypoint should have the same index of the first keypoint, but should be negative (as this is the 'negative keypoint'). It is very important you get this correct, as this is how the ML backend differentiates between types of keypoints.
  - Add one rectangle label for each of your classes that you want to annotate

Base example:
```
<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="brush" toName="image">
  	<Label value="Brush_1" background="blue" alias="1_brush"/>
  	<Label value="Brush_2" background="purple" alias="2_brush"/>
  </BrushLabels>
  <KeypointLabels name="tag2" toName="image">
  	<Label value="1_+Keypoint" background="blue" alias="1"/>
  	<Label value="1-Keypoint" background="red" alias="-1"/>
    <Label value="2+Keypoint" background="purple" alias="2"/>
  	<Label value="2-Keypoint" background="red" alias="-2"/>
  </KeypointLabels>
  <RectangleLabels name="tag4" toName="image">
  	<Label value="1_Rectangle" background="blue" alias="1"/>
  	<Label value="2_Rectangle" background="purple" alias="2"/>
  </RectangleLabels>
</View>
```

Notice how there are two keypoints for each brush. The first keypoint per brush is a "postive keypoint" or tells the model where to select an area, and the second keypoint is a "negative keypoint" or tells the model where to avoid the predictions. 

The values for each of the labels does not matter as much, and are just there to help you distinguish labels. It is recommended to make negative keypoints red in order to tell the difference. However, aliases for keypoints and rectangle label MUST be correct. 

## When Using the ONNX model

Label values for the keypoints, rectangle, and brush labels must correspond. Other than that, make sure that smart="True" for each keypoint label and rectangle label. 

For the ONNX model-
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
  </KeyPointLabels>
  <RectangleLabels name="tag3" toName="image">
    <Label value="Banana" smart="true" background="#000000" showInline="true"/>
    <Label value="Orange" smart= "true" background="#000000" showInline="true"/>
  </RectangleLabels>
</View>
```
