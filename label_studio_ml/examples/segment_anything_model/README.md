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
- Downside: image size must be specified before using the ONNX model, and cannot be generalized to other image sizes while labeling. Also, does not yet offer the mixed labeling and refinement that AdvancedSAM does.

# Your Choices
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
`python onnxconverter.py`

### 3. Install Requirements
Change your directory into this folder and then install all requirements.

- [Label Studio Installation Instructions](https://labelstud.io/guide/install.html#Install-with-Anaconda)

```
pip install requirements.txt
```

### 4. Adjust variables and _wsgi.py for your model.

**Choosing whether to use Advanced or ONNX model**
- To use AdvancedSAM model, set RUN_ONNX_SAM environment variable to False (this is the default in the code)
- To use ONNX model, set RUN_ONNX_SAM environment variable to True

**To choose between MobileSAM and regular SAM when using AdvancedSAM**
- To use MobileSAM: set SAM_CHOICE environment variable to "MobileSAM" (this is the default in the code)
- To use regular SAM: set SAM_CHOICE environment variable to "SAM"

### 5. Start the Backend and Run Label Studio
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

6. Next -> go to "labeling Interface". This is on the same side where you chose the "Machine Learning" tab.<br>
7. Choose the code option and choose [your template](#labeling-configs)

# Creating the Annotation

See the following video tutorial for annotating -> 

## Notes for AdvancedSAM:
- Please watch the video first
- For the best experience, and follow the video tutorial above and uncheck 'Auto accept annotation suggestions' when running predictions.
- After generating the prediction, if you want to create another instance of the same class, you must delete all keypoints and boxes used (as shown in the video) and _refresh the page_. If you are only labeling one instance of each class then you do not have to worry about this.
  - This is because, at this moment, AdvancedSAM uses the alias number of the object you are labeling from previous tasks to inform future predictions. However, if you want to create a class and not use previous keypoints and rectangles to derive predictions of a previous class, you must delete the previous keypoints and rectangles if labeling a separate instance in the image with the same class.
- After labeling your object, select the label in the menu and select the type of brush label you would like to give it at the top of your label keys below your image. This allows you to change the class of your prediction. See the video for a better explanation.
- Only the negative keypoints can be used for subtracting from prediction areas for the model.
- Multiple keypoints may be used to provide areas for the model where predictions should be extended. Only one rectangle label may be used when generating a prediction as anarea where the model prediction should occur/be extended. If you place multiple rectangle labels, the model will use the newest label when aiding the model prediction. 

## Notes for ONNX:
The ONNX model uses the 'orig_img_size' in `onnx_converter.py` that defines an image ratio for the ONNX model. Change this to the ratio of the images that you are labeling before generating the model. If you are labeling images of different sizes, use Advanced SAM instead, or generate a new ONNX model for different image groups with different sizes. If you do not adjust `orig_img_size`, and your image aspect ratios do not match what is already defined, then your predictions will be offset from the image.
- Make sure you adjust `orig_img_size` BEFORE generating the ONNX model when using `onnx_converter.py`
- Guide on changing the code - `"orig_im_size": torch.tensor([#heightofimages, #widthofimages], dtype=torch.float),`

## Notes for Exporting:
- COCO and YOLO format is not supported (this project exports using brush labels, so try numpy or PNG export instead)

1. After finishing the above, import an image into your project.<br/>
2. Click into the labeling interface. <br>
3. Check *"Auto-Annotation"* in the upper right hand corner<br>
4. (Optional, but recommended) Check *"Auto accept annotation suggestions"*<br>
5. Click the smart tool icon and make sure it is set to the keypoint option<br>
6. Choose the smart keypoint box on the bottom of the image. <br>
- If you set your labels the same as under *"Settings on the frontend"*, this should be the label with number 3 or 4
- (the first two are brush labels. These are not smart)

7. Click on the image where you want SAM to return the auto-segmentation for. <br>

> NOTE: The first time you retrieve a prediction after starting the frontend, it will take a while due to the way Label Studio works with loading models. There is a workaround in this code so that **AFTER THE FIRST RUN, THE PREDICTIONS WILL BE RECIEVED QUICKER.** On top of this, this commit allows for faster individual inference times overall, but has a slower first label so that a map of the image can be generated. If you would prefer to have overall slower individual inference times, but a faster first inference, then refer to [this commit](https://github.com/shondle/label-studio-ml-backend/tree/4367b18a52a7a494125874467c5e980a6068eca5/label_studio_ml/examples/segment_anything_model).

9. Create more predictions by following step 6-8, then press submit!<br>

# Labeling Configs
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
