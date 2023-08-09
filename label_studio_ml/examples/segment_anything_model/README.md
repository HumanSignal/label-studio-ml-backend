# Interactive Annotation in Label Studio with Segment Anything Model



https://github.com/shondle/label-studio-ml-backend/assets/106922533/42a8a535-167c-404a-96bd-c2e2382df99a



Use Facebook's Segment Anything Model with Label Studio!

# Intro

There are two models in this repo that you can use.
- **1. Advanced Segment Anything Model**
- **2. ONNX Segment Anything Model**

The **Advanced Segment Anything Model** introduces the ability to combine a multitude of different prompts to achieve a prediction, and the ability to use MobileSAM.
- Mix one rectangle label with multiple positive keypoints to refine your predictions! Use negative keypoints to take away area from predictions for increased control.
- Use MobileSAM, an extremely lightweight alternative to the heavy Segment Anything Model from Facebook, to retrieve predictions. This can run inference within a second using a laptop GPU!

The **ONNX Segment Anything Model** gives the ability to use either a single keypoint or single rectangle label to prompt the original SAM.
- This offers a much faster prediction using the original Segment Anything Model due to using the ONNX version.
- Downside: image size must be specified before using the ONNX model, and cannot be generalized to other image sizes while labeling. Also, does not yet offer the mixed labeling and refinement that AdvancedSAM does.

# Your Choices

Here are the pros and cons broken down for choosing each model, and the choices you have for each.

**Using AdvancedSAM**
1. *Use with MobileSAM architecture*
-  Pros: very lightweight can be run on laptops, mix many different combinations of input prompts to fine-tune prediction
-  Cons: Lower accuracy than Facebook's original SAM architecture
2. _Use with original SAM architecture_
- Pros: higher accuracy than MobileSAM, mix many different combinations of input prompts to fine-tune prediction
- Cons: takes long to gather predictions (~2s to create embedding of an image), requires access to good GPUs

**Using ONNXSAM**
1. _Use with regular SAM architecture_
- Pros: much faster than when you use it in Advanced SAM
- Cons: can only use one smart label per prediction, image size must be defined before generating the ONNX model, cannot label images with different sizes without running into issues

# Setup

## Setting Up the Backend

### 1. Clone This Repo

### 2. Download Model Weights

- For using MobileSAM-
Install the weights using [this link](https://cdn.githubraw.com/ChaoningZhang/MobileSAM/01ea8d0f/weights/mobile_sam.pt) and place in folder (along with the advanced_sam.py and onnx_sam.py files)

- For using regular SAM and/or ONNX-
Follow [SAM installation instructions with pip](https://github.com/facebookresearch/segment-anything). 
Then, install the [ViT-H SAM model](https://github.com/facebookresearch/segment-anything)

  - For the ONNX model- `python onnxconverter.py`

### 3. Install Requirements
Change your directory into this folder and then install all requirements.

```
pip install -r requirements.txt
```

As an aside, make sure you have [Label Studio installed](https://labelstud.io/guide/install.html#Install-with-Anaconda)


### 4. Adjust variables and _wsgi.py depending on your choice of model.

**Choosing whether to use Advanced or ONNX model**
- To use AdvancedSAM model, set RUN_ONNX_SAM environment variable to False (this is the default in the code, you only have to adjust the environment variable if it is set to something previously)
- To use ONNX model, set RUN_ONNX_SAM environment variable to True

**To choose between MobileSAM and regular SAM architectures when using AdvancedSAM**
- To use MobileSAM: set SAM_CHOICE environment variable to "MobileSAM" (this is the default in the code, you only have to adjust the environment variable if it is set to something previously)
- To use regular SAM: set SAM_CHOICE environment variable to "SAM"

### 5. Start the Backend and Run Label Studio
```
# change into this project folder from where you are
cd segment_anything_model
python _wsgi.py

# in a new terminal
label-studio start
```

## Settings on the frontend

[The video](#creating-the-annotation) also goes over this process, but does part of it while in the newly created project menu.

1. Create a project and go to settings.
2. Under "Machine Learning" click "Add Model"<br>
3. Under "URL" paste the URL of where the model backend is running (you can find this in the terminal where you started the backend)<br>
4. Switch on "Use for interactive preannotations"<br>
5. Click "Validate and Save"<br>

6. Next -> go to "labeling Interface". This is on the same side where you chose the "Machine Learning" tab.<br>
7. Choose the code option and choose [your template](#labeling-configs)

# Creating the Annotation

See [this video tutorial](https://drive.google.com/file/d/1OMV1qLHc0yYRachPPb8et7dUBjxUsmR1/view?usp=sharing) to get a better understanding of the workflow when annotating with SAM.

## Notes for AdvancedSAM:
- _**Please watch the [video](#creating-the-annotation) first**_

For the best experience, follow the video tutorial above and _**uncheck 'Auto accept annotation suggestions'**_ when running predictions.

After generating the prediction from an assortment of inputs, make sure you _**click the check mark that is outside of the image**_ to finalize the region (this should either be above or below the image. Watch the [video](#creating-the-annotation) for a visual guide).
- There may be a check mark inside the image next to a generated prediction, but _do not use that one_. For some reason, the check mark that is not on the image itself makes sure to clean the other input prompts used for generating the region, and only leaves the predicted region after being clicked (this is the most compatible way to use the backend.
- You may run into problems creating instances of the same class if you click the check mark on the image and it leaves the labels used to guide the region).

After labeling your object, select the label in the menu and select the type of brush label you would like to give it at the top of your label keys below your image.
- This allows you to change the class of your prediction.
- See the [video](#creating-the-annotation) for a better explanation.

_**Only the negative keypoints can be used for subtracting from prediction areas**_ for the model. Positive keypoints and rectangles tell the model areas of interest to make positive predictions. 

Multiple keypoints may be used to provide areas for the model where predictions should be extended. _**Only one rectangle label may be used**_ when generating a prediction as an area where the model prediction should occur/be extended.
- If you place multiple rectangle labels, the model will use the newest rectangle label along with all other keypoints when aiding the model prediction. 

## Notes for ONNX:
The ONNX model uses the 'orig_img_size' in `onnx_converter.py` that defines an image ratio for the ONNX model. Change this to the ratio of the images that you are labeling before generating the model. If you are labeling images of different sizes, use Advanced SAM instead, or generate a new ONNX model for different image groups with different sizes. If you do not adjust `orig_img_size`, and your image aspect ratios do not match what is already defined, then your predictions will be offset from the image.
- Make sure you adjust `orig_img_size` BEFORE generating the ONNX model when using `onnx_converter.py`
- Guide on changing the code - `"orig_im_size": torch.tensor([#heightofimages, #widthofimages], dtype=torch.float),`

## Notes for Exporting:
- COCO and YOLO format is not supported (this project exports using brush labels, so try NumPy or PNG export instead)

# Labeling Configs

## When using the AdvancedSAM-
- Give one brush label per class you want to annotate.
- Give each brush an alias. 
- For each class, create two keypoints. The first keypoint is for gaining predictions from the model where a keypoint is placed. The second can be referred to as a 'negative keypoint' telling the model to avoid predictions in the area where it is placed.
  - You MUST give each keypoint an alias. The first alias will be the index (starting from one) of your class labels. The alias for the second keypoint should have the same index as the first keypoint, but should be negative (as this is the 'negative keypoint'). It is very important you get this correct, as this is how the ML backend differentiates between types of keypoints.
  - Add one rectangle label for each of your classes that you want to annotate
- [The video](#creating-the-annotation) reviews these points as well if you are confused after reading this

Base example:
```
<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="brush" toName="image">
  	<Label value="1" hint="brush" background="blue" alias="1_brush"/>
  	<Label value="2" hint="brush" background="purple" alias="2_brush"/>
  </BrushLabels>
  <KeypointLabels name="tag2" toName="image">
  	<Label value="(+1)" hint="keypoint" background="blue" alias="1"/>
  	<Label value="(-1)" hint="keypoint" background="red" alias="-1"/>
    <Label value="(+2)" hint="keypoint" background="purple" alias="2"/>
  	<Label value="(-2)" hint="keypoint" background="red" alias="-2"/>
  </KeypointLabels>
  <RectangleLabels name="tag4" toName="image">
  	<Label value="[1]" hint="rectangle" background="blue" alias="1"/>
  	<Label value="[2]" hint="rectangle" background="purple" alias="2"/>
  </RectangleLabels>       
</View>
```

Notice how there are two keypoints for each brush. The first keypoint per brush is a "positive keypoint" or tells the model where to select an area, and the second keypoint is a "negative keypoint" or tells the model where to avoid the predictions. 

The values for each of the labels do not matter as much and are there just to help you distinguish labels. It is recommended to make negative keypoints red in order to tell the difference. However, aliases for keypoints and rectangle label MUST be correct. 

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

## Credits

Original Segment Anything Model paper-
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

MobileSAM paper-
```
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```


