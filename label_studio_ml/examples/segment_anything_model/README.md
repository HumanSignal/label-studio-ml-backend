# Interactive Annotation in Label Studio with Segment Anything Model

<img src="https://user-images.githubusercontent.com/106922533/232959476-7fc74bbb-24c8-46f3-a1c1-d16f9efcec5f.gif" width="500" />

Use Facebook's Segment Anything Model with Label Studio!

# Setup

## Setting Up the Backend

### 1. Clone this repo

Place the images you want to annotate in this project's folder.

### 2. Retrieve Label Studio Code

```
git clone https://github.com/heartexlabs/label-studio-ml-backend
cd label-studio-ml-backend

# Install label-studio-ml and its dependencies
pip install -U -e .

# Install the dependencies for the example or your custom ML backend
pip install -r path/to/my_ml_backend/requirements.txt
```

### 3. Download SAM

Follow [SAM installation instructions with pip](https://github.com/facebookresearch/segment-anything). 
Then, install the [ViT-H SAM model](https://github.com/facebookresearch/segment-anything) *into this project's directory*

### 4. Add to your bashrc
```
nano ~/.bashrc
# add the bottom of your bashrc
export ML_TIMEOUT_SETUP=120
```

### 5. Installations
```
pip install label-studio numpy python-opencv
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
  </KeyPointLabels>
</View>
```
To change for your use case - 
- Label values must be the same for KeyPointLabels and BrushLabels
- smartOnly must only be set to the label values for the Keypoints


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

> NOTE: The first time you retrieve a prediction after starting the frontend, it will take a while due to the way Label Studio works with loading models. There is a workaround in this code so that **AFTER THE FIRST RUN, THE PREDICTIONS WILL BE RECIEVED QUICKER.** 

8. Click the generated prediction on the left side<br>
- Click the eraser on the icon tab and erase away
- Or, add to the brush prediction by choosing the one of the brush labels under the images and drawing on the object you want to label.
- Or, do nothing if it predicted perfectly :)

8. Create more predictions by following step 6-8, then press submit!<br>