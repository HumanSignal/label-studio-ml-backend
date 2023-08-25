# Interactive Annotation in Label Studio with Segment Anything Model

https://github.com/shondle/label-studio-ml-backend/assets/106922533/42a8a535-167c-404a-96bd-c2e2382df99a

Use Facebook's Segment Anything Model with Label Studio!

# Quick Start

## Using Docker Compose (recommended)

To start the server with lightweight mobile version of SAM, run the following command:

```bash
docker-compose up
```

# Intro

There are three models in this repo that you can use.
1. Advanced Segment Anything Model
2. ONNX Segment Anything Model

# About the Models

## Advanced Segment Anything Model

The Advanced Segment Anything Model introduces the ability to combine a
multitude of different prompts to achieve a prediction, and the ability to use
MobileSAM.

- Mix one rectangle label with multiple positive keypoints to refine your
  predictions! Use negative keypoints to take away area from predictions for
  increased control.
- Use MobileSAM, an extremely lightweight alternative to the heavy Segment
  Anything Model from Facebook, to retrieve predictions. This can run inference
  within a second using a laptop GPU!

## ONNX Segment Anything Model

The ONNX Segment Anything Model gives the ability to use either a single
keypoint or single rectangle label to prompt the original SAM.
- This offers a much faster prediction using the original Segment Anything
  Model due to using the ONNX version.
- Downside: image size must be specified before using the ONNX model, and
  cannot be generalized to other image sizes while labeling. Also, does not yet
  offer the mixed labeling and refinement that AdvancedSAM does.

## Model Configuration Options

Here are the pros and cons broken down for choosing each model, and the choices you have for each.

* AdvancedSAM
  * Mobile SAM Configuration
    - Pros: Lightweight model that can be run on laptops, and can mix many
      different combinations of input prompts to fine-tune prediction.
    - Cons: Lower accuracy than Facebook's original SAM architecture.
  * Original SAM architecture
    - Pros: Higher accuracy than MobileSAM, with ability to mix many different
      combinations of input prompts to fine-tune predictions.
    - Cons: takes long to gather predictions (~2s to create embedding of an
      image), requires access to good GPUs

* Using ONNXSAM
  * Original SAM Architecture
    - Pros: Much faster than when you use it in Advanced SAM
    - Cons: Can only use one smart label per prediction. Image size must be
      defined before generating the ONNX model. Cannot label images with
      different sizes without running into issues

# Setup

The Label Studio SAM Backend works best if you have [Local
Storage](https://labelstud.io/guide/storage.html#Local-storage) enabled for
your project. It is also possible to set up shared local storage, but is not
recommended. Currently, the backend does not work with cloud storage (S3,
Azure, GCP).

## Setting Up the Label Studio Server

### Enabling Local Storage File Serving

You can enable local storage file serving by setting the following variables.

```
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<path_to_image_data>
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
```

For example, if you're launching Label Studio with Docker, you can enable these variables with
```
docker run -it -p 8080:8080 \
               -v $(pwd)/mydata:/label-studio/data \
               --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
               --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data/images \
               heartexlabs/label-studio:latest
```

Note the IP address that you are running your Label Studio instance as the
`LABEL_STUDIO_HOST`. This will be nessary for setting up the connection to your
SAM model. _Because you are hosting both Label Studio and the ML Backend in
Docker containers, the hostname `localhost` will not resolve to the correct
address._ There are a number of ways to determine your host ip address.  These
can include calling either `ip a` or `ifconfig` from the command line and
inspecting the output, or finding the address that has been assigned to your
computer through the system network configuration settings.

### Obtain Your API Token

Log into the Label Studio interface (in the example above, at
`http://<LABEL_STUDIO_HOST>:8080`). Create a new user, go into your `Account &
Settings` tab, and make a note of the Access Token, which we will use later as
the `LABEL_STUDIO_ACCESS_TOKEN`.

## Setting Up the SAM Backend

### Clone the Repository

Make a clone of this repository on your host system and move into the working
directory.

```
git clone https://github.com/humansignal/label-studio-ml-backend
cd label-studio-ml-backend/label_studio_ml/examples/segment_anything_model
```

### Using Docker Compose (recommended)

We suggest using [Docker Compose](https://docs.docker.com/compose/) to host and
run the backend. For GPU support, please consult the [Docker Compose GPU Access
Guide](https://docs.docker.com/compose/gpu-support/) to understand how to pass
through GPU resources to services.

Edit the `docker-compose.yml` file and fill in the values for the
`LABEL_STUDIO_HOST` and `LABEL_STUDIO_ACCESS_TOKEN` for your particular
installation. Be sure to append the port that Label Studio is running on in
your `LABEL_STUDIO_HOST` variable, for example `http://192.168.1.36:8080` if
Label Studio is running on port 8080.

Run the command `docker compose up --build` to build the container and run it
locally.

### Setting up the Backend Manually

#### Download Model Weights

This step is only necessary if you are not using the Docker build for this model.

- For MobileSAM install the weights using [this
  link](https://cdn.githubraw.com/ChaoningZhang/MobileSAM/01ea8d0f/weights/mobile_sam.pt)
  and place in folder (along with the advanced_sam.py and onnx_sam.py files)

- For using regular SAM and/or ONNX- Follow [SAM installation instructions with
  pip](https://github.com/facebookresearch/segment-anything). Then, install
  the [ViT-H SAM model](https://github.com/facebookresearch/segment-anything)

- For the ONNX model install using `python onnxconverter.py`

You can download all weights and models using the following command:

```bash
./download_weights.sh
```

#### Install Requirements
Change your directory into this folder and then install all if the python requirements.

```
pip install -r requirements.txt
```

#### Adjust variables and `_wsgi.py` depending on your choice of model.

You can set the following environment variables to change the behavior of the model.

* `LABEL_STUDIO_HOST` sets the endpoint of the Label Studio host.
* `LABEL_STUDIO_ACCESS_TOKEN` sets the API access token for the Label Studio host.
* `SAM_CHOICE` selects which model to use.
    * `SAM_CHOICE=MobileSAM` to use MobileSAM (default)
    * `SAM_CHOICE=SAM` to use the original SAM model.
    * `SAM_CHOICE=ONNX` to use the ONNX model.

#### Start the Backend

You can now manually start the ML backend.

```
python _wsgi.py
```

or

```bash
docker-compose up
```
to start the backend in a docker container.

## Set up a Project in Label Studio for Segment Anything

Log into your Label Studio instance and perform the following steps.

1. Create a new project.
2. Under the *Labeling Setup* in the *Create Project* interface, or under the
   *Labeling Interface* menu item in the project settings, paste the [sample
   template](#labeling-configs) into the code dialog. Save the interface.
3. Under the *Machine Learning* menu item in the project settings, select *Add Model*.
4. Enter a title for the model, and the URL for the instance of the model you
   just created. If you're running Label Studio in Docker or on another host, you
   should use the direct IP address of where the model is hosted (`localhost`)
   will not work. Be sure to include the port number that the model is hosted on
   (the default is `9090`). For example, if the model is hosted on `192.168.1.36`,
   the URL for the model would be `http://192.168.1.36:9090`
5. Click *Validate and Save*.

You can now upload images into your project and begin annotating.

[The video](#creating-the-annotation) also goes over this process, but does part of it while in the newly created project menu.


# Creating Annotations

See [this video tutorial](https://drive.google.com/file/d/1OMV1qLHc0yYRachPPb8et7dUBjxUsmR1/view?usp=sharing)
to get a better understanding of the workflow when annotating with SAM.

Use `Alt` hotkey to alter keypoint positive and negative labels.

## Notes for AdvancedSAM

* _**Please watch the [video](#creating-the-annotation) first**_
* For the best experience, follow the video tutorial above and _**uncheck 'Auto
  accept annotation suggestions'**_ when running predictions.
* After generating the prediction from an assortment of inputs, make sure you _**click the
  check mark that is outside of the image**_ to finalize the region (this should either
  be above or below the image. Watch the [video](#creating-the-annotation) for a visual guide).
* There may be a check mark inside the image next to a generated prediction,
  but _do not use that one_. For some reason, the check mark that is not on the
  image itself makes sure to clean the other input prompts used for generating
  the region, and only leaves the predicted region after being clicked (this is
  the most compatible way to use the backend.
* You may run into problems creating instances of the same class if you click
  the check mark on the image and it leaves the labels used to guide the
  region).
* After labeling your object, select the label in the menu and select the type
  of brush label you would like to give it at the top of your label keys below
  your image. This allows you to change the class of your prediction. See the
  [video](#creating-the-annotation) for a better explanation.
* _**Only the negative keypoints can be used for subtracting from prediction
  areas**_ for the model. Positive keypoints and rectangles tell the model
  areas of interest to make positive predictions. 
* Multiple keypoints may be used to provide areas for the model where predictions
  should be extended. _**Only one rectangle label may be used**_ when generating
  a prediction as an area where the model prediction should occur/be extended.
  If you place multiple rectangle labels, the model will use the newest
  rectangle label along with all other keypoints when aiding the model
  prediction. 

## Notes for ONNX

* The ONNX model uses the `orig_img_size` in `onnx_converter.py` that defines
  an image ratio for the ONNX model. Change this to the ratio of the images
  that you are labeling before generating the model. If you are labeling images
  of different sizes, use Advanced SAM instead, or generate a new ONNX model
  for different image groups with different sizes. If you do not adjust
  `orig_img_size`, and your image aspect ratios do not match what is
  already defined, then your predictions will be offset from the image. If
  using `docker compose` to launch the model, be sure to rebuild the host
  container.
* Make sure you adjust `orig_img_size` BEFORE generating the ONNX model when
  using `onnx_converter.py`
* Guide on changing the code - `"orig_im_size": torch.tensor([#heightofimages, #widthofimages], dtype=torch.float),`

## Notes for Exporting

* COCO and YOLO format is not supported (this project exports using brush
  labels, so try NumPy or PNG export instead)

# Labeling Configs

## When using the AdvancedSAM

- Give one brush label per class you want to annotate.
- Hold `Alt` hotkey to create negative keypoints.
- Add one rectangle label for each of your classes that you want to annotate
- [The video](#creating-the-annotation) reviews these points as well if you are
  confused after reading this

Base example:
```
<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="tag" toName="image">
  	<Label value="Banana" background="#FF0000"/>
  	<Label value="Orange" background="#0d14d3"/>
  </BrushLabels>
  <KeyPointLabels name="tag2" toName="image" smart="true">
    <Label value="Banana" smart="true" background="#000000" showInline="true"/>
    <Label value="Orange" smart="true" background="#000000" showInline="true"/>
  </KeyPointLabels>
  <RectangleLabels name="tag3" toName="image" smart="true">
    <Label value="Banana" background="#000000" showInline="true"/>
    <Label value="Orange" background="#000000" showInline="true"/>
  </RectangleLabels>
</View>
```

## When Using the ONNX model

Label values for the keypoints, rectangle, and brush labels must correspond.
Other than that, make sure that smart="True" for each keypoint label and
rectangle label. 

For the ONNX model-
```
<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="tag" toName="image">
  	<Label value="Banana" background="#FF0000"/>
  	<Label value="Orange" background="#0d14d3"/>
  </BrushLabels>
  <KeyPointLabels name="tag2" toName="image" smart="true">
    <Label value="Banana" smart="true" background="#000000" showInline="true"/>
    <Label value="Orange" smart="true" background="#000000" showInline="true"/>
  </KeyPointLabels>
  <RectangleLabels name="tag3" toName="image" smart="true">
    <Label value="Banana" background="#000000" showInline="true"/>
    <Label value="Orange" background="#000000" showInline="true"/>
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
