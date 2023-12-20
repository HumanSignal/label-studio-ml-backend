https://github.com/HumanSignal/label-studio-ml-backend/assets/106922533/d1d2f233-d7c0-40ac-ba6f-368c3c01fd36



## GroundingDINO Backend Integration

Use text prompts for zero-shot detection of objects in images! Specify the detection of any object and get State of the Art results without any model fine tuning! In addition, get segmentation predictions from SAM with just text prompts!

See [here](https://github.com/IDEA-Research/GroundingDINO) for more details about the pretrained GroundingDINO model. 


## Quickstart
=======
Quickstart

1. Make sure docker is installed
2. Edit `docker-compose.yml` to include your LABEL_STUDIO_ACCESS_TOKEN found in the Label Studio software, and the LABEL_STUDIO_HOST which includes the address on which the frontend is hosted on.

Example-
- `LABEL_STUDIO_HOST=http://123.456.7.8:8080`
- `LABEL_STUDIO_ACCESS_TOKEN=c9djf998eii2948ee9hh835nferkj959923`

3. Run `docker compose up`
4. Check the IP of your backend using `docker ps` and add it to the Machine Learning backend in the Label Studio software. Usually this is on `http://localhost:9090`.

5. Create a project and edit the labelling config (an example is provided below). When editing the labeling config, make sure to add all rectangle labels under the RectangleLabels tag, and all corresponding brush labels under the BrushLabels tag.

```
<View>
  <Image name="image" value="$image"/>
  <Style>
    .lsf-main-content.lsf-requesting .prompt::before { content: ' loading...'; color: #808080; }
  </Style>
  <View className="prompt">
  <TextArea name="prompt" toName="image" editable="true" rows="2" maxSubmissions="1" showSubmitButton="true"/>
  </View>
  <RectangleLabels name="label" toName="image">
    <Label value="cats" background="yellow"/>
    <Label value="house" background="blue"/>
  </RectangleLabels>
  <BrushLabels name="label2" toName="image">
    <Label value="cats" background="yellow"/>
    <Label value="house" background="blue"/>
  </BrushLabels>
</View>
```

6. Go to an image task in your project. Turn on the Auto-annotation switch. Then, type in the prompt box and press add. After this, you should receive your predictions. See the video above for a demo. 


## Using GroundingSAM

Combine the Segment Anything Model with your text input to automatically generate mask predictions! 

To do this, set `USE_SAM=True` before running. 

If you want to use a more efficient version of SAM, set `USE_MOBILE_SAM=True` as well.


## Batching Inputs

https://github.com/HumanSignal/label-studio-ml-backend/assets/106922533/79b788e3-9147-47c0-90db-0404066ee43f

> Note: this is an experimental feature.

1. Clone the label studio branch with the added batching features.

`git clone -b feature/dino-support https://github.com/HumanSignal/label-studio.git`

2. Run this branch with `docker compose up`
3. Do steps 2-5 from the [quickstart section](#quickstart), now using access code and host IP info of the newly clones Label Studio branch. GroundingSAM is supported.
4. Go to the task menu inside your project and select the tasks you would like to annotate.
5. Click the dropdown in the upper left hand side and select `Add Text Prompt for GroundingDINO`
6. Enter in the prompt you would like to retrieve predictions for and press submit.
- If your prompt is different from the label values you have assigned, you can use the underscore to give the correct label values to your prompt outputs. For example, if I wanted to select all brown cats but still give them the label value "cats" from my labeling config, my prompt would be "brown cat_cats".


## Other Environment Variables

Adjust `BOX_THRESHOLD` and `TEXT_THRESHOLD` values in the Dockerfile to a number between 0 to 1 if experimenting. Defaults are set in `dino.py`. See explanation of these values in this [section](https://github.com/IDEA-Research/GroundingDINO#star-explanationstips-for-grounding-dino-inputs-and-outputs).

If you want to use SAM models saved from either directories, you can use the `MOBILESAM_CHECKPOINT` and `SAM_CHECKPOINT` as shown in the Dockerfile.
