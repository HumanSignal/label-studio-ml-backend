https://github.com/HumanSignal/label-studio-ml-backend/assets/106922533/470c6cfc-bb2e-4fe3-9627-30a7e3bc28d5


## GroundingDINO Backend Integration

Use text prompts for zero-shot detection of objects in images! Specify the detection of any object and get State of the Art results without any model fine tuning!

See [here](https://github.com/IDEA-Research/GroundingDINO) for more details about the pretrained GroundingDINO model. 


Quickstart

1. Make sure docker is installed
2. Edit `docker-compose.yml` to include your LABEL_STUDIO_ACCESS_TOKEN found in teh Label Studio software, and the LABEL_STUDIO_HOST which includes the address on which the frontend is hosted on.

Example-
- `LABEL_STUDIO_HOST=http://123.456.7.8:8080`
- `LABEL_STUDIO_ACCESS_TOKEN=c9djf998eii2948ee9hh835nferkj959923`

3. Run `docker compose up`
4. Check the IP of your backend using `docker ps` and add it to the Machine Learning backend in the Label Studio software. Usually this is on `http://localhost:9090`.

5. Edit the labelling config to the below.

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
  </RectangleLabels>
</View>
```

This may be adjusted to your needs, but please keep the promp section and some rectangle labels.

6. Go to an image task in one of your projects. Turn on the Auto-annotation switch. Then, type in the prompt box and press add. After this, you should receive your predictions. See the video above for a demo. 