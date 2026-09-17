# Using HQ-SAM2 with Label Studio for Image Annotation

HQ-SAM2, upgrade SAM2 for high quality zero-shot segmentation.

## Before you begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart). 

This tutorial uses the [`hq_segment_anything_2_image` example](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/hq_segment_anything_2_image). 


## Labeling configuration

The current implementation of the Label Studio HQ-SAM2 ML backend works using Interactive mode. The user-guided inputs are:
- `KeypointLabels`
- `RectangleLabels`

And then HQ-SAM2 outputs `BrushLabels` as a result.

This means all three control tags should be represented in your labeling configuration:

```xml
<View>
<Style>
  .main {
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
    margin: 0;
    padding: 20px;
  }
  .container {
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
  }
  .column {
    flex: 1;
    padding: 10px;
    background-color: #fff;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    text-align: center;
  }
  .column .title {
    margin: 0;
    color: #333;
  }
  .column .label {
    margin-top: 10px;
    padding: 10px;
    background-color: #f9f9f9;
    border-radius: 3px;
  }
  .image-container {
    width: 100%;
    height: 300px;
    background-color: #ddd;
    border-radius: 5px;
  }
</Style>
<View className="main">
  <View className="container">
    <View className="column">
      <View className="title">Choose Label</View>
      <View className="label">
        <BrushLabels name="tag" toName="image">
          
          
        <Label value="defect" background="#FFA39E"/></BrushLabels>
      </View>
    </View>
    <View className="column">
      <View className="title">Use Keypoint</View>
      <View className="label">
        <KeyPointLabels name="tag2" toName="image" smart="true">
          
          
        <Label value="defect" background="#250dd3"/></KeyPointLabels>
      </View>
    </View>
    <View className="column">
      <View className="title">Use Rectangle</View>
      <View className="label">
        <RectangleLabels name="tag3" toName="image" smart="true">
          
          
        <Label value="defect" background="#FFC069"/></RectangleLabels>
      </View>
    </View>
  </View>
  <View className="image-container">
    <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  </View>
</View>
</View>
```

## Running with Docker

1. Start Machine Learning backend on `http://localhost:9090` with prebuilt image:

```bash
docker-compose up
```

2. Validate that backend is running

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Connect to the backend from Label Studio running on the same host: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL.


## Configuration
Parameters can be set in `docker-compose.yml` before running the container.


The following common parameters are available:
- `DEVICE` - specify the device for the model server (currently only `cuda` is supported, `cpu` is coming soon)
- `MODEL_CONFIG` - HQ-SAM2 model configuration file (`sam2.1_hq_hiera_l.yaml` by default)
- `MODEL_CHECKPOINT` - HQ-SAM2 model checkpoint file (`sam2.1_hiera_large.pt` by default)
- `BASIC_AUTH_USER` - specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - specify the basic auth password for the model server
- `LOG_LEVEL` - set the log level for the model server
- `WORKERS` - specify the number of workers for the model server
- `THREADS` - specify the number of threads for the model server 
