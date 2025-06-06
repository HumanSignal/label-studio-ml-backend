# Time Series Segmenter for Label Studio

This example demonstrates a minimal ML backend that performs time series segmentation.
It trains a logistic regression model on labeled CSV data and predicts segments
for new tasks. The backend expects the labeling configuration to use
`<TimeSeries>` and `<TimeSeriesLabels>` tags.

## Before you begin

1. Install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart).
2. Set `LABEL_STUDIO_HOST` and `LABEL_STUDIO_API_KEY` in `docker-compose.yml`
   so the backend can download labeled tasks for training.

## Quick start

```bash
# build and run
docker-compose up --build
```

Connect the model from the **Model** page in your project settings. The default
URL is `http://localhost:9090`.

## Labeling configuration

Use a configuration similar to the following:

```xml
<View>
  <TimeSeriesLabels name="label" toName="ts">
    <Label value="Run"/>
    <Label value="Walk"/>
  </TimeSeriesLabels>
  <TimeSeries name="ts" valueType="url" value="$csv_url" timeColumn="time">
    <Channel column="sensorone" />
    <Channel column="sensortwo" />
  </TimeSeries>
</View>
```

The backend reads the time column and channels to build feature vectors for
training and prediction. Each CSV referenced by `csv_url` is expected to contain
at least the time column and the listed channels.

## Training

Training starts automatically when annotations are created or updated. The model
collects all labeled segments, extracts sensor values inside each segment and
fits a logistic regression classifier. Model artifacts are stored in the
`MODEL_DIR` (defaults to the current directory).

## Prediction

For each task, the backend loads the CSV, applies the trained classifier to each
row and groups consecutive predictions into labeled segments. Prediction scores
are averaged per segment and returned to Label Studio.
