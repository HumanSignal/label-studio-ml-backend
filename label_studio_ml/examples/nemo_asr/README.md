# ASR with NeMo

This example demonstrates how to use the [NeMo](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/README.md) to perform ASR (Automatic Speech Recognition) in Label Studio.
Use this ML backend if you want to transcribe and fix your audio data.

## Labeling Interface

It works with the `Audio Transcription` labeling interface from project `Settings -> Labeling interfaces -> Browse Templates -> Audio Processing -> Audio Transcription`:

```xml
<View>
  <Audio name="audio" value="$audio" zoom="true" hotkey="ctrl+enter" />
  <Header value="Provide Transcription" />
  <TextArea name="transcription" toName="audio"
            rows="4" editable="true" maxSubmissions="1" />
</View>
```

or any other labeling interface that combines `<Audio>` and `<TextArea>` elements.

> Warning: if you use files hosted in Label Studio (e.g. audio files directly uploaded via import dialog), you should provide `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY` environment variable to the ML backend.

## Running with Docker (Recommended)

1. Start Machine Learning backend on `http://localhost:9090` with prebuilt image:

```bash
docker-compose up
```

2. Validate that backend is running

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Connect to the backend from Label Studio running on the same host: go to your project `Settings -> Model -> Connect Model` and specify `http://localhost:9090` as a URL.


## Building from source (Advanced)

To build the ML backend from source, you have to clone the repository and build the Docker image:

```bash
docker-compose build
```

## Running without Docker (Advanced)

To run the ML backend without Docker, you have to clone the repository and install all dependencies using pip:

```bash
python -m venv ml-backend
source ml-backend/bin/activate
pip install -r requirements.txt
```

Then you can start the ML backend:

```bash
label-studio-ml start ./nemo_asr
```

# Configuration
Parameters can be set in `docker-compose.yml` before running the container.


The following common parameters are available:
- `MODEL_NAME` - specify the model name for the ASR. (`QuartzNet15x5Base-En` by default)
- `BASIC_AUTH_USER` - specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - specify the basic auth password for the model server
- `LOG_LEVEL` - set the log level for the model server
- `WORKERS` - specify the number of workers for the model server
- `THREADS` - specify the number of threads for the model server
- `LABEL_STUDIO_HOST`: The host of the Label Studio instance. Default is 'http://localhost:8080'.
- `LABEL_STUDIO_API_KEY`: The API key for the Label Studio instance.

# Customization

The ML backend can be customized by adding your own models and logic inside the `./nemo_asr/model.py`. 