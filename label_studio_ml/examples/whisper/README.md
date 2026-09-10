<!--
---
title: Automatic Speech Recognition with Whisper
type: guide
tier: all
order: 61
hide_menu: true
hide_frontmatter_title: true
meta_title: Automatic Speech Recognition with Whisper
meta_description: Tutorial on how to transcribe audio in Label Studio with faster-whisper, either as one transcript or as timestamped segments
categories:
    - Audio/Speech Processing
    - Automatic Speech Recognition
    - Whisper
image: "/guide/ml_tutorials/whisper.png"
---
-->

# ASR with Whisper

This example transcribes audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a CTranslate2 reimplementation of [OpenAI Whisper](https://github.com/openai/whisper) that runs on CPU as well as GPU.

Whisper returns timestamps along with the text, so this backend can fill in either of the two audio transcription workflows in Label Studio: one transcript for the whole file, or one region per spoken segment with its own transcript. It picks the shape from your labeling config, so there is nothing to switch on.

## Before you begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart).

This tutorial uses the [`whisper` example](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/whisper).

## Labeling interface

For a single transcript, use the **Automatic Speech Recognition** template (**Audio/Speech Processing > Automatic Speech Recognition**):

```xml
<View>
  <Audio name="audio" value="$audio" zoom="true" hotkey="ctrl+enter" />
  <Header value="Provide Transcription" />
  <TextArea name="transcription" toName="audio"
            rows="4" editable="true" maxSubmissions="1" />
</View>
```

For timestamped segments, use **Automatic Speech Recognition using Segments**. Adding a `<Labels>` tag and marking the `<TextArea>` as `perRegion` is what tells this backend to return one region per Whisper segment:

```xml
<View>
  <Labels name="labels" toName="audio">
    <Label value="Speech" />
    <Label value="Noise" />
  </Labels>

  <Audio name="audio" value="$audio"/>

  <TextArea name="transcription" toName="audio"
            rows="2" editable="true"
            perRegion="true" required="true" />
</View>
```

Every segment is returned as a region carrying `WHISPER_SEGMENT_LABEL` (`Speech` by default) plus a transcript attached to that same region. If the label is not in your config, the first label is used.

> Warning: If you use files hosted in Label Studio (meaning they were added using the import action), hosted in cloud storage, or connected through local storage, then you must provide the `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY` environment variables to the ML backend. For more information, see [Allow the ML backend to access Label Studio data](https://labelstud.io/guide/ml#Allow-the-ML-backend-to-access-Label-Studio-data). For information about finding your Label Studio API key, see [Access token](https://labelstud.io/guide/user_account#Access-token).

## Running with Docker (recommended)

1. Start the Machine Learning backend on `http://localhost:9090` with the prebuilt image:

```bash
docker-compose up
```

2. Validate that backend is running:

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Create a project in Label Studio. Then from the **Model** page in the project settings, [connect the model](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio). The default URL is `http://localhost:9090`.

## Building from source (advanced)

To build the ML backend from source, you have to clone the repository and build the Docker image:

```bash
docker-compose build
```

## Running without Docker (advanced)

To run the ML backend without Docker, you have to clone the repository and install all dependencies using pip:

```bash
python -m venv ml-backend
source ml-backend/bin/activate
pip install -r requirements.txt
```

Then you can start the ML backend:

```bash
label-studio-ml start ./whisper
```

## Configuration

Parameters can be set in `docker-compose.yml` before running the container.

The following parameters are available:
- `WHISPER_MODEL` - Model size or a path to a converted model, for example `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`. (`base` by default)
- `WHISPER_DEVICE` - `cpu`, `cuda` or `auto`. (`cpu` by default)
- `WHISPER_COMPUTE_TYPE` - Quantization used by CTranslate2, for example `int8` on CPU or `float16` on GPU. (`int8` by default)
- `WHISPER_LANGUAGE` - Two letter language code such as `en`. Leave it empty to let Whisper detect the language per file.
- `WHISPER_BEAM_SIZE` - Beam size used during decoding. (`5` by default)
- `WHISPER_SEGMENT_LABEL` - Label applied to every segment when the config has a `<Labels>` tag. (`Speech` by default)
- `BASIC_AUTH_USER` - Specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - Specify the basic auth password for the model server
- `LOG_LEVEL` - Set the log level for the model server
- `WORKERS` - Specify the number of workers for the model server
- `THREADS` - Specify the number of threads for the model server
- `LABEL_STUDIO_URL`: The host of the Label Studio instance. Default is `http://localhost:8080`.
- `LABEL_STUDIO_API_KEY`: The API key for the Label Studio instance.

The weights are downloaded from Hugging Face on the first request and cached afterwards, so the first prediction on a fresh container is slower than the ones after it.

## Customization

The ML backend can be customized by adding your own models and logic inside `./whisper/model.py`.
