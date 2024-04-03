<!--
---
title: Hugging Face Large Language Model Backend
type: blog
tier: all
order: 30
meta_title: Label Studio tutorial to run Hugging Face Large Language Model Backend
meta_description: This tutorial explains how to run Hugging Face Large Language Model Backend in Label Studio. Hugging Face Large Language Model Backend is a machine learning backend designed to work with Label Studio, providing a custom model for text generation.
categories:
    - huggingface
    - llm
    - text-generation
image: "/tutorials/huggingface_llm.png"
---
-->

# Hugging Face Large Language Model Backend

This machine learning backend is designed to work with Label Studio, providing a custom model for text generation. The model is based on the Hugging Face's transformers library and uses a pre-trained.
Check [text generation pipelines on Hugging Face](https://huggingface.co/tasks/text-generation) for more details.

## Label Studio XML Labeling Config

This ML backend is compatible with a Label Studio labeling configuration that uses a `<TextArea>` tag. Here is an example of a compatible labeling configuration:

```xml
<View>
    <Text name="input_text" value="$text"/>
  <TextArea name="generated_text"  toName="input_text"/>
</View>
```

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

3. Connect to the backend from Label Studio running on the same host: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL.


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
label-studio-ml start ./dir_with_your_model
```

# Configuration
Parameters can be set in `docker-compose.yml` before running the container.

The following common parameters are available:
- `MODEL_NAME`: The name of the pre-trained model to use for text generation. Default is `facebook/opt-125m`.
- `MAX_LENGTH`: The maximum length of the generated text. Default is `50`.
- `BASIC_AUTH_USER`: The basic auth user for the model server.
- `BASIC_AUTH_PASS`: The basic auth password for the model server.
- `LOG_LEVEL`: The log level for the model server.
- `WORKERS`: The number of workers for the model server.
- `THREADS`: The number of threads for the model server.

# Customization

The ML backend can be customized by adding your own models and logic inside the `./huggingface_llm` directory. 