
# BERT-based Text Classification

The NewModel is a BERT-based text classification model that is designed to work with Label Studio. This model uses the Hugging Face Transformers library to fine-tune a BERT model for text classification. The model is trained on the labeled data from Label Studio and then used to make predictions on new data.  With this model connected to Label Studio, you can:  
Train a BERT model on your labeled data directly from Label Studio.

Use any model for [AutoModelForSequenceClassification](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelforsequenceclassification) from the Hugging Face model hub.

- Fine-tune the model on your specific task and use it to make predictions on new data.
- Automatically download the labeled tasks from Label Studio and prepare the data for training.
- Customize the training parameters such as learning rate, number of epochs, and weight decay.


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

> Warning! Please note the current limitation of the ML backend: models are loaded dynamically from huggingface.co. You may need `HF_TOKEN` env variable provided in your environment. Consequently, this may result in a slow response time for the first prediction request. If you are experiencing timeouts on Label Studio side (i.e., no predictions are visible when opening the task), please check the logs of the ML backend for any errors, and refresh the page in a few minutes.

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
- `BASIC_AUTH_USER` - specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - specify the basic auth password for the model server
- `LOG_LEVEL` - set the log level for the model server
- `WORKERS` - specify the number of workers for the model server
- `THREADS` - specify the number of threads for the model server
- `BASELINE_MODEL_NAME`: The name of the baseline model to use for training. Default is `bert-base-multilingual-cased`.

## Training

The following parameters are available for training:

- `LABEL_STUDIO_HOST` (required): The URL of the Label Studio instance. Default is http://localhost:8080.
- `LABEL_STUDIO_API_KEY` (required): The API key for the Label Studio instance.
- `START_TRAINING_EACH_N_UPDATES`: The number of labeled tasks to download from Label Studio before starting training. Default is 10.
- `LEARNING_RATE`: The learning rate for the model training. Default is 2e-5.
- `NUM_TRAIN_EPOCHS`: The number of epochs for model training. Default is 3.
- `WEIGHT_DECAY`: The weight decay for the model training. Default is 0.01.
- `FINETUNED_MODEL_NAME`: The name of the fine-tuned model. Default is finetuned-model.


# Customization

The ML backend can be customized by adding your own models and logic inside the `./bert_classifier` directory.
