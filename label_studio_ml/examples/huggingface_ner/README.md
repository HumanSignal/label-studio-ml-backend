# Huggingface NER Model with Label Studio

This project uses a custom machine learning backend model for Named Entity Recognition (NER) with Huggingface's transformers and Label Studio.
The model instantiate `AutoModelForTokenClassification` from Huggingface's transformers library and fine-tunes it on the NER task.

If you want to use this model only in inference mode, it serves predictions from the pre-trained model. 
If you want to fine-tune the model, you can use the Label Studio interface to provide training data and train the model.

Read more about the compatible models from [Huggingface official documentation](https://huggingface.co/docs/transformers/en/tasks/token_classification)

> Note: if you plan to train the model, you have to provide the baseline pretrained model that can be finetuned (i.e. where the last layer can be trained, for example, `distilbert/distilbert-base-uncased`). Otherwise you may see the error about tensor sizes mismatch during training.


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
label-studio-ml start ./hubbingface_ner
```

# Configuration
Parameters can be set in `docker-compose.yml` before running the container.


The following common parameters are available:
- `BASIC_AUTH_USER` - specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - specify the basic auth password for the model server
- `LOG_LEVEL` - set the log level for the model server
- `WORKERS` - specify the number of workers for the model server
- `THREADS` - specify the number of threads for the model server
- `BASELINE_MODEL_NAME`: The name of the baseline model to use. Default is 'dslim/bert-base-NER'.
- `FINTUNED_MODEL_NAME`: The name of the fine-tuned model. Default is 'finetuned_model'.
- `LABEL_STUDIO_HOST`: The host of the Label Studio instance. Default is 'http://localhost:8080'.
- `LABEL_STUDIO_API_KEY`: The API key for the Label Studio instance.
- `START_TRAINING_EACH_N_UPDATES`: The number of updates after which to start training. Default is 10.
- `LEARNING_RATE`: The learning rate for the model. Default is 1e-3.
- `NUM_TRAIN_EPOCHS`: The number of training epochs. Default is 10.
- `WEIGHT_DECAY`: The weight decay for the model. Default is 0.01.
- `MODEL_DIR`: The directory where the model is stored. Default is './results'.

# Customization

The ML backend can be customized by adding your own models and logic inside the `./huggingface_ner/model.py`.
Modify the `predict()` and `fit()` methods to implement your own logic.