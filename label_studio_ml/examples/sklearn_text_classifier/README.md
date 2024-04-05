# Sklearn Text Classifier Model for Label Studio

The Sklearn Text Classifier Model is a custom machine learning backend for Label Studio. It uses a [Logistic Regression model from the Scikit-learn](https://scikit-learn.org/) library to classify text data. This model is particularly useful for text classification tasks in Label Studio, providing an efficient way to generate pre-annotations based on the model's predictions.

The model is trained on the labeled texts collected from Label Studio, and it uses the Label Studio API to fetch the labeled tasks for training. This integration with Label Studio allows for a seamless and efficient labeling workflow, as the model can be retrained and updated as new labeled data becomes available.

## Labeling Configuration

The Sklearn Text Classifier Model is designed to work with the default labeling configuration for text classification in Label Studio. This configuration includes a single output of `<Choices>` type and a single input of `<Text>` type. The model retrieves the first occurrence of these tags from the labeling configuration and uses them for prediction:

```xml
<View>
  <Text name="text" value="$text" />
  <Choices name="label" toName="text" choice="single" showInLine="true">
    <Choice value="positive" />
    <Choice value="negative" />
  </Choices>
</View>
```

> Please note that you must specify the `LABEL_STUDIO_HOST` and `LABEL_STUDIO_API_KEY` environment variables in order to download examples for training the model. These variables should point to your Label Studio instance and its API key, respectively.


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
- `LOGISTIC_REGRESSION_C`: This is the inverse regularization strength for Logistic Regression. It is a float value and can be set via environment variable "LOGISTIC_REGRESSION_C". If not set, it defaults to 10.
- `LABEL_STUDIO_HOST`: This is the host URL for Label Studio, used for training. It can be set via environment variable "LABEL_STUDIO_HOST". If not set, it defaults to 'http://localhost:8080'.
- `LABEL_STUDIO_API_KEY`: This is the API key for Label Studio, used for training. It can be set via environment variable "LABEL_STUDIO_API_KEY". There is no default value for this, so it must be set.
- `START_TRAINING_EACH_N_UPDATES`: This is the number of updates after which training starts. It is an integer value and can be set via environment variable "START_TRAINING_EACH_N_UPDATES". If not set, it defaults to 10.
- `BASIC_AUTH_USER` - specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - specify the basic auth password for the model server
- `LOG_LEVEL` - set the log level for the model server
- `WORKERS` - specify the number of workers for the model server
- `THREADS` - specify the number of threads for the model server

# Customization

The ML backend can be customized by adding your own models and logic inside the `./dir_with_your_model` directory. 