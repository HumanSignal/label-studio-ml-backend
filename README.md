# What is the Label Studio ML backend?

The Label Studio ML backend is an SDK that lets you wrap your machine learning code and turn it into a web server.
The web server can be connected to a running [Label Studio](https://labelstud.io/) instance to automate labeling tasks.

If you just need to load static pre-annotated data into Label Studio, running an ML backend might be overkill for you.
Instead, you can [import preannotated data](https://labelstud.io/guide/predictions.html).

# Quickstart

In order to start using the models, use [docker-compose](https://docs.docker.com/compose/install/) to run the ML backend
server.

Use the following command to start serving the ML backend at `http://localhost:9090`:

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend/label_studio_ml/examples/{MODEL_NAME}
docker-compose up
```

Replace `{MODEL_NAME}` with the name of the model you want to use:

## Models

The following models are supported in the repository. Some of them work without any additional setup, and some of them
require additional parameters to be set.

Check the **Required parameters** column to see if you need to set any additional parameters.

- **Pre-annotation** column indicates if the model can be used for pre-annotation in Label Studio:  
  you can see pre-annotated data when opening the labeling page or after running predictions for a batch of data.
- **Interactive mode** column indicates if the model can be used for interactive labeling in Label Studio: see
  interactive predictions when performing actions on labeling page.
- **Training** column indicates if the model can be used for training in Label Studio: update the model state based the
  submitted annotations.

| MODEL_NAME                                                                                 | Description                                                                                                                               | Pre-annotation | Interactive mode | Training | Required parameters                           |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|-----------------|------------------|----------|-----------------------------------------------|
| [segment_anything_model](/label_studio_ml/examples/segment_anything_model)                 | Image segmentation by [Meta](https://segment-anything.com/)                                                                               | ❌               | ✅                |   ❌       | None                                          |
| [llm_interactive](/label_studio_ml/examples/llm_interactive)                               | Prompt engineering with [OpenAI](https://platform.openai.com/), Azure LLMs.                                                               | ✅               | ✅                | ✅        | OPENAI_API_KEY                                |
| [grounding_dino](/label_studio_ml/examples/grounding_dino)                                 | Object detection with prompts. [Details](https://github.com/IDEA-Research/GroundingDINO)                                                  | ❌               | ✅                | ❌        | None                                          |
| [tesseract](/label_studio_ml/examples/tesseract)                                           | Interactive OCR. [Details](https://github.com/tesseract-ocr/tesseract)                                                                    | ❌               | ✅                | ❌        | None                                          |
| [easyocr](/label_studio_ml/examples/easyocr)                                               | Automated OCR. [EasyOCR](https://github.com/JaidedAI/EasyOCR)                                                                             | ✅               | ❌                | ❌        | None                                          |
| [spacy](/label_studio_ml/examples/spacy)                                                   | NER by [SpaCy](https://spacy.io/)                                                                                                         | ✅               | ❌                | ❌        | None                                          |
| [flair](/label_studio_ml/examples/flair)                                                   | NER by [flair](https://flairnlp.github.io/)                                                                                               | ✅               | ❌                | ❌        | None                                          |
| [bert_classifier](/label_studio_ml/examples/bert_classifier)                               | Text classification with [Huggingface](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelforsequenceclassification) | ✅               | ❌                | ✅        | None                                          |
| [huggingface_llm](/label_studio_ml/examples/huggingface_llm)                               | LLM inference with [Hugging Face](https://huggingface.co/tasks/text-generation)                                                           | ✅               | ❌                | ❌        | None                                          |
| [huggingface_ner](/label_studio_ml/examples/huggingface_ner)                               | NER by [Hugging Face](https://huggingface.co/docs/transformers/en/tasks/token_classification)                                             | ✅               | ❌                | ✅        | None                                          |
| [nemo_asr](/label_studio_ml/examples/nemo_asr)                                             | Speech ASR by [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)                                                                               | ✅               | ❌                | ❌        | None                                          |
| [mmdetection](/label_studio_ml/examples/mmdetection-3)                                     | Object Detection with [OpenMMLab](https://github.com/open-mmlab/mmdetection)                                                              | ✅               | ❌                | ❌        | None                                          |
| [sklearn_text_classifier](/label_studio_ml/examples/sklearn_text_classifier)               | Text classification with [scikit-learn](https://scikit-learn.org/stable/)                                                                 | ✅               | ❌                | ✅        | None                                          |
| [interactive_substring_matching](/label_studio_ml/examples/interactive_substring_matching) | Simple keywords search                                                                                                                    | ❌               | ✅                | ❌        | None                                          |
| [langchain_search_agent](/label_studio_ml/examples/langchain_search_agent)                 | RAG pipeline with Google Search and [Langchain](https://langchain.com/)                                                                   | ✅               | ✅                | ✅        | OPENAI_API_KEY, GOOGLE_CSE_ID, GOOGLE_API_KEY |

# (Advanced usage) Develop your model

To start developing your own ML backend, follow the instructions below.

### 1. Installation

Download and install `label-studio-ml` from the repository:

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend/
pip install -e .
```

### 2. Create empty ML backend:

```bash
label-studio-ml create my_ml_backend
```

You can go to the `my_ml_backend` directory and modify the code to implement your own inference logic.
The directory structure should look like this:

```
my_ml_backend/
├── Dockerfile
├── docker-compose.yml
├── model.py
├── _wsgi.py
├── README.md
└── requirements.txt
```

`Dockefile` and `docker-compose.yml` are used to run the ML backend with Docker.
`model.py` is the main file where you can implement your own training and inference logic.
`_wsgi.py` is a helper file that is used to run the ML backend with Docker (you don't need to modify it)
`README.md` is a readme file with instructions on how to run the ML backend.
`requirements.txt` is a file with Python dependencies.

### 3. Implement prediction logic

In your model directory, locate the `model.py` file (for example, `my_ml_backend/model.py`).

The `model.py` file contains a class declaration inherited from `LabelStudioMLBase`. This class provides wrappers for
the API methods that are used by Label Studio to communicate with the ML backend. You can override the methods to
implement your own logic:

```python
def predict(self, tasks, context, **kwargs):
    """Make predictions for the tasks."""
    return predictions
```

The `predict` method is used to make predictions for the tasks. It uses the following:

- `tasks`: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
- `context`: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend) - for
  interactive labeling scenario
- `predictions`: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)

Once you implement the `predict` method, you can see predictions from the connected ML backend in Label Studio.

### 4. Implement training logic (optional)

You can also implement the `fit` method to train your model. The `fit` method is typically used to train the model on
the labeled data, although it can be used for any arbitrary operations that require data persistence (for example,
storing labeled data in database, saving model weights, keeping LLM prompts history, etc).
By default, the `fit` method is called at any data action in Label Studio, like creating a new task or updating
annotations. You can modify this behavior in Label Studio > Settings > Webhooks.

To implement the `fit` method, you need to override the `fit` method in your `model.py` file:

```python
def fit(self, event, data, **kwargs):
    """Train the model on the labeled data."""
    old_model = self.get('old_model')
    # write your logic to update the model
    self.set('new_model', new_model)
```

with

- `event`: event type can be `'ANNOTATION_CREATED'`, `'ANNOTATION_UPDATED', etc.
- `data` the payload received from the event (check more
  on [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))

Additionally, there are two helper methods that you can use to store and retrieve data from the ML backend:

- `self.set(key, value)` - store data in the ML backend
- `self.get(key)` - retrieve data from the ML backend

Both methods can be used elsewhere in the ML backend code, for example, in the `predict` method to get the new model
weights.

#### Other methods and parameters

Other methods and parameters are available within the `LabelStudioMLBase` class:

- `self.label_config` - returns the [Label Studio labeling config](https://labelstud.io/guide/setup.html) as XML string.
- `self.parsed_label_config` - returns the [Label Studio labeling config](https://labelstud.io/guide/setup.html) as
  JSON.
- `self.model_version` - returns the current model version.
- `self.get_local_path(url, task_id)` - this helper function is used to download and cache an url that is typically stored in `task['data']`, 
and to return the local path to it. The URL can be: LS uploaded file, LS Local Storage, LS Cloud Storage or any other http(s) URL.      

#### Run without Docker

To run without docker (for example, for debugging purposes), you can use the following command:

```bash
label-studio-ml start my_ml_backend
```

#### Test your ML backend

Modify the `my_ml_backend/test_api.py` to ensure that your ML backend works as expected.

#### Modify the port

To modify the port, use the `-p` parameter:

```bash
label-studio-ml start my_ml_backend -p 9091
```

## Deploy your ML backend to GCP

Before you start:

1. Install [gcloud](https://cloud.google.com/sdk/docs/install)
2. Init billing for account if it's not [activated](https://console.cloud.google.com/project/_/billing/enable)
3. Init gcloud, type the following commands and login in browser:

```bash
gcloud auth login
```

4. Activate your Cloud Build API
5. Find your GCP project ID
6. (Optional) Add GCP_REGION with your default region to your ENV variables

To start deployment:

1. Create your own ML backend
2. Start deployment to GCP:

```bash
label-studio-ml deploy gcp {ml-backend-local-dir} \
--from={model-python-script} \
--gcp-project-id {gcp-project-id} \
--label-studio-host {https://app.heartex.com} \
--label-studio-api-key {YOUR-LABEL-STUDIO-API-KEY}
```

3. After label studio deploys the model - you will get model endpoint in console.

# Troubleshooting

## Troubleshooting Docker Build on Windows

If you encounter an error similar to the following when running `docker-compose up --build` on Windows:

```
exec /app/start.sh : No such file or directory
exited with code 1
```

This issue is likely caused by Windows' handling of line endings in text files, which can affect scripts
like `start.sh`. To resolve this issue, follow the steps below:

### Step 1: Adjust Git Configuration

Before cloning the repository, ensure your Git is configured to not automatically convert line endings to
Windows-style (CRLF) when checking out files. This can be achieved by setting `core.autocrlf` to `false`. Open Git Bash
or your preferred terminal and execute the following command:

```
git config --global core.autocrlf false
```

### Step 2: Clone the Repository Again

If you have already cloned the repository before adjusting your Git configuration, you'll need to clone it again to
ensure that the line endings are preserved correctly:

1. **Delete the existing local repository.** Ensure you have backed up any changes or work in progress.
2. **Clone the repository again.** Use the standard Git clone command to clone the repository to your local machine.

### Step 3: Build and Run the Docker Containers

Navigate to the appropriate directory within your cloned repository that contains the Dockerfile
and `docker-compose.yml`. Then, proceed with the Docker commands:

1. **Build the Docker containers:** Run `docker-compose build` to build the Docker containers based on the configuration
   specified in `docker-compose.yml`.

2. **Start the Docker containers:** Once the build process is complete, start the containers using `docker-compose up`.

### Additional Notes

- This solution specifically addresses issues encountered on Windows due to the automatic conversion of line endings. If
  you're using another operating system, this solution may not apply.
- Remember to check your project's `.gitattributes` file, if it exists, as it can also influence how Git handles line
  endings in your files.

By following these steps, you should be able to resolve issues related to Docker not recognizing the `start.sh` script
on Windows due to line ending conversions.


## Troubleshooting Pip Cache Reset in Docker Images

Sometimes, you want to reset the pip cache to ensure that the latest versions of the dependencies are installed. 
For example, Label Studio ML Backend library is used as 
`label-studio-ml @ git+https://github.com/HumanSignal/label-studio-ml-backend.git` in requirements.txt. Let's assume that it
is updated, and you want to jump on the latest version in your docker image with the ML model. 

You can rebuild a docker image from scratch with the following command:

```bash
docker compose build --no-cache
```
