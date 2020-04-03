# Label Studio ML Backend &middot; :sunny:

[Website](https://labelstud.io/) • [Docs](https://labelstud.io/guide) • [Twitter](https://twitter.com/heartexlabs) • [Join Slack Community <img src="https://go.heartex.net/docs/images/slack-mini.png" width="18px"/>](https://join.slack.com/t/label-studio/shared_invite/zt-cr8b7ygm-6L45z7biEBw4HXa5A2b5pw)

<br/>

 **Label Studio is a swiss army knife of data labeling and annotation tools :v:**
 
 <br />

## Introduction

ML Backend can be used to provide pre-labeling, automatic labeling, and active learning support. 

## Quickstart

Build and start Machine Learning backend on `http://localhost:9090`

```bash
docker-compose up
```

Check if it works:

```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

Then connect running backend to Label Studio:

```bash
label-studio start --init new_project --ml-backend-url http://localhost:9090 --template image_classification
```


## Writing your own model
1. Place your scripts for model training & inference inside root directory. Follow the [API guidelines](#api-guidelines) described bellow. You can put everything in a single file, or create 2 separate one say `my_training_module.py` and `my_inference_module.py`

2. Write down your python dependencies in `requirements.txt`

3. Open `wsgi.py` and make your configurations under `init_model_server` arguments:
    ```python
    from my_training_module import training_script
    from my_inference_module import InferenceModel
   
    init_model_server(
        create_model_func=InferenceModel,
        train_script=training_script,
        ...
    ```

4. Make sure you have docker & docker-compose installed on your system, then run
    ```bash
    docker-compose up --build
    ```
   
## API guidelines


#### Inference module
In order to create module for inference, you have to declare the following class:

```python
from htx.base_model import BaseModel

# use BaseModel inheritance provided by pyheartex SDK 
class MyModel(BaseModel):
    
    # Describe input types (Label Studio object tags names)
    INPUT_TYPES = ('Image',)

    # Describe output types (Label Studio control tags names)
    INPUT_TYPES = ('Choices',)

    def load(self, resources, **kwargs):
        """Here you load the model into the memory. resources is a dict returned by training script"""
        self.model_path = resources["model_path"]
        self.labels = resources["labels"]

    def predict(self, tasks, **kwargs):
        """Here you create list of model results with Label Studio's prediction format, task by task"""
        predictions = []
        for task in tasks:
            # do inference...
            predictions.append(task_prediction)
        return predictions
```

#### Training module
Training could be made in a separate environment. The only one convention is that data iterator and working directory are specified as input arguments for training function which outputs JSON-serializable resources consumed later by `load()` function in inference module.

```python
def train(input_iterator, working_dir, **kwargs):
    """Here you gather input examples and output labels and train your model"""
    resources = {"model_path": "some/model/path", "labels": ["aaa", "bbb", "ccc"]}
    return resources
```

## Ecosystem

| Project | Description |
|-|-|
| [label-studio](https://github.com/heartexlabs/label-studio) | Server part, distributed as a pip package |
| [label-studio-frontend](https://github.com/heartexlabs/label-studio-frontend) | Frontend part, written in JavaScript and React, can be embedded into your application | 
| [label-studio-converter](https://github.com/heartexlabs/label-studio-converter) | Encode labels into the format of your favorite machine learning library | 
| [label-studio-transformers](https://github.com/heartexlabs/label-studio-transformers) | Transformers library connected and configured for use with label studio | 

## License

This software is licensed under the [Apache 2.0 LICENSE](/LICENSE) © [Heartex](https://www.heartex.ai/). 2020

<img src="https://github.com/heartexlabs/label-studio/blob/master/images/opossum_looking.png?raw=true" title="Hey everyone!" height="140" width="140" />
