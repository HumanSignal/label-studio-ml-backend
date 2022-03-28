## Quick usage

Build and start Machine Learning backend on `http://localhost:9090`

```bash
docker-compose up -d
```

## What is the Label Studio ML backend?

The Label Studio ML backend is an SDK that lets you wrap your machine learning code and turn it into a web server.
You can then connect that server to a Label Studio instance to perform 2 tasks:

- Dynamically pre-annotate data based on model inference results
- Retrain or fine-tune a model based on recently annotated data

If you just need to load static pre-annotated data into Label Studio, running an ML backend might be overkill for you. Instead, you can [import preannotated data](https://labelstud.io/guide/predictions.html).

## How it works

1. Get your model code
2. Wrap it with the Label Studio SDK
3. Create a running server script
4. Launch the script
5. Connect Label Studio to ML backend on the UI


## Quickstart

Follow this example tutorial to run an ML backend with a simple text classifier:

0. Clone the repo
   ```bash
   git clone https://github.com/heartexlabs/label-studio-ml-backend  
   ```
   
1. Setup environment
    
    It is highly recommended to use `venv`, `virtualenv` or `conda` python environments. You can use the same environment as Label Studio does. [Read more](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) about creating virtual environments via `venv`.
   ```bash
   cd label-studio-ml-backend
   
   # Install label-studio-ml and its dependencies
   pip install -U -e .
   
   # Install example dependencies
   pip install -r label_studio_ml/examples/dummy_model/requirements.txt
   ```
   
2. Initialize an ML backend based on an example script:
   ```bash
   label-studio-ml init my_ml_backend --script label_studio_ml/examples/dummy_model/dummy_model.py
   ```
   This ML backend is an example provided by Label Studio. See [how to create your own ML backend](#Create_your_own_ML_backend).

3. Start ML backend server
   ```bash
   label-studio-ml start my_ml_backend
   ```
   
4. Start Label Studio and connect it to the running ML backend on the project settings page.

## Create your own ML backend

Follow this tutorial to wrap existing machine learning model code with the Label Studio ML SDK to use it as an ML backend with Label Studio. 

Before you start, determine the following:
1. The expected inputs and outputs for your model. In other words, the type of labeling that your model supports in Label Studio, which informs the [Label Studio labeling config](https://labelstud.io/guide/setup.html#Set-up-the-labeling-interface-for-your-project). For example, text classification labels of "Dog", "Cat", or "Opossum" could be possible inputs and outputs. 
2. The [prediction format](https://labelstud.io/guide/predictions.html) returned by your ML backend server.

This example tutorial outlines how to wrap your model with the Label Studio ML SDK.

Start by creating a class declaration. You can create a Label Studio-compatible ML backend server in one command by inheriting it from `LabelStudioMLBase`. 
```python
from label_studio_ml.model import LabelStudioMLBase

class MyModel(LabelStudioMLBase):
```

Then, define loaders & initializers in the `__init__` method. 

```python
def __init__(self, **kwargs):
    # don't forget to initialize base class...
    super(MyModel, self).__init__(**kwargs)
    self.model = self.load_my_model()
```

There are special variables provided by the inherited class:
- `self.parsed_label_config` is a Python dict that provides a Label Studio project config structure. See [ref for details](). Use might want to use this to align your model input/output with Label Studio labeling configuration;
- `self.label_config` is a raw labeling config string;
- `self.train_output` is a Python dict with the results of the previous model training runs (the output of the `fit()` method described bellow) Use this if you want to load the model for the next updates for active learning and model fine-tuning.

After you define the loaders, you can define two methods for your model: an inference call and a training call. 

### Inference call

Use an inference call to get pre-annotations from your model on-the-fly. You must update the existing predict method in the example ML backend scripts to make them work for your specific use case. Write your own code to override the `predict(tasks, **kwargs)` method, which takes [JSON-formatted Label Studio tasks](https://labelstud.io/guide/tasks.html#Basic-Label-Studio-JSON-format) and returns predictions in the [format accepted by Label Studio](https://labelstud.io/guide/predictions.html).

**Example**

```python
def predict(self, tasks, **kwargs):
    predictions = []
    # Get annotation tag first, and extract from_name/to_name keys from the labeling config to make predictions
    from_name, schema = list(self.parsed_label_config.items())[0]
    to_name = schema['to_name'][0]
    for task in tasks:
        # for each task, return classification results in the form of "choices" pre-annotations
        predictions.append({
            'result': [{
                'from_name': from_name,
                'to_name': to_name,
                'type': 'choices',
                'value': {'choices': ['My Label']}
            }],
            # optionally you can include prediction scores that you can use to sort the tasks and do active learning
            'score': 0.987
        })
    return predictions
```


### Training call
Use the training call to update your model with new annotations. You don't need to use this call in your code, for example if you just want to pre-annotate tasks without retraining the model. If you do want to retrain the model based on annotations from Label Studio, use this method. 

Write your own code to override the `fit(annotations, **kwargs)` method, which takes [JSON-formatted Label Studio annotations](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-labeled-tasks) and returns an arbitrary dict where some information about the created model can be stored.

**Example**
```python
def fit(self, completions, workdir=None, **kwargs):
    # ... do some heavy computations, get your model and store checkpoints and resources
    return {'checkpoints': 'my/model/checkpoints'}  # <-- you can retrieve this dict as self.train_output in the subsequent calls
```


After you wrap your model code with the class, define the loaders, and define the methods, you're ready to run your model as an ML backend with Label Studio. 

For other examples of ML backends, refer to the [examples in this repository](label_studio_ml/examples). These examples aren't production-ready, but can help you set up your own code as a Label Studio ML backend.