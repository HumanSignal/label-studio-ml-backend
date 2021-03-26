## What is the Label Studio ML backend?

The Label Studio ML backend is an SDK that lets you wrap you machine learning code and turn it into a web server.
You can then connect that server to a Label Studio instance to perform 2 tasks:

- Dynamically pre-annotate data based on model inference results
- Retrain or fine-tune a model based on recently annotated data

If you need to just load static pre-annotated data into Label Studio, running an ML backend  might be overkill for you. Instead, you can [import preannotated data](https://labelstud.io/guide/predictions.html).

## How it works

<diagram>

1. Get your model code
2. Wrap it with the Label Studio SDK
3. Create a running server script
4. Launch script
5. Connect Label Studio to ML backend on the UI


## Quickstart

Here is a quick example tutorial on how to run the ML backend with a simple text classifier:

0. Clone the repo
   ```bash
   git clone https://github.com/heartexlabs/label-studio-ml-backend  
   ```
   
1. Set up environment
   ```bash
   cd label-studio-ml-backend
   pip install -e .
   cd label_studio_ml/examples
   pip install -r requirements.txt
   ```
   
2. Create an ML backend
   ```bash
   label-studio-ml init my_ml_backend --script label_studio_ml/examples/simple_text_classifier.py
   ```
   
3. Start ML backend server
   ```bash
   label-studio-ml start my_ml_backend
   ```
   
4. Start Label Studio and connect it to the running ML backend on the project settings page.

## Create your own ML backend

Follow this tutorial to wrap your existing machine learning model code with the Label Studio ML SDK to use it as an ML backend with Label Studio. 

Before you start, determine the following:
1. What types of labeling does your model support? In other words, what are the expected inputs and outputs for your model? This informs the [Label Studio labeling config]() that your model supports.
2. The [prediction format](https://labelstud.io/guide/predictions.html) returned by your ML backend server.

This example tutorial outlines how to wrap a simple text classifier based on the [scikit-learn]() framework with the Label Studio ML SDK.

First, create a class declaration. You can create a Label Studio-compatible ML backend server in one command by inheriting it from `LabelStudioMLBase`. 
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

After you define the loaders, define two methods for your model:

`predict(tasks, **kwargs)`, which takes [JSON-formatted Label Studio tasks](doc-link) and returns predictions in [format accepted by Label Studio](doc-link).
`fit(annotations, **kwargs)`, which takes [JSON-formatted Label Studio annotations](doc-link) and returns an arbitrary dict where some information about the created model can be stored.

After you wrap your model code with this class, define the loaders, and define the methods, you're ready to run your model as an ML backend with Label Studio. 

For other examples of ML backends, refer to the [examples in this repository](label_studio_ml/examples). These examples aren't production-ready, but can help you set up your own code as a Label Studio ML backend.