## What is Label Studio ML backend

Label Studio ML backend is SDK that allows you to wrap you machine learning code and turn it into the web server.
This server could be connected to Label Studio instance to perform 2 tasks:

- Dynamically preannotating the data based on model inference results
- Retrain / finetune the model based on recently annotated data.

If you need to just load static preannotated data into Label Studio - may be running ML backend is overkill for you, you may try [importing preannotated data](doc-link)

## How it works

<diagram>

1. Get your code
2. Wrap it into Label Studio SDK
3. Create running server script
4. Launch script
5. Connect Label Studio to ML backend via UI


## Quickstart

Here is a quick example tutorial on how to run the ML backend with a simple text classifier:

0. Clone repo
   ```bash
   git clone https://github.com/heartexlabs/label-studio-ml-backend  
   ```
   
1. Setup environment
   ```bash
   cd label-studio-ml-backend
   pip install -e .
   cd label_studio_ml/examples
   pip install -r requirements.txt
   ```
   
2. Create new ML backend
   ```bash
   label-studio-ml init my_ml_backend --script label_studio_ml/examples/simple_text_classifier.py
   ```
   
3. Start ML backend server
   ```bash
   label-studio-ml start my_ml_backend
   ```
   
4. Run Label Studio connecting it to the running ML backend from the project settings page

## Create your own ML backend

Let's follow step-by-step tutorial how to create your own ML backend.
There are 2 things you should bear in mind:

1. What is the [Label Studio labeling config](doc-link) your model can work with (in other words, what is the inputs and the outputs of your model)
2. [The prediction format](doc-link) your server should return

Let's start showing up how to wrap a simple text classifier based on [scikit-learn]() framework.

First we create a class declaration. By inheriting it from `LabelStudioMLBase`, it allows you to create Label Studio-compatible ML backend server in one command.

```python
from label_studio_ml.model import LabelStudioMLBase

class MyModel(LabelStudioMLBase):
```

Then you can define all loaders in `__init__` method. 

```python
def __init__(self, **kwargs):
    # don't forget to initialize base class...
    super(MyModel, self).__init__(**kwargs)
    self.model = self.load_my_model()
```

Also there are special fields available inside the class to simplify Label Studio integration:

`self.parsed_label_config` - python dict that provides a Label Studio project config structure [ref for details]()
`self.train_output` - python dict with results of the previous model training runs. It is useful if you want to load the model for the next updates in case of active learning and finetuning.

Now you need to define two methods:

`predict(tasks, **kwargs)` - which takes [JSON-formatted Label Studio tasks](doc-link) and returns predictions in [format accepted by Label Studio](doc-link)
`fit(annotations, **kwargs)` - which takes [JSON-formatted Label Studio annotations](doc-link) and returns arbitrary dict (where some information about created model could be stored)

You can check more examples [here](label_studio_ml/examples). Those are listed just for the references, they are not production-ready models.