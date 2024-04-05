<!--
---
title: Transfer learning with PyTorch
type: blog
tier: all
order: 50
meta_title: Label Studio backend to demonstrate transfer learning with PyTorch
meta_description: This is a tutorial on how to use the example model for image classification with the possibility of transfer learning. 
categories:
    - tutorial
    - pytorch
    - image classification
image: "/tutorials/pytorch.png"
---
-->

## Overview 

This is the Label Studio ML Backend for image classification with the possibility of transfer learning. 

## Quickstart

Build and start the Machine Learning backend on `http://localhost:9090`

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
label-studio start --init new_project --ml-backends http://localhost:9090 --template image_classification
```


## Writing your own model

1. Place your scripts for model training & inference inside root directory. Follow the [API guidelines](#api-guidelines) described bellow. You can put everything in a single file, or create 2 separate one say `my_training_module.py` and `my_inference_module.py`

2. Write down your python dependencies in `requirements.txt`

3. Create ML backend with your model
```bash
label-studio-ml init my-ml-backend --script pytorch_transfer_learning/pytorch_transfer_learning.py
```

4. Start ML backend at http://localhost:9090
```bash
label-studio-ml start my-ml-backend
```

5. Start Label Studio with ML backend connection
```bash
label-studio start my-annotation-project --init --ml-backend http://localhost:9090
```
   
## API guidelines

Check out [how to create your own ML backend](https://github.com/HumanSignal/label-studio-ml-backend/tree/master?tab=readme-ov-file#advanced-usage-develop-your-model).

## License

This software is licensed under the [Apache 2.0 LICENSE](/LICENSE) © [HumanSignal](https://www.humansignal.com/). 2024

<img src="https://github.com/heartexlabs/label-studio/blob/master/images/opossum_looking.png?raw=true" title="Hey everyone!" height="140" width="140" />
