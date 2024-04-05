<!--
---
title: Simple backend example
type: blog
tier: all
order: 90
meta_title: Simple backend for getting started with Label Studio models
meta_description: This is a tutorial on how to get started with Label Studio models using `the_simplest_backend` example. 
categories:
    - tutorial
    - getting started
image: "/tutorials/object-detection-with-bounding-boxes.png"
---
-->

## Quickstart

Build and start Machine Learning (ML) backend on `http://localhost:9090`. 

From the command line, run the following:
```bash
docker-compose up
```

Then, check the status of the ML backend:
```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

After starting the ML backend, [connect it to a Label Studio project](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio). 
