<!--
---
title: Backend example for LangChain integration
type: blog
tier: all
order: 80
meta_title: Label Studio ML backend for LangChain
meta_description: This is a tutorial on how to use the Label Studio ML backend with LangChain to power your labeling projects 
categories:
    - tutorial
    - langchain
image: "/tutorials/object-detection-with-bounding-boxes.png"
---
-->

# Quickstart

1. Build and start Machine Learning backend on `http://localhost:9090`

```bash
docker-compose up
```

2. Validate that backend is running

```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

3. Connect to the backend from Label Studio: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL.