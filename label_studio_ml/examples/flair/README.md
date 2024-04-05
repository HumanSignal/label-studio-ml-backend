<!--
---
title: NLP labeling with Flair 
type: blog
tier: all
order: 65
meta_title: Use Flair with Label Studio
meta_description: Tutorial on how to use Label Studio and Flair for faster NLP labeling 
categories:
    - tutorial
    - named language processing
    - flair
image: "/tutorials/flair.png"
---
-->

# Quickstart

1. Build and start the Machine Learning backend on `http://localhost:9090`

```bash
docker-compose up
```

2. Validate that the backend is running

```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

3. Create a project in Label Studio. Then from the **Model** page in the project settings, [connect the model](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio). Specify `http://localhost:9090` as the URL.


# Parameters

- `FLAIR_MODEL_NAME`: The name of the Flair model to use. Default is `ner`. See all options [here](https://flairnlp.github.io/docs/tutorial-basics/tagging-entities#list-of-ner-models)