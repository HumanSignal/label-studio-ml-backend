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

3. Connect to the backend from Label Studio: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL.[requirements.txt](requirements.txt)


# Parameters

- `FLAIR_MODEL_NAME`: The name of the Flair model to use. Default is `ner`. See all options [here](https://flairnlp.github.io/docs/tutorial-basics/tagging-entities#list-of-ner-models)