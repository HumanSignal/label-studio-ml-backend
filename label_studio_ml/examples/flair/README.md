# Flair NER example

This example demonstrates how to use Flair NER model with Label Studio.

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

3. Connect to the backend from Label Studio: go to your project `Settings -> Model -> Connect Model` and specify `http://localhost:9090` as a URL.

## Labeling Configuration

```xml
<View>
  <Labels name="label" toName="text">
    <Label value="PER" background="red"/>
    <Label value="ORG" background="darkorange"/>
    <Label value="LOC" background="orange"/>
    <Label value="MISC" background="green"/>
  </Labels>

  <Text name="text" value="$text"/>
</View>
```


# Parameters

- `FLAIR_MODEL_NAME`: The name of the Flair model to use. Default is `ner`. See all options [here](https://flairnlp.github.io/docs/tutorial-basics/tagging-entities#list-of-ner-models)