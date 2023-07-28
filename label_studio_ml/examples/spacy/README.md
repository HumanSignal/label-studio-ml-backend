Spacy ML backend provides a simple way to use [spaCy](https://spacy.io/) models for Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.

Current implementation includes the following models:
- Named Entity Recognition (NER)
- [coming soon...] Part-of-Speech (POS) tagging


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

# Usage

## Labeling configuration

The model is compatible with the following labeling configurations:
```xml
<View>
    <Labels name="label" toName="text">
        <Label value="CARDINAL" background="#FFA39E"/>
        <Label value="DATE" background="#D4380D"/>
        <Label value="EVENT" background="#FFC069"/>
        <Label value="FAC" background="#AD8B00"/>
        <Label value="GPE" background="#D3F261"/>
        <Label value="LANGUAGE" background="#389E0D"/>
        <Label value="LAW" background="#5CDBD3"/>
        <Label value="LOC" background="#096DD9"/>
        <Label value="ORG" background="#ADC6FF"/>
        <Label value="PERSON" background="#9254DE"/>
        <Label value="TIME" background="#F759AB"/>
    </Labels>
    <Text name="text" value="$text"/>
</View>
```

## Parameters
To change default parameters, specify the following environment variables:

- `PORT` - port to run the server on, default is `9090`
- `WORKERS` - number of workers to run the server with, default is `2`
- `SPACY_MODEL` - spaCy model to use, default is `en_core_web_sm`