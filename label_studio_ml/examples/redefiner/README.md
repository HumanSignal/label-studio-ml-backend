<!--
---
title: Use ReDefiNER for rule-based NER pre-annotation
type: guide
tier: all
order: 38
hide_menu: true
hide_frontmatter_title: true
meta_title: Use ReDefiNER for NER pre-annotation from your labeling guidelines
meta_description: Tutorial on pre-annotating NER tasks in Label Studio with ReDefiNER, using your labels and their descriptions as a plain-English rulebook
categories:
    - Natural Language Processing
    - Named Entity Recognition
    - ReDefiNER
    - Hugging Face
---
-->

# Use ReDefiNER for NER pre-annotation

[ReDefiNER](https://github.com/Jungroo-AI/redefiner) is an open 0.8B model (Apache 2.0) that extracts the entity types
you describe in a plain-English rulebook. It marks spans in your text and does not generate text. This backend turns
your project's labels and their descriptions into a rulebook, so pre-annotations follow your labeling guidelines. It
runs on CPU.

ReDefiNER is not fine-tuned on your annotations. When the pre-annotations make the same mistake again and again, add a
description or a rule that says what you want, and the next predictions follow it.

## Before you begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart).

This tutorial uses the [`redefiner` example](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/redefiner).

## Labeling configuration

Use a `<Labels>` tag over a `<Text>` tag. Each `<Label>` becomes an entity type, and its `hint` becomes the type's
description. Write descriptions the way you would brief an annotator: they matter more than the label names.

```xml
<View>
  <Labels name="label" toName="text">
    <Label value="PERSON" background="red" hint="A person's name, without titles such as Dr or Prof."/>
    <Label value="ORGANIZATION" background="blue" hint="A company, university or other institution."/>
    <Label value="PRODUCT" background="green" hint="A named product or software, with its version number if given."/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
```

Label Studio shows the hint when an annotator hovers over the label, so the model and your annotators read the same
definitions.

## Running with Docker (recommended)

1. Start the Machine Learning backend on `http://localhost:9090`:

```bash
docker-compose up
```

The first prediction downloads the model from Hugging Face (about 0.8 GB) into `./data/server`.

2. Validate that the backend is running:

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Create a project in Label Studio. Then from the **Model** page in the project settings, [connect the model](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio). The default URL is `http://localhost:9090`.

## Running without Docker (advanced)

```bash
python -m venv ml-backend
source ml-backend/bin/activate
pip install -r requirements-base.txt -r requirements.txt
label-studio-ml start ./redefiner
```

## Writing rules

Descriptions are often enough. For more control, write a full rulebook and point `RULEBOOK` at it. Rules can say what
to leave out, where a span starts and ends, and which type wins when two could apply:

```
[RULES]
Extract every entity of the supplied types. Return exact spans from the text.
G1 Never label job titles or department names.

[TYPE] PERSON
[DESCRIPTION] A person's name.
PE1 Label the full name; leave out titles such as "Dr" or "Prof".

[TYPE] ORGANIZATION
[DESCRIPTION] A company, university or other institution.

[TYPE] PRODUCT
[DESCRIPTION] A named product or software, with its version number if given.
PR1 A product name that starts with its company's name is one PRODUCT span: "Google Maps", "Microsoft Excel 2021".
```

The `[TYPE]` names must match your label values. Labels without a `[TYPE]` are never predicted, and types that are not
labels in the project are dropped. See the [guide to writing rules](https://github.com/Jungroo-AI/redefiner/blob/main/docs/writing-rules.md).

Each rulebook gets its own model version (`redefiner-` and a hash of the rulebook), so after an edit you can compare the
old and new predictions in Label Studio.

## PII and redaction projects

`jungroo-ai/redefiner-pii` ships a rulebook for personal data. Set `REDEFINER_MODEL=jungroo-ai/redefiner-pii` and
`RULEBOOK=default`, and use its type names as your labels: `PERSON`, `TITLE`, `USERNAME`, `EMAIL`, `PHONE`, `ADDRESS`,
`LOCATION`, `COORDINATE`, `ORGANIZATION`, `OCCUPATION`, `DATE_TIME`, `AGE`, `GENDER`, `DEMOGRAPHIC`, `GOVERNMENT_ID`,
`ID_NUMBER`, `IP_ADDRESS`, `URL`, `FINANCIAL`, `SECRET`. You can use only some of them.

## Configuration

Parameters can be set in `docker-compose.yml` before running the container.

- `REDEFINER_MODEL` - The model page on Hugging Face. Default: `jungroo-ai/redefiner-0.8b`.
- `RULEBOOK` - Leave empty to build the rulebook from the labeling config. Otherwise rulebook text, the path of a
  rulebook file, a folder of `<project id>.txt` rulebooks (projects without a file use their labels), or `default` for
  the model page's own rulebook.
- `THRESHOLD` - Minimum span probability. Empty uses the model's default (0.5). Lower it for more pre-annotations.
- `DEVICE` - `cpu` or `cuda`. Empty uses a GPU when one is available.
- `BATCH_SIZE` - Texts per model batch. Default: 16.
- `MAX_SPAN_TOKENS` - Longest span in model tokens. Raise it for long spans such as addresses.
- `BASIC_AUTH_USER` - Specify the basic auth user for the model server.
- `BASIC_AUTH_PASS` - Specify the basic auth password for the model server.
- `LOG_LEVEL` - Set the log level for the model server.
- `WORKERS` - Specify the number of workers for the model server. Each worker loads its own copy of the model.
- `THREADS` - Specify the number of threads for the model server.
- `LABEL_STUDIO_URL` - Specify the URL of your Label Studio instance. Note that this might need to be `http://host.docker.internal:8080` if you are running Label Studio on another Docker container.
- `LABEL_STUDIO_API_KEY`- Specify the API key for authenticating your Label Studio instance. You can find this by logging into Label Studio and [going to the **Account & Settings** page](https://labelstud.io/guide/user_account#Access-token).

## Notes

- Each prediction's score is the confidence of its least confident span, so sorting tasks by prediction score puts
  the most uncertain pre-annotations first.
- Long texts are split into overlapping windows of about 256 model tokens. On CPU, a sentence takes about half a
  second.
- ReDefiNER reads English best.
