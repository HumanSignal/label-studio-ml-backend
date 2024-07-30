<!--
---
title: Question answering with RAG using Label Studio
type: guide
tier: all
order: 5 
hide_menu: true
hide_frontmatter_title: true
meta_title: RAG labeling with OpenAI using Label Studio
meta_description: Label Studio tutorial for RAG labeling with OpenAI and LangChain
categories:
    - Generative AI
    - Large Language Model
    - OpenAI
    - ChatGPT
    - RAG
    - Ragas
    - Embeddings
image: "/tutorials/ragas.png"
---
-->

# RAG Quickstart Labeling

This example server connects Label Studio to [OpenAI](https://platform.openai.com/), to interact with chat and embedding models. It supports question answering and evaluation using RAG, given a list of questions as tasks, and a folder containing documentation (e.g. a `/docs` path within a Github repository that has been cloned on your computer.)

## Starting the ML Backend

1. Make your reference documentation available to the backend.

Create a `docker-compose.override.yml` file alongside `docker-compose.yml`, and use it to mount a folder containing your documentation into the filesystem of the ML backend's image. This example will mount the folder at `/host/path/to/your/documentation` on your computer, to the path /data/documentation inside the ML backend Docker image. The `DOCUMENTATION_PATH` and `DOCUMENTATION_GLOB` settings given below will match all `.md` files within `/data/documentation` (or its subfolders).

```
services:
  rag_quickstart:
    volumes:
      - /host/path/to/your/documentation:/data/documentation
    environment:
      - DOCUMENTATION_PATH=/data/documentation
      - DOCUMENTATION_GLOB=**/*.md
      - OPENAI_API_KEY=<your OpenAI API key>
```

2. Build and start the Machine Learning backend on `http://localhost:9090` <br /><br />
```bash
docker-compose up
```

3. Check if it works: <br /><br />
 ```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

4. Open a Label Studio project and go to **Settings > Model**. [Connect the model](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio), specifying `http://localhost:9090` as the URL.

   Ensure the **Interactive preannotations** toggle is enabled and click **Validate and Save**.
5. Use the label config below. The config and backend can be customized to fit your needs.
6. Open a task and ensure the **Auto-Annotation** toggle is enabled (it is located at the bottom of the labeling interface).
7. The text fields should be auto-completed by the LLM. However, you can provide additional instructions with the field provided. To submit, press `Shift+Enter`. The LLM response will be generated and displayed in the response field.
8. If you want to apply LLM auto-annotation to multiple tasks at once, go to the [Data Manager](https://labelstud.io/guide/manage_data), select a group of tasks and then select **Actions > Retrieve Predictions** (or **Batch Predictions** in Label Studio Enterprise).

## Label Config

```
<View>
    <Style>
        .lsf-main-content.lsf-requesting .prompt::before { content: ' loading...'; color: #808080; }
        .text-container {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        font-family: 'Courier New', monospace;
        line-height: 1.6;
        font-size: 16px;
        }
        .ragas input {
            background: none;
            border: none;
            padding: 0;
            margin-top: -8px;
            font-size: 20px;
            font-weight: 600;
        }
        .ragas input::-webkit-inner-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }
    </Style>
    <Header value="Question:"/>
    <View className="text-container">
        <Text name="context" value="$text"/>
    </View>
    <Header value="Additional instructions for the LLM prompt (optional):"/>
    <View className="prompt">
        <TextArea name="prompt"
                  toName="context"
                  rows="4"
                  editable="true"
                  showSubmitButton="false"
                  placeholder="Provide additional instructions here then Shift+Enter - to provide none, simply enter a space then shift+enter."
        />
    </View>
    <Header value="Response:"/>
    <TextArea name="response"
              toName="context"
              rows="4"
              editable="true"
              maxSubmissions="1"
              showSubmitButton="false"
              smart="false"
              placeholder="Generated response will appear here..."
    />
  	<View className="ragas" >
    <View style="display: flex;">
      <Header style="padding-right: 1em;" value="RAGAS evaluation (averaged, 0 to 100):"/><Number name="float_eval" toName="context" defaultValue="0" />
    </View>
    <TextArea name="ragas"
              toName="context"
              rows="2"
              maxSubmissions="1"
              showSubmitButton="false"
              smart="false"
              placeholder="RAGAS evaluation will appear here..."
    />
  	</View>
    <View className="evaluation" >
    <View style="display: flex;">
      <Header style="padding-right: 1em;" value="Textual evaluation:"/>
    </View>
    <TextArea name="evaluation"
              toName="context"
              rows="2"
              maxSubmissions="1"
              showSubmitButton="false"
              smart="false"
              placeholder="Textual evaluation will appear here..."
    />
    </View>
    <Header value="Documentation:"/>
    <View className="documentation">
    <TextArea name="documentation"
              toName="context"
              rows="2"
              maxSubmissions="1"
              showSubmitButton="false"
              smart="false"
              placeholder="Retrieved documentation will appear here..."
    />
    </View>
</View>
```

For more information on this labeling config, see the [Evaluate RAG with Ragas](https://labelstud.io/templates/llm_ragas) template documentation.

**Example data input:**

Tip: when generating questions for your project, it may be helpful to pass this snippet to ChatGPT etc to give it an example of Label Studio's tasks format to work from.

```
[
  {
    "id": 1,
    "data": {
      "text": "How do I install Label Studio?"
    }
  },
  {
    "id": 2,
    "data": {
      "text": "How do I reinstall Label Studio?"
    }
  }
]
```