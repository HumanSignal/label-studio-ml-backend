<!--
---
title: Interactive LLM labeling with GPT
type: guide
tier: all
order: 5
hide_menu: true
hide_frontmatter_title: true
meta_title: Interactive LLM labeling with OpenAI, Azure, or Ollama
meta_description: Label Studio tutorial for interactive LLM labeling with OpenAI, Azure, or Ollama
categories:
    - Generative AI
    - Large Language Model
    - OpenAI
    - Azure
    - Ollama
    - ChatGPT
image: "/tutorials/llm-interactive.png"
---
-->

# Interactive LLM labeling

This example server connects Label Studio to [OpenAI](https://platform.openai.com/), [Ollama](https://ollama.com/),
or [Azure](https://azure.microsoft.com/en-us/products/ai-services/openai-service) API to interact with GPT chat models (
gpt-3.5-turbo, gpt-4, etc.).

The interactive flow allows you to perform the following scenarios:

* Autolabel data given an LLM prompt (e.g. "Classify this text as sarcastic or not")
* Collect pairs of user prompts and response inputs to fine tune your own LLM.
* Automate data collection and summarization over image documents.
* Create a RLHF (Reinforcement Learning from Human Feedback) loop to improve the LLM's performance.
* Evaluate the LLM's performance.

Check the [Generative AI templates](https://labelstud.io/templates/gallery_generative_ai) section for more examples.

## Quickstart

1. Build and start the Machine Learning backend on `http://localhost:9090` <br /><br />
```bash
docker-compose up
```

2. Check if it works: <br /><br />
 ```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

3. Open a Label Studio project and go to **Settings > Model**. [Connect the model](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio), specifying `http://localhost:9090` as the URL.

   Ensure the **Interactive preannotations** toggle is enabled and click **Validate and Save**.
4. The project config should be compatible with the ML backend. This ML backend can support various input data formats
   like plain text, hypertext, images, and structured dialogs. To ensure the project config is compatible, follow these
   rules:

   - The project should contain at least one `<TextArea>` tag to be used as a prompt input. To specify which `<TextArea>` tag  to use, set the `PROMPT_PREFIX` environment variable.
   For example, if your labeling config includes `<TextArea name="prompt" ...>`, then you would specify `PROMPT_PREFIX=prompt`.
   - The project should contain at least one input data tag from the following list of supported tags: `<Text>`, `<Image>`, `<HyperText>`, `<Paragraphs>`.
   - If you want to capture the generated LLM response as a label, your labeling config should contain a `<Choices>` tag.
   For example, `<Choices name="choices" ...>`.
   - If you want to set the default prompt to be shown before the user input, you can set the `DEFAULT_PROMPT` environment variable. For example, `DEFAULT_PROMPT="Classify this text as sarcastic or not. Text: {text}, Labels: {labels}"` or `DEFAULT_PROMPT=/path/to/prompt.txt`.

    Note that the default prompt isn't supported with `USE_INTERNAL_PROMPT_TEMPLATE=1` mode, so you will need to set `USE_INTERNAL_PROMPT_TEMPLATE=0` to use default prompt. You can use the fields from `task['data']` in the prompt template, as well as special `{labels}` field to show the list of available labels.

5. Open a task and ensure the **Auto-Annotation** toggle is enabled (it is located at the bottom of the labeling interface).
6. Enter a prompt in the prompt input field and press `Shift+Enter`. The LLM response will be generated and displayed in
   the response field.
7. If you want to apply LLM auto-annotation to multiple tasks at once, go to the [Data Manager](https://labelstud.io/guide/manage_data), select a group of tasks and then select **Actions > Retrieve Predictions** (or **Batch Predictions** in Label Studio Enterprise).

## Configuration examples

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
    </Style>
    <Header value="Question:"/>
    <View className="text-container">
        <Text name="context" value="$text"/>
    </View>
    <Header value="Additional context (optional):"/>
    <View className="prompt">
        <TextArea name="prompt"
                  toName="context"
                  rows="4"
                  editable="true"
                  showSubmitButton="false"
                  placeholder="Provide additional context here then Shift+Enter - to provide none, simply enter a space then shift+enter."
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
</View>

**Example data input:**

```
{
  "data": {
    "text": "How do I install Label Studio?"
  }
}
```

## Parameters

When deploying the server, you can specify the following parameters as environment variables:

- `DEFAULT_PROMPT`: Define a default prompt to be shown before the user input. For example, `DEFAULT_PROMPT="Classify this text as sarcastic or not. Text: {text}, Labels: {labels}"` or `DEFAULT_PROMPT=/path/to/prompt.txt`.

    Note that `USE_INTERNAL_PROMPT_TEMPLATE` should be set to `0` if you are setting a default prompt.

- `PROMPT_PREFIX` (default: `prompt`): An identifier for the prompt input field. For example, if you set
  `PROMPT_PREFIX` to `my-prompt`, the following input field will be used for the
  prompt: `<TextArea name="my-prompt" ...>`.

- `USE_INTERNAL_PROMPT_TEMPLATE` (default: `1`). If set to `1`, the server will use the internal prompt template. If set
  to
  `0`, the server will use the prompt template provided in the input prompt.

- `PROMPT_TEMPLATE` (default: `"Source Text: {text}\n\nTask Directive: {prompt}"`): The prompt template to use:

  - If `USE_INTERNAL_PROMPT_TEMPLATE` is set to `1`, the server will use
  the default internal prompt template.

  - If `USE_INTERNAL_PROMPT_TEMPLATE` is set to `0`, the server will use the prompt template provided
  in the input prompt (i.e. the user input from `<TextArea name="my-prompt" ...>`).

  In the later case, the user has to provide the placeholders that match input task fields. For example, if the user wants to use the `input_text` and `instruction` field from the input task `{"input_text": "user text", "instruction": "user instruction"}`, the user has to provide the prompt template like this: `"Source Text: {input_text}, Custom instruction : {instruction}"`.

- `OPENAI_MODEL` (default: `gpt-3.5-turbo`) : The OpenAI model to use.

- `OPENAI_PROVIDER` (available options: `openai`, `azure`, `ollama`, default - `openai`) : The OpenAI provider to use.

- `TEMPERATURE` (default: `0.7`): The temperature to use for the model.

- `NUM_RESPONSES` (default: `1`): The number of responses to generate in `<TextArea>` output fields. Useful if you want
  to generate multiple responses and let the user rank the best one.

- `OPENAI_API_KEY`: The OpenAI or Azure API key to use. Must be set before deploying the server.

### Azure Configuration

If you are using Azure as your OpenAI provider (`OPENAI_PROVIDER=azure`), you need to specify the following environment
variables:

- `AZURE_RESOURCE_ENDPOINT`: This is the endpoint for your Azure resource. It should be set to the appropriate value
  based on your Azure setup.

- `AZURE_DEPLOYMENT_NAME`: This is the name of your Azure deployment. It should match the name you've given to your
  deployment in Azure.

- `AZURE_API_VERSION`: This is the version of the Azure API you are using. The default value is `2023-05-15`.

### Ollama Configuration

If you are using Ollama as your LLM provider (`OPENAI_PROVIDER=ollama`), you need to specify the following environment variables:

- `OPENAI_MODEL` : The Ollama model to use, for example `llama3`.

- `OLLAMA_ENDPOINT`: This is the endpoint for your Ollama endpoint. It should be set to the appropriate value based on your setup. If you are running it locally, then it can typically be reached on `http://host.docker.internal:11434/v1/`
