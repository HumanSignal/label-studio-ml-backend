## Interactive LLM labeling

This example server connects Label Studio to [OpenAI](https://platform.openai.com/) or [Azure](https://azure.microsoft.com/en-us/products/ai-services/openai-service) API to interact with GPT chat models (gpt-3.5-turbo, gpt-4, etc.).

The interactive flow allows you to perform the following scenarios:

1. Autolabel data given LLM prompt (e.g. "Classify this text as sarcastic or not")
2. Collect pair of input user prompts and responses to finetune your own LLM
3. Automate data collection and summarization over image documents
4. Create RLHF (Reinforcement Learning from Human Feedback) loop to improve LLM's performance
5. Evaluate LLM's performance.

Check [Generative AI templates](https://labelstud.io/templates/gallery_generative_ai) section for more examples.

## Quickstart

1. Build and start Machine Learning backend on `http://localhost:9090`

```bash
docker-compose up
```

Check if it works:

```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

2. Open [Label Studio project](http://localhost:8080) and navigate to `Settings > Machine Learning` page. Add a new ML
   backend and specify the URL `http://localhost:9090`. Ensure `Use for interactive preannotations` toggle is set **ON**
   . Save the settings.
3. The project config should be compatible with the ML backend. This ML backend can support various input data formats
   like plain text, hypertext, image, structured dialogs. To ensure the project config is compatible, follow these
   rules:

- The project should contain at least one `<TextArea>` tag to be used as a prompt input. To specify which `<TextArea>`
  tag to use, set `PROMPT_PREFIX` environmental variable. For example, `<TextArea name="prompt" ...>` tag should be used
  with `PROMPT_PREFIX=prompt`.
- The project should contain at least one input data tag from the following list of supported tags: `<Text>`, `<Image>`
  , `<HyperText>`, `<Paragraphs>`.
- If you want to directly capture generated LLM response, your labeling config should contain a `<TextArea>` tag
  different from the prompt input. To specify which `<TextArea>` tag to use, set `RESPONSE_PREFIX` environmental
  variable. For example, `<TextArea name="response" ...>`.
- If you want to capture generated LLM response as a label, your labeling config should contain a `<Choices>` tag. For
  example, `<Choices name="choices" ...>`.

4. Go to the labeling page, and ensure the `Auto-Annotation` toggle is enabled (it is located below the labeling screen)
   .
5. Type a prompt in the prompt input field and press `Shift+Enter`. The LLM response will be generated and displayed in
   the response field.
6. If you want to apply LLM auto-annotation to the multiple tasks at once , go to the Data Manager, select the batch of tasks then use `Batch Predictions` option from [the `Actions` dropdown](https://labelstud.io/guide/manage_data)

## Configuration examples

#### Automatic text classification

```xml

<View>
    <Style>
        .lsf-main-content.lsf-requesting .prompt::before { content: ' loading...'; color: #808080; }
    </Style>
    <!-- Input data -->
    <Text name="text" value="$text"/>
    <!-- Prompt input -->
    <TextArea name="prompt" toName="text" editable="true" rows="2" maxSubmissions="1" showSubmitButton="false"/>
    <!-- LLM response output -->
    <TextArea name="response" toName="text" editable="true"/>
    <View style="box-shadow: 2px 2px 5px #999;
               padding: 20px; margin-top: 2em;
               border-radius: 5px;">
        <Choices name="sentiment" toName="text"
                 choice="multiple" showInLine="true">
            <Choice value="Sarcastic"/>
            <Choice value="Not Sarcastic"/>
        </Choices>
    </View>
</View>
```

**Example data input:**

```json
{
  "text": "I love it when my computer crashes"
}
```

#### Collecting data for LLM supervised finetuning

Representing ChatGPT-style interface with [`<Paragraphs>`](https://labelstud.io/tags/paragraphs) tag:

```xml

<View>
    <Style>
        .lsf-main-content.lsf-requesting .prompt::before { content: ' loading...'; color: #808080; }
    </Style>
    <Paragraphs name="chat" value="$dialogue" layout="dialogue" textKey="content" nameKey="role"/>
    <Header value="User prompt:"/>
    <View className="prompt">
        <TextArea name="prompt" toName="chat" rows="4" editable="true" maxSubmissions="1" showSubmitButton="false"/>
    </View>
    <Header value="Bot answer:"/>
    <TextArea name="response" toName="chat" rows="4" editable="true" maxSubmissions="1" showSubmitButton="false"/>

</View>
```

**Example data input:**

```json
{
  "dialogue": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": "The capital of France is Paris."
    },
    {
      "role": "user",
      "content": "Tell me a joke."
    }
  ]
}
```

#### Automating data collection and summarization over image documents

```xml

<View>
    <Style>
        .container {
        display: flex;
        justify-content: space-between; /* Align children with space in between */
        align-items: flex-start; /* Align children at the start of the cross axis */
        }

        .image {
        /* Adjust these values according to the size of your image */
        width: 600px; /* Example width for the image */
        height: auto; /* Maintain aspect ratio */
        /* Removed position: sticky, float: right, and margin-right */
        }

        .blocks {
        width: calc(100% - 220px); /* Adjust the calculation to account for the image width and some margin */
        height: 300px; /* Set the height for the scrolling area */
        overflow-y: scroll; /* Allow vertical scrolling */
        }

        .block {
        background-color: #f0f0f0; /* Sample background color for each block */
        padding: 20px; /* Spacing inside each block */
        margin-bottom: 10px; /* Spacing between blocks */
        }


    </Style>
    <View className="container">
        <View className="blocks">

            <View className="block">
                <Header value="Classification:"/>
                <TextArea name="classification-prompt" toName="image"
                          showSubmitButton="false"
                          editable="true"
                          rows="3"
                          required="true"/>
                <Choices name="category" toName="image" layout="select">
                    <Choice value="Groceries"/>
                    <Choice value="Dining/Restaurants"/>
                    <Choice value="Clothing/Apparel"/>
                    <Choice value="Electronics"/>
                    <Choice value="Home Improvement"/>
                    <Choice value="Health/Pharmacy"/>
                    <Choice value="Gasoline/Fuel"/>
                    <Choice value="Transportation/Travel"/>
                    <Choice value="Entertainment/Leisure"/>
                    <Choice value="Utilities/Bills"/>
                    <Choice value="Insurance"/>
                    <Choice value="Gifts/Donations"/>
                    <Choice value="Personal Care"/>
                    <Choice value="Education/Books"/>
                    <Choice value="Professional Services"/>
                    <Choice value="Membership/Subscriptions"/>
                    <Choice value="Taxes"/>
                    <Choice value="Vehicle Maintenance/Repairs"/>
                    <Choice value="Pet Care"/>
                    <Choice value="Home Furnishings/Decor"/>
                    <Choice value="Other"/>
                </Choices>
            </View>
            <View className="block">
                <Header value="Summary:"/>
                <TextArea name="summarization-response" toName="image"
                          showSubmitButton="false"
                          maxSubmissions="0"
                          editable="true"
                          rows="3"
                />
            </View>
        </View>
        <View className="image">
            <Image name="image" value="$image"/>
        </View>
    </View>
</View>
```

**Example data input:**

```json
{
  "image": "https://sandbox2-test-bucket.s3.amazonaws.com/receipts/113494_page1.png"
}
```

## Parameters
When deploying the server, you can specify the following parameters as environment variables:

- `PROMPT_PREFIX` (default: `prompt`): An identifier for the prompt input field. For example, if you set
  `PROMPT_PREFIX` to `my-prompt`, the following input field will be used for the prompt: `<TextArea name="my-prompt" ...>`.
- `USE_INTERNAL_PROMPT_TEMPLATE` (default: `1`). If set to `1`, the server will use the internal prompt template. If set to
  `0`, the server will use the prompt template provided in the input prompt.
- `PROMPT_TEMPLATE` (default: `"Source Text: {text}\n\nTask Directive: {prompt}"`): The prompt template to use. If `USE_INTERNAL_PROMPT_TEMPLATE` is set to `1`, the server will use
  the default internal prompt template. If `USE_INTERNAL_PROMPT_TEMPLATE` is set to `0`, the server will use the prompt template provided
  in the input prompt (i.e. the user input from `<TextArea name="my-prompt" ...>`). In the later case, the user has to provide the placeholders that match input task fields. For example, if the user wants to use the `input_text` and `instruction` field from the input task `{"input_text": "user text", "instruction": "user instruction"}`, the user has to provide the prompt template like this: `"Source Text: {input_text}, Custom instruction : {instruction}"`.
- `OPENAI_MODEL` (default: `gpt-3.5-turbo`) : The OpenAI model to use. 
- `OPENAI_PROVIDER` (available options: `openai`, `azure`, default - `openai`) : The OpenAI provider to use.
- `TEMPERATURE` (default: `0.7`): The temperature to use for the model.
- `NUM_RESPONSES` (default: `1`): The number of responses to generate in `<TextArea>` output fields. Useful if you want to generate multiple responses and let the user rank the best one.
- `OPENAI_API_KEY`: The OpenAI or Azure API key to use. Must be set before deploying the server.

### Azure Configuration

If you are using Azure as your OpenAI provider (`OPENAI_PROVIDER=azure`), you need to specify the following environment variables:

- `AZURE_RESOURCE_ENDPOINT`: This is the endpoint for your Azure resource. It should be set to the appropriate value based on your Azure setup.

- `AZURE_DEPLOYMENT_NAME`: This is the name of your Azure deployment. It should match the name you've given to your deployment in Azure.

- `AZURE_API_VERSION`: This is the version of the Azure API you are using. The default value is `2023-05-15`.