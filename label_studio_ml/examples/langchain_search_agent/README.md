# Langchain Search Agent

This example demonstrates how to use Label Studio with a custom Machine Learning backend.
It uses [Langchain](https://www.langchain.com/)-based agent that takes a text input, search for Google search results,
and returns the answer based on the search results (a.k.a Retrieval Augmented Generation).

# Pre-requisites

## Use Google Search Engine

To use Google Search Engine, you need to have a Google Custom Search Engine (CSE) API key and a search engine ID.

```
GOOGLE_API_KEY=<your_google_api_key>
GOOGLE_CSE_ID=<your_google_search_engine_id>
```

## Use OpenAI

To use OpenAI, you need to have an OpenAI API key.

```
OPENAI_API_KEY=<your_openai_api_key>
```

More details [here](https://support.google.com/programmable-search/answer/12499034?hl=en).

# Labeling Interface

The labeling interface must include:

- input prompt
- LLM response
- search results snippets
- classification labels

#### Example

```xml

<View>
    <Style>
        .lsf-main-content.lsf-requesting .prompt::before { content: ' loading...'; color: #808080; }
    </Style>
    <Text name="input" value="$text"/>
    <View className="prompt">
        <TextArea name="prompt" toName="input" maxSubmissions="1" editable="true"/>
    </View>
    <TextArea name="response" toName="input" maxSubmissions="1" editable="true"/>
    <TextArea name="snippets" toName="input"/>
    <Choices name="classification" toName="input" choice="single" showInLine="true">
        <Choice value="Good"/>
        <Choice value="Bad"/>
    </Choices>
</View>
```

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

3. Connect to the backend from Label Studio: go to your project `Settings -> Machine Learning -> Add Model` and
   specify `http://localhost:9090` as a URL.