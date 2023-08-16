## Quickstart

Build and start Machine Learning backend on `http://localhost:9090`

```bash
docker-compose up
```

Check if it works:

```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

Then connect running backend to Label Studio using Machine Learning settings.

## Supported configurations

This ML backend supports the following configurations:
- _Input data_: <Text>, <Image>, <HyperText>, <Paragraphs>
- At least one <TextArea> tag must be presented in the task config - it will be used as a prompt input. To specify which `<TextArea> tag` to use, set `PROMPT_PREFIX` environmental variable.
