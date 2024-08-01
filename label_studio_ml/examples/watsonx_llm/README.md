

# Integrate WatsonX to Label Studio
WatsonX offers a suite of machine learning tools, including access to many LLMs, prompt
refinement interfaces, and datastores via WatsonX.data. When you integrate WatsonX with Label Studio, you get 
access to these models and can automatically keep your annotated data up to date in your WatsonX.data tables. 

To run the integration, you'll need to pull this repo and host it locally or in the cloud. Then, you can link the model 
to your Label Studio project under the `models` section in the settings. To use the WatsonX.data integration, 
set up a webhook in settings under `webhooks` by using the following structure for the link: 
`<link to your hosted container>/data/upload` and set the triggers to `ANNOTATION_CREATED` and `ANNOTATION_UPDATED`.

See the configuration notes at the bottom for details on how to set up your environment variables to get the system to work.

## Running with Docker (recommended)

1. Start Machine Learning backend on `http://localhost:9090` with prebuilt image:

```bash
docker-compose up
```

2. Validate that backend is running

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Create a project in Label Studio. Then from the **Model** page in the project settings, [connect the model](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio). The default URL is `http://localhost:9090`.


## Building from source (advanced)

To build the ML backend from source, you have to clone the repository and build the Docker image:

```bash
docker-compose build
```

## Running without Docker (advanced)

To run the ML backend without Docker, you have to clone the repository and install all dependencies using pip:

```bash
python -m venv ml-backend
source ml-backend/bin/activate
pip install -r requirements.txt
```

Then you can start the ML backend:

```bash
label-studio-ml start ./dir_with_your_model
```

## Configuration

Parameters can be set in `docker-compose.yml` before running the container.

The following common parameters are available:
- `BASIC_AUTH_USER` - Specify the basic auth user for the model server.
- `BASIC_AUTH_PASS` - Specify the basic auth password for the model server.
- `LOG_LEVEL` - Set the log level for the model server.
- `WORKERS` - Specify the number of workers for the model server.
- `THREADS` - Specify the number of threads for the model server.

The following parameters allow you to link the WatsonX models to Label Studio:

- `LABEL_STUDIO_URL` - Specify the URL of your Label Studio instance. Note that this might need to be `http://host.docker.internal:8080` if you are running Label Studio on another Docker container.
- `LABEL_STUDIO_API_KEY`- Specify the API key for authenticating your Label Studio instance. You can find this by logging into Label Studio and and [going to the **Account & Settings** page](https://labelstud.io/guide/user_account#Access-token).
- `WATSONX_API_KEY`- Specify the API key for authenticating into WatsonX. You can generate this by following the instructions at [here](https://www.ibm.com/docs/en/watsonx/watsonxdata/1.0.x?topic=started-generating-api-keys)
- `WATSONX_PROJECT_ID`- Specify the ID of your WatsonX project from which you will run the model. Must have WML capabilities. You can find this in the `General` section of your project, which is accessible by clicking on the project from the homepage of WatsonX.
- `WATSONX_MODELTYPE`- Specify the name of the WatsonX model you'd like to use. A full list can be found in [IBM's documentation](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#TextModels:~:text=CODELLAMA_34B_INSTRUCT_HF)
- `DEFAULT_PROMPT` - If you want the model to automatically predict on new data samples, you'll need to provide a default prompt or the location to a default prompt file. 
- `USE_INTERNAL_PROMPT` - If using a default prompt, set to 0. Otherwise, set to 1.  

The following parameters allow you to use the webhook connection to transfer data from Label Studio to WatsonX.data:

-`WATSONX_ENG_USERNAME`- MUST be `ibmlhapikey` for the intergration to work.

To get the host and port information below, you can folllow the steps under [Pre-requisites](https://cloud.ibm.com/docs/watsonxdata?topic=watsonxdata-con-presto-serv#conn-to-prestjava).

- `WATSONX_ENG_HOST` - the host information for your WatsonX.data Engine
- `WATSONX_ENG_PORT` - the port information for your WatsonX.data Engine
- `WATSONX_CATALOG` - the name of the catalog for the table you'll insert your data into. Must be created in the WatsonX.data platform.
- `WATSONX_SCHEMA` - the name of the schema for the table you'll insert your data into. Must be created in the WatsonX.data platofrm.
- `WATSONX_TABLE` - the name of the table you'll insert your data into. Does not need to be already created.

