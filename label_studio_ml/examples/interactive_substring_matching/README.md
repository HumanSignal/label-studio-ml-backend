# Interactive Substring Matching

The Machine Learning (ML) backend is designed to enhance the efficiency of auto-labeling in Named Entity Recognition (NER) tasks. It achieves this by selecting a keyword and automatically matching the same keyword in the texts. 

# Recommended Labeling Config

For the ML backend to work correctly, it is recommended to use the Named Entity Recognition (NER) template in Label Studio in project `Settings -> Labeling Interface -> Browse Templates -> Natural Language Processing -> Named Entity Recognition`.

Here is an example of a labeling configuration that can be used with this ML backend:

```xml
<View>
  <Labels name="label" toName="text">
    <Label value="ORG" background="orange" />
    <Label value="PER" background="lightgreen" />
    <Label value="LOC" background="lightblue" />
    <Label value="MISC" background="lightgray" />
  </Labels>
  <Text name="text" value="$text" />
</View>
```

## Running with Docker (Recommended)

1. Start Machine Learning backend on `http://localhost:9090` with prebuilt image:

```bash
docker-compose up
```

2. Validate that backend is running

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Connect to the backend from Label Studio running on the same host: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL.


## Building from source (Advanced)

To build the ML backend from source, you have to clone the repository and build the Docker image:

```bash
docker-compose build
```

## Running without Docker (Advanced)

To run the ML backend without Docker, you have to clone the repository and install all dependencies using pip:

```bash
python -m venv ml-backend
source ml-backend/bin/activate
pip install -r requirements.txt
```

Then you can start the ML backend:

```bash
label-studio-ml start ./interactive_substring_matching
```

# Configuration
Parameters can be set in `docker-compose.yml` before running the container.


The following common parameters are available:
- `BASIC_AUTH_USER` - specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - specify the basic auth password for the model server
- `LOG_LEVEL` - set the log level for the model server
- `WORKERS` - specify the number of workers for the model server
- `THREADS` - specify the number of threads for the model server

# Customization

The ML backend can be customized by adding your own models and logic inside the `./interactive_substring_matching` directory. 