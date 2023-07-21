# OpenAI Label Studio ML Backend Example
This repository provides an example of how to use the OpenAI GPT model with the Label Studio Machine Learning (ML) backend. The Label Studio ML backend is an SDK that allows you to wrap your machine learning code and turn it into a web server. You can then connect that server to a Label Studio instance to perform tasks such as dynamically pre-annotating data based on model inference results.

## Getting Started
### Prerequisites
Before you begin, ensure you have the following installed:
- Docker
- Docker Compose

### Installation
1. Clone the repository:

```bash
git clone https://github.com/JimmyWhitaker/label-studio-ml-backend.git
```

2. Navigate to the example directory:
```bash
cd label-studio-ml-backend/examples/openai/
```

3. Configure the OpenAI API Key:
Open the `docker-compose.yml` file and replace <your_openai_key> with your OpenAI API key. 
```yaml
OPENAI_API_KEY=<your_openai_key>
```

4. Start ML backend
With everything configured, we can start ML backend locally with Docker Compose. 
```bash
docker-compose up
```

5. Configure the model endpoint. 
Once all of the services have started, we will create a sentiment analysis project and [configure the model endpoint](https://labelstud.io/guide/ml.html#Add-an-ML-backend-using-the-Label-Studio-UI) in our Label Studio project. 
In Docker Compose, our model service can be accessed at `http://localhost:9090`.

6. You can now select tasks to retrieve predictions from the GPT models. 