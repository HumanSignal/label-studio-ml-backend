## Quickstart

1. Build and start Machine Learning backend on `http://localhost:9090`

```bash
docker-compose up -d
```

2. Check if it works:

```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

3. Then connect running backend to Label Studio:

```bash
label-studio start --init new_project --ml-backends http://localhost:9090
```

4. Open your project `new_project` and set this labeling config:
https://labelstud.io/templates/sentiment_analysis.html

```
<View>
  <Header value="Choose text sentiment:"/>
  <Text name="my_text" value="$reviewText"/>
  <Choices name="sentiment" toName="my_text" choice="single" showInline="true">
    <Choice value="Positive"/>
    <Choice value="Negative"/>
    <Choice value="Neutral"/>
  </Choices>
</View>
```

## Writing your own model
1. Place your scripts for model training & inference inside root directory. Follow the [API guidelines](#api-guidelines) described bellow. You can put everything in a single file, or create 2 separate one say `my_training_module.py` and `my_inference_module.py`

2. Write down your python dependencies in `requirements.txt`

3. Create ML backend with your model
```bash
label-studio-ml init my-ml-backend --script pytorch_transfer_learning/pytorch_transfer_learning.py
```

4. Set Label Studio HOSTNAME and API_KEY
```bash 
export LABEL_STUDIO_HOSTNAME=http://localhost:8080 
export LABEL_STUDIO_API_KEY=<ls-token>
``` 

4. Start ML backend at http://localhost:9090
```bash 
label-studio-ml start my-ml-backend
```

5. Start Label Studio with ML backend connection
```bash
label-studio start my-annotation-project --init --ml-backend http://localhost:9090
```
   
## API guidelines

Check out https://github.com/heartexlabs/label-studio-ml-backend/tree/master#Create_your_own_ML_backend

## License

This software is licensed under the [Apache 2.0 LICENSE](/LICENSE) © [Heartex](https://www.heartex.com/). 2022

<img src="https://github.com/heartexlabs/label-studio/blob/master/images/opossum_looking.png?raw=true" title="Hey everyone!" height="140" width="140" />
