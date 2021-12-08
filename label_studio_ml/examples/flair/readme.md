

## Named Entity Recognition with Flair based embeddings
Flair is a lightweight NLP task library with some cutting-edge methods and good documentation and tutorials to get started: https://github.com/flairNLP/flair

If you want to train a custom Named Entity Recognition model using high performance embeddings of Flair library and use active learning, this example is for you!
This example uses embeddings for the English language, but it's easy to change the embeddings to a different language.

* If you want to use Flairs unique embeddings for different languages (or multi-languages) check [HERE!](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md).\
Adapt the line of code in the fit function of this example [HERE!](https://github.com/AaronDeRybelHowest/label-studio-ml-backend/blob/ab0c926f1513200b60ae48d4be6b718aad9e31be/label_studio_ml/examples/flair/ner_ml_backend.py#L117)
* If you want to use different types of embeddings or experiment with different language models for making embeddings check [HERE!](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md)


## Start using it
1. Install ML backend:
    ```bash
    pip install -r label_studio_ml/examples/flair/requirements.txt
    label-studio-ml init my-ml-backend --from label_studio_ml/examples/flair/ner_ml_backend.py
    label-studio-ml start my-ml-backend -d -p=9090
    ```
It's recommended to start ml-backend in debug mode, printouts are being made.

2. Start Label Studio and create a new project.
   
3. In the project **Settings**, set up the **Labeling Interface**.
   Fill in the following template code, important to specifiy label name as `ner_tags`
```
<View>
  <Labels name="ner_tags" toName="text">
    <Label value="label_1" background="#FFA39E"/>
    <Label value="label_2" background="#D4380D"/>
    <Label value="label_3" background="#FFC069"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
```

4. Open the **Machine Learning** settings and click **Add Model**. 

6. Add the URL `http://localhost:9090` and save the model as an ML backend.

7. Import around 20 sentences minimum and annotate them in label-studio

extra: The prediction function returns certaincy scores of the model between 0.0 and 1.0. If there are **NO entities** predicted in a sentence a score of 2.0 is returned. This way you can filter out sentences without entities while labeling or simply remove them from your dataset using the label-studio interface.\
Make Sure to add the prediction Score column and use filter and sort functionality (see screenshot)

![image](https://user-images.githubusercontent.com/43145159/143429476-e81e6986-8309-490b-91d1-e6c47a1911c1.png)

![image](https://user-images.githubusercontent.com/43145159/143429611-229c6357-d83e-4e6b-a416-b101d017174f.png)

