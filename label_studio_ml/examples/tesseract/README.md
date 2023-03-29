

## Interactive BBOX OCR using Tesseract
Using an OCR engine for Interactive ML-Assisted Labelling, this functionality can speed up annotation for layout detection, classification and recognition models.

Tesseract is used for OCR but minimal adaptation is needed to connect other OCR engines or models.

Tested with LabelStudio v1.4.1.post1, and assuming data for annotation is stored in AWS S3 (some adaptation is needed if using other storage methods).

### Setup process
0. Install label-studio-ml and Tesseract

1. Start LabelStudio and create a new project

2. In the project **Settings**, set up the **Cloud storage**. Add your Source and Target storage by connecting to AWS S3 Bucket

3. In the project **Settings**, set up the **Labeling Interface**
   Fill in the following template code, important to specifiy `smart="true"` in RectangleLabels
```
<View>    
   <Image name="image" value="$ocr" zoom="true" zoomControl="false" rotateControl="true" width="100%" height="100%" maxHeight="auto" maxWidth="auto"/>
   
   <RectangleLabels name="bbox" toName="image" strokeWidth="1" smart="true">
      <Label value="Label1" background="green"/>
      <Label value="Label2" background="blue"/>
      <Label value="Label3" background="red"/>
   </RectangleLabels>

   <TextArea name="transcription" toName="image" 
     editable="true" perRegion="true" required="false" 
     maxSubmissions="1" rows="5" placeholder="Recognized Text" 
     displayMode="region-list"/>
</View>
```

4. Setup Tesseract ML backend:
    ```
    pip install -r label_studio_ml/examples/tesseract/requirements.txt
    label-studio-ml init my-ml-backend --from label_studio_ml/examples/tesseract/tesseract.py --force
    label-studio-ml start my-ml-backend -d -p=9090 --debug
    ```
    
5. Open the **Machine Learning** settings and click **Add Model**. Add the URL `http://localhost:9090` and save the model as an ML backend.

6. To use this functionality, activate `Auto-Annotation` and use `Autotdetect` rectangle for drawing boxes

Exemple below :

![ls_demo_ocr](https://user-images.githubusercontent.com/17755198/165186574-05f0236f-a5f2-4179-ac90-ef11123927bc.gif)

Reference links : 
- https://labelstud.io/blog/Improve-OCR-quality-with-Tesseract-and-Label-Studio.html
- https://labelstud.io/blog/release-130.html
