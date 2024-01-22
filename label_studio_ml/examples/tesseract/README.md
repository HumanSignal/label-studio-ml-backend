## Interactive BBOX OCR using Tesseract
Using an OCR engine for Interactive ML-Assisted Labelling, this functionality
can speed up annotation for layout detection, classification and recognition
models.

Tesseract is used for OCR but minimal adaptation is needed to connect other OCR
engines or models.

Tested againt Label Studio 1.10.1, with basic support for both Label Studio
Local File Storage and S3-compatible storage, with a example data storage with
Minio.

### Setup process
0. Download and install Docker with Docker Compose. For MacOS and Windows users,
   we suggest using Docker Desktop. You will also need to have git installed.

1. Launch LabelStudio.

   ```
   docker run -it \
      -p 8080:8080 \
      -v `pwd`/mydata:/label-studio/data \
      heartexlabs/label-studio:latest
   ```

   Optionally, you may enable local file serving in Label Studio

   ```
   docker run -it \
      -p 8080:8080 \
      -v `pwd`/mydata:/label-studio/data \
      --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
      --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data/images \
      heartexlabs/label-studio:latest
   ```
   If you're using local file serving, be sure to get a copy of the API token from
   Label Studio to connect the model.

2. Create a new project for Tesseract OCR. In the project **Settings** set up the **Labeling Interface**.

   Fill in the following template code. It's important to specify `smart="true"` in RectangleLabels.
   ```
   <View>    
      <Image name="image" value="$ocr" zoom="true" zoomControl="false"
            rotateControl="true" width="100%" height="100%"
            maxHeight="auto" maxWidth="auto"/>
      
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

3. Download the Label Studio Machine Learning backend backend repository.
   ```
   git clone https://github.com/humansignal/label-studio-ml-backend
   cd label-studio-ml-backend/label_studio_ml/examples/tesseract
   ```

4. Configure the backend and the Minio server by editing the `example.env` file. If you opted to use Label Studio
   Local File Storage, be sure to set the `LABEL_STUDIO_HOST` and `LABEL_STUDIO_ACCESS_TOKEN` variables. If you're
   using the Minio storage example, set the `MINIO_ROOT_USER` AND `MINIO_ROOT_PASSWORD` variables, and make the 
   `AWS_ACCESS_KEY_ID` AND `AWS_SECRET_ACCESS_KEY` variables equal to those values. You may optionally connect to your
   own AWS cloud storage by setting those variables. Note that you may need to make additional software changes to the
   `tesseract.py` file to match your particular infrastructure configuration.

   ```
   LABEL_STUDIO_HOST=http://host.docker.internal:8080
   LABEL_STUDIO_ACCESS_TOKEN=<optional token for local file access>

   AWS_ACCESS_KEY_ID=<set to MINIO_ROOT_USER for minio example>
   AWS_SECRET_ACCESS_KEY=<set to MINIO_ROOT_PASSWORD for minio example>
   AWS_ENDPOINT=http://host.docker.internal:9000

   MINIO_ROOT_USER=<username>
   MINIO_ROOT_PASSWORD=<password>
   MINIO_API_CORS_ALLOW_ORIGIN=*
   ```

5. Start the Tesseract and minio servers.

   ```
   docker compose up
   ```

6. Upload tasks.

   If you're using the Label Studio Local File Storage option, upload images
   directly to Label Studio using the Label Studio interface.

   If you're using minio for task storage, log into the minio control panel at
   `http://localhost:9001`. Create a new bucket, making a note of the name, and
   upload your tasks to minio. Set the visibility of the tasks to be public.
   Furtner configuration of your cloud storage is beyond the scope of this
   tutorial, and you will want to configure your storage according to your
   particular needs. 

7. If using minio, In the project **Settings**, set up the **Cloud storage**.

   Add your source S3 storage by connecting to the S3 Endpoint
   `http://host.docker.internal:9000`, using the bucket name from the previous
   step, and Access Key ID and Secret Access Key as configured in the previous
   steps. For the minio example, uncheck **Use pre-signed URLS**. Check the
   connection and save the storage.

8. Open the **Machine Learning** settings and click **Add Model**.

   Add the URL `http://host.docker.internal:9090` and save the model as an ML backend.

9. To use this functionality, activate `Auto-Annotation` and use `Autotdetect` rectangle for drawing boxes

Example below :

![ls_demo_ocr](https://user-images.githubusercontent.com/17755198/165186574-05f0236f-a5f2-4179-ac90-ef11123927bc.gif)

Reference links : 
- https://labelstud.io/blog/Improve-OCR-quality-with-Tesseract-and-Label-Studio.html
- https://labelstud.io/blog/release-130.html
