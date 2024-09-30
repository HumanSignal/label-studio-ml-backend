## Interactive BBOX OCR using PaddleOCR
Using an OCR engine for Interactive ML-Assisted Labelling, this functionality
can speed up annotation for layout detection, classification and recognition
models.

PaddleOCR is used for OCR but minimal adaptation is needed to connect other OCR
engines or models.

PaddleOCR supports 80 languages. refer to https://github.com/Mushroomcat9998/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md#language_abbreviations

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

2. Create a new project for PaddleOCR. In the project **Settings** set up the **Labeling Interface**.

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
   cd label-studio-ml-backend/label_studio_ml/examples/paddleocr
   ```

4. Configure the backend and the Minio server by editing the `example.env` file. If you opted to use Label Studio
   Local File Storage, be sure to set the `LABEL_STUDIO_HOST` and `LABEL_STUDIO_ACCESS_TOKEN` variables. If you're
   using the Minio storage example, set the `MINIO_ROOT_USER` AND `MINIO_ROOT_PASSWORD` variables, and make the 
   `AWS_ACCESS_KEY_ID` AND `AWS_SECRET_ACCESS_KEY` variables equal to those values. You may optionally connect to your
   own AWS cloud storage by setting those variables. Note that you may need to make additional software changes to the
   `paddleocr_ch.py` file to match your particular infrastructure configuration.

   ```
   LABEL_STUDIO_HOST=http://host.docker.internal:8080
   LABEL_STUDIO_ACCESS_TOKEN=<optional token for local file access>

   AWS_ACCESS_KEY_ID=<set to MINIO_ROOT_USER for minio example>
   AWS_SECRET_ACCESS_KEY=<set to MINIO_ROOT_PASSWORD for minio example>
   AWS_ENDPOINT=http://host.docker.internal:9000

   MINIO_ROOT_USER=<username>
   MINIO_ROOT_PASSWORD=<password>
   MINIO_API_CORS_ALLOW_ORIGIN=*

   OCR_LANGUAGE=<Language Abbreviation ch,en,fr,japan> # support 80 languages. refer to https://github.com/Mushroomcat9998/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md#language_abbreviations
   ```

5. Start the PaddleOCR and minio servers.

   ```
   # build image
   sudo docker build . -t paddleocr-backend:latest
   # or
   sudo docker compose build .

   # only start paddleocr-backend
   sudo docker compose up paddleocr-backend
   # or and start with minio
   docker compose up -d
   # shutdown container
   docker compose down 


   # or docker pull image from docker-hub
   docker pull blazordevlab/paddleocr-backend:latest
   
   ```
   below is my docker-compose file include label-studio,minio and paddleocr-backend

   ```
   version: "3.9"

   x-logging:
     logging: &default-logging
       driver: "local"
       options:
         max-size: "10m"
         max-file: "3"

   services:
     label-studio:
       container_name: label-studio
       image: heartexlabs/label-studio:latest
       restart: unless-stopped
       ports:
         - "8080:8080"
       depends_on:
         - minio
       environment:
         - JSON_LOG=1
         - LOG_LEVEL=DEBUG
       volumes:
         - label-studio-data:/label-studio/data

     # not replicated setup for test setup, use a proper aws S3 compatible cluster in production
     minio:
       container_name: minio
       image: bitnami/minio:latest
       restart: unless-stopped
       logging: *default-logging
       ports:
         - "9000:9000"
         - "9001:9001"
       volumes:
         - minio-data:/data
         - minio-certs:/certs
       # configure env vars in .env file or your systems environment
       environment:
         - MINIO_ROOT_USER=${MINIO_ROOT_USER:-minio_admin_do_not_use_in_production}
         - MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD:-minio_admin_do_not_use_in_production}
         - MINIO_PROMETHEUS_AUTH_TYPE=${MINIO_PROMETHEUS_AUTH_TYPE:-public}
     paddleocr-backend:
       container_name: paddleocr-backend
       image: blazordevlab/paddleocr-backend:latest
       environment:
         - LABEL_STUDIO_HOST=${LABEL_STUDIO_HOST:-http://label-studio:8080}
         - LABEL_STUDIO_ACCESS_TOKEN=${LABEL_STUDIO_ACCESS_TOKEN}
         - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
         - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
         - AWS_ENDPOINT=${AWS_ENDPOINT:-http://minio:9000}
         - MINIO_ROOT_USER=${MINIO_ROOT_USER:-minio_admin_do_not_use_in_production}
         - MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD:-minio_admin_do_not_use_in_production}
         - MINIO_API_CORS_ALLOW_ORIGIN=${MINIO_API_CORS_ALLOW_ORIGIN:-*}
         - OCR_LANGUAGE=${OCR_LANGUAGE:-ch}
       ports:
         - 9090:9090
       volumes:
         - paddleocr-backend-data:/data
         - paddleocr-backend-logs:/tmp
   volumes:
     label-studio-data:
     minio-data:
     minio-certs:
     paddleocr-backend-data:
     paddleocr-backend-logs:
   ```

7. Upload tasks.

   If you're using the Label Studio Local File Storage option, upload images
   directly to Label Studio using the Label Studio interface.

   If you're using minio for task storage, log into the minio control panel at
   `http://localhost:9001`. Create a new bucket, making a note of the name, and
   upload your tasks to minio. Set the visibility of the tasks to be public.
   Furtner configuration of your cloud storage is beyond the scope of this
   tutorial, and you will want to configure your storage according to your
   particular needs. 

8. If using minio, In the project **Settings**, set up the **Cloud storage**.

   Add your source S3 storage by connecting to the S3 Endpoint
   `http://host.docker.internal:9000`, using the bucket name from the previous
   step, and Access Key ID and Secret Access Key as configured in the previous
   steps. For the minio example, uncheck **Use pre-signed URLS**. Check the
   connection and save the storage.

9. Open the **Machine Learning** settings and click **Add Model**.

   Add the URL `http://host.docker.internal:9090` and save the model as an ML backend.

10. To use this functionality, activate `Auto-Annotation` and use `Autotdetect` rectangle for drawing boxes

Example below :

![ls_demo_ocr](https://github.com/HumanSignal/label-studio-ml-backend/assets/1549611/fcc44c8b-12fd-495c-b0b4-9d5c0ceb2ed2)

Reference links : 
- https://labelstud.io/blog/Improve-OCR-quality-with-Tesseract-and-Label-Studio.html
- https://labelstud.io/blog/release-130.html
