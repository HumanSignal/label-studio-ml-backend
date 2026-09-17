# Label Studio ML Backend for DocLayout-YOLO

This directory contains an example ML backend that integrates the [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) model for document layout analysis into Label Studio.

It uses the `doclayout-yolo` package (based on YOLOv10) to perform object detection on document images and return bounding box predictions for layout elements like text, titles, tables, figures, etc.

## Features

-   Loads DocLayout-YOLO models (`.pt` files).
-   Performs inference on image tasks.
-   Returns `RectangleLabels` predictions compatible with Label Studio.
-   Configurable model path, confidence threshold, and image size via environment variables or Label Studio labeling config.
-   Includes Docker setup for easy deployment.

## Prerequisites

-   Docker and Docker Compose installed.
-   Label Studio installed and running.
-   A trained/downloaded DocLayout-YOLO `.pt` model file.

## Setup

1.  **Download Model:**
    -   Download your desired DocLayout-YOLO model (e.g., `doclayout_yolo_docstructbench_imgsz1024.pt`) from the [official Hugging Face collection](https://huggingface.co/collections/juliozhao/doclayout-yolo-670cdec674913d9a6f77b542).
    -   Place the downloaded `.pt` file inside the `models/` directory within this example (`label_studio_ml/examples/doclayout_yolo/models/`).

2.  **Configure `docker-compose.yml`:**
    -   Open `docker-compose.yml`.
    -   **Crucially**, set the `LABEL_STUDIO_API_KEY` environment variable to your Label Studio API key (find it on your Label Studio Account page).
    -   Adjust `LABEL_STUDIO_URL` if your Label Studio instance is not running on `http://localhost:8080`. Use `http://host.docker.internal:8080` if Label Studio runs on the same machine as Docker.
    -   Verify the `MODEL_NAME` environment variable matches the filename of the model you placed in the `models/` directory.
    -   Adjust `MODEL_SCORE_THRESHOLD` and `DEFAULT_IMGSZ` if needed.

## Running the Backend

1.  **Build and Start:**
    Navigate to the `label_studio_ml/examples/doclayout_yolo/` directory in your terminal and run:
    ```bash
    docker-compose up --build -d
    ```
    This will build the Docker image and start the ML backend container in the background.

2.  **Check Logs (Optional):**
    ```bash
    docker-compose logs -f doclayout-yolo
    ```
    Look for messages indicating the server has started, typically on port 9090.

## Connecting to Label Studio

1.  Open Label Studio.
2.  Go to your project's Settings > Machine Learning.
3.  Click "Add Model".
4.  Enter a Title (e.g., "DocLayout-YOLO Backend").
5.  Enter the URL: `http://<your-machine-ip-or-hostname>:9090` (use the IP address of the machine running Docker, not localhost or 127.0.0.1 unless LS runs in the *same* Docker network).
6.  Enable the "Use for interactive pre-annotations" toggle.
7.  Click "Validate and Save".

## Labeling Configuration

Use a Label Studio labeling configuration that includes an `Image` object tag and a `RectangleLabels` control tag. Ensure the labels within `<RectangleLabels>` match the classes your DocLayout-YOLO model predicts.

**Example:**

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <!-- Add labels corresponding to your DocLayout-YOLO model's classes -->
    <!-- Example labels from DocStructBench -->
    <Label value="paragraph" background="#FFA39E"/>
    <Label value="title" background="#FFD59E"/>
    <Label value="list_item" background="#FFFFB0"/>
    <Label value="figure" background="#B5FF9E"/>
    <Label value="table" background="#9EEAFF"/>
    <Label value="page_header" background="#A89EFF"/>
    <Label value="page_footer" background="#FFA8FF"/>
    <Label value="section_header" background="#CCCCCC"/>
    <Label value="equation" background="#E6E6E6"/>
    <Label value="figure_caption" background="#D9D9D9"/>
    <Label value="table_caption" background="#C0C0C0"/>
    <Label value="reference" background="#A6A6A6"/>
    <Label value="footnote" background="#8C8C8C"/>
  </RectangleLabels>
</View>
```
**Example (with model selection):**

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
  <RectangleLabels name="label" toName="image"
                   model_path="<your-model-path>/doclayout_yolo_docstructbench_imgsz1024.pt"
                   model_score_threshold="0.2"
                   model_imgsz="1600">
    <!-- Add labels corresponding to your DocLayout-YOLO model's classes -->
    <!-- Example labels from DocStructBench -->
    <Label value="title" background="#1cffb5"/>
    <Label value="plain text" background="#c4c400"/>
    <Label value="abandon" background="#B5FF9E"/>
    <Label value="figure" background="#9EEAFF"/>
    <Label value="figure_caption" background="#3c99f7"/>
    <Label value="table" background="#FFA8FF"/>
    <Label value="table_caption" background="#c6a2f7"/>
    <Label value="table_footnote" background="#e94e03"/>
    <Label value="isolate_formula" background="#fd921c"/>
    <Label value="formula_caption" background="#fccd41"/>
  </RectangleLabels>
</View>
```

