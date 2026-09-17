# D-FINE ML Backend for Label Studio

This ML backend integrates the [D-FINE](https://github.com/Peterande/D-FINE) object detection model with Label Studio. It allows you to get pre-annotations for object detection tasks using pre-trained D-FINE models.

## Features

-   Loads pre-trained D-FINE models (e.g., COCO-trained models).
-   Provides bounding box predictions for `RectangleLabels` in Label Studio.
-   Configurable via environment variables for model paths, device, and thresholds.

## Prerequisites

1.  **Docker and Docker Compose**: For building and running the ML backend.
2.  **D-FINE Model Files**:
    *   **Source Code**: You need the `src` and `configs` directories from the [official D-FINE repository](https://github.com/Peterande/D-FINE).
    *   **Model Weights**: Download the desired `.pth` model weights (e.g., `dfine_l_coco.pth`).
3.  **Label Studio**: A running instance of Label Studio.

## Setup

1.  **Clone this repository** (if you haven't already) and navigate to this example directory:
    ```bash
    # Assuming you are in the root of label-studio-ml-backend
    cd label_studio_ml/examples/d_fine
    ```

2.  **Prepare D-FINE code**:
    *   Create a directory named `d-fine-code` within the current `label_studio_ml/examples/d_fine` directory.
    *   Copy the `src` and `configs` directories from your clone of the [D-FINE repository](https://github.com/Peterande/D-FINE) into this newly created `d-fine-code` directory.
    Your structure should look like:
    ```
    label_studio_ml/examples/d_fine/
    ├── d-fine-code/
    │   ├── src/
    │   └── configs/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── model.py
    └── ... (other files in this example)
    ```

3.  **Prepare D-FINE model weights**:
    *   Create a directory named `models` within the current `label_studio_ml/examples/d_fine` directory.
    *   Place your downloaded D-FINE `.pth` model weights file (e.g., `dfine_l_coco.pth`) into this `models` directory.
    Your structure should look like:
    ```
    label_studio_ml/examples/d_fine/
    ├── models/
    │   └── dfine_l_coco.pth  (or your chosen model weights)
    └── ... (other files)
    ```

4.  **Configure `docker-compose.yml`**:
    *   Adjust environment variables as needed, especially:
        *   `DFINE_CONFIG_FILE`: Name of the D-FINE `.yml` config file (e.g., `dfine_hgnetv2_l_coco.yml`). This file must exist in `d-fine-code/configs/dfine/`.
        *   `DFINE_MODEL_WEIGHTS`: Name of the D-FINE `.pth` weights file (e.g., `dfine_l_coco.pth`). This file must exist in the `models` directory you created.
        *   `DEVICE`: Set to `cuda` if you have a GPU and want to use it, otherwise `cpu`.
        *   `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY` (if Label Studio needs to serve image data to the backend, e.g., for local file uploads or cloud storage not directly accessible by the backend).

## Running the ML Backend

1.  **Build and start the Docker container**:
    ```bash
    docker-compose up --build
    ```
    If you have a GPU and configured it in `docker-compose.yml`, it should be utilized.

2.  **Verify the backend is running**:
    Open your browser or use `curl` to check the health endpoint:
    ```bash
    curl http://localhost:9090/health
    ```
    You should see `{"status":"UP","model_class":"DFINEModel"}`.

## Connecting to Label Studio

1.  Open your Label Studio project.
2.  Go to **Settings > Machine Learning**.
3.  Click **Add Model**.
4.  Enter a **Title** for your ML backend (e.g., "D-FINE Detector").
5.  Set the **URL** to `http://localhost:9090` (or the appropriate host/port if not running locally or on a different port).
6.  Enable **Interactive preannotations** if desired.
7.  Click **Validate and Save**.

## Labeling Configuration

This ML backend expects a labeling configuration with an `Image` object tag and a `RectangleLabels` control tag.

Example:
```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image" model_score_threshold="0.3">
    <!-- Map Label Studio labels to D-FINE model's COCO class names -->
    <!-- D-FINE outputs COCO class names like 'person', 'car', etc. -->
    <Label value="Pedestrian" background="green" predicted_values="person"/>
    <Label value="Vehicle" background="blue" predicted_values="car,truck,bus,motorcycle"/>
    <Label value="Bicycle" background="orange" predicted_values="bicycle"/>
    <!-- Add more labels as needed, mapping to COCO_CLASSES -->
  </RectangleLabels>
</View>