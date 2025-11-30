# DocLayout-YOLO Models

This directory is intended to store the DocLayout-YOLO model files (`.pt`).

## Downloading Models

You need to download the pre-trained or fine-tuned DocLayout-YOLO models manually and place them in this directory.

Models can be downloaded from the official Hugging Face collection:
[https://huggingface.co/collections/juliozhao/doclayout-yolo-670cdec674913d9a6f77b542](https://huggingface.co/collections/juliozhao/doclayout-yolo-670cdec674913d9a6f77b542)

**Example:** Download `doclayout_yolo_docstructbench_imgsz1024.pt` and place it here.

## Configuration

The ML backend will look for models in this directory by default. You can specify which model to use via:

1.  **Environment Variable:** Set `MODEL_NAME` in `docker-compose.yml` (e.g., `MODEL_NAME=doclayout_yolo_docstructbench_imgsz1024.pt`).
2.  **Labeling Config:** Add the `model_path` attribute to your `<RectangleLabels>` tag in the Label Studio UI (e.g., `<RectangleLabels name="label" toName="image" model_path="your_custom_model.pt">`). This requires `ALLOW_CUSTOM_MODEL_PATH=true` in `docker-compose.yml`.

If `MODEL_NAME` is not set and `model_path` is not used in the config, the backend will try to load the default specified in `control_models/rectangle_labels.py` (currently `doclayout_yolo_docstructbench_imgsz1024.pt`).