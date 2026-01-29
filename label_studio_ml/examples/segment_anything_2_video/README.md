<!--
---
title: SAM2 with Videos
type: guide
tier: all
order: 15
hide_menu: true
hide_frontmatter_title: true
meta_title: Using SAM2 with Label Studio for Video Annotation
categories:
    - Computer Vision
    - Video Annotation
    - Object Detection
    - Segment Anything Model
image: "/tutorials/sam2-video.png"
---
-->

# Using SAM2 with Label Studio for Video Annotation

This guide describes the simplest way to start using **SegmentAnything 2** with Label Studio.

This repository is specifically for working with object tracking in videos. For working with images, 
see the [segment_anything_2_image repository](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image)

![sam2](./Sam2Video.gif)

## Before you begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart). 

This tutorial uses the [`segment_anything_2_video` example](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_video). 

## Running from source

1. To run the ML backend without Docker, you have to clone the repository and install all dependencies using pip:

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend
pip install -e .
cd label_studio_ml/examples/segment_anything_2_video
pip install -r requirements.txt
```

2. Download [`segment-anything-2` repo](https://github.com/facebookresearch/segment-anything-2) into the root directory. Install SegmentAnything model and download checkpoints using [the official Meta documentation](https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#installation). Make sure that you complete the steps for downloadingn the checkpoint files! 

3. Export the following environment variables (fill them in with your credentials!):
- LABEL_STUDIO_URL: the http:// or https:// link to your label studio instance (include the prefix!) 
- LABEL_STUDIO_API_KEY: your api key for label studio, available in your profile. 

4. Then you can start the ML backend on the default port `9090`:

```bash
cd ../
label-studio-ml start ./segment_anything_2_video
```
Note that if you're running in a cloud server, you'll need to run on an exposed port. To change the port, add `-p <port number>` to the end of the start command above.
5. Connect running ML backend server to Label Studio: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL. Read more in the official [Label Studio documentation](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio).
 Again, if you're running in the cloud, you'll need to replace this localhost location with whatever the external ip address is of your container, along with the exposed port.

# Labeling Config
For your project, you can use any labeling config with video properties. Here's a basic one to get you started!

```xml     
<View>
    <Labels name="videoLabels" toName="video" allowEmpty="true">
        <Label value="Player" background="#11A39E"/>
        <Label value="Ball" background="#D4380D"/>
    </Labels>

    <!-- Please specify FPS carefully, it will be used for all project videos -->
    <Video name="video" value="$video" framerate="25.0"/>
    <VideoRectangle name="box" toName="video" smart="true"/>
</View>
```

## CLI Usage

This directory contains several powerful CLI tools for video tracking automation. All commands should be run inside the running Docker container.

### 1. Automatic Object Detection & Tracking (`initial_seeding_video.py`)
**Use case:** You have a raw video and want to automatically find and track all objects of a certain class (e.g., "person", "car").
**Input:** Raw video task. No existing annotations or bounding boxes are required.
**Method:** Uses Grounding DINO to find objects at keyframes, then SAM2 to track them, and stitches the results into complete tracks.

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/initial_seeding_video.py \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <PROJECT_ID> \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID> \
  --prompt "person" \
  --keyframe-frac 0.1
'
```

### 2. Track Existing Manual Keyframes (`initial_seeding_video_boxes.py`)
**Use case:** You have already drawn some bounding boxes (keyframes) in Label Studio and want SAM2 to track them forward and backward to fill the gaps.
**Input:** Video task with at least a few manual bounding boxes (keyframes).
**Method:** Uses your existing boxes as anchors, generates bidirectional tracklets using SAM2, and uses the Hungarian algorithm to stitch them robustly. Supports processing long videos in parallel segments.

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/initial_seeding_video_boxes.py \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <PROJECT_ID> \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID> \
  --prompt "person" \
  --global-start 0 \
  --global-end 2000 \
  --max-frames-to-track 300
'
```
*   `--prompt`: (Optional) Label name to apply to the tracked objects (default: "object").
*   `--global-start` / `--global-end`: Process only this frame range (useful for parallel batching). **0-indexed**. Defaults to `0` and the last frame of the video if not specified.
*   `--max-frames-to-track`: How far to track from each keyframe in **each direction**. A value of 300 means it will track 300 frames forward AND 300 frames backward (total window ~600 frames).

### 2a. Manual Merge Tracking (`initial_seeding_video_boxes_manual_merge.py`)
**Use case:** You want bidirectional tracking per seed box, but you prefer to merge tracks manually using `mergevideoregions.py` and meta text IDs.
**Input:** Video task with manual keyframes. Optional track-id filtering lets you target specific seed regions per iteration.
**Method:** Builds forward+backward tracks per seed and keeps them in the same region; no automatic cross-seed merging. Each output region gets `meta.text="id:"` to ease manual ID assignment.

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/initial_seeding_video_boxes_manual_merge.py \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <PROJECT_ID> \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID> \
  --track-id auto-track-14,auto-track-15 \
  --global-start 1000 \
  --global-end 1800 \
  --max-frames-to-track 300 \
  --overlap-mode iou-weighted \
  --overlap-iou-threshold 0.3
'
```
*   `--track-id`: (Optional) Comma-separated list of region IDs to use as anchors. If omitted, all manual keyframes are used.
*   `--global-start` / `--global-end`: Same semantics as above (0-indexed, inclusive).
*   `--max-frames-to-track`: Tracks N frames backward and N frames forward from each seed.
*   `--overlap-mode`: Resolve overlaps within the same region (`iou-weighted`, `weighted`, `winner`). Default: `iou-weighted`.
*   `--overlap-iou-threshold`: IoU threshold for `iou-weighted` (default: 0.3).
*   `--overlap-mode` / `--overlap-iou-threshold` are optional; omit them to use the defaults shown above.

### 3. Simple Forward Tracking (`cli.py`)
**Use case:** Simple "predict" functionality similar to the UI button. Tracks from start to finish linearly.
**Input:** Video task with at least one manual keyframe (bounding box) to start tracking from.
**Method:** Loads the entire video and propagates all keyframes at once. Best for short clips.

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/cli.py \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <PROJECT_ID> \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID>
'
```

### 4. Post-Processing Tools (`video_tools.py`)
**Use case:** Clean up tracking results (sparsify dense frames, swap IDs, smooth jitter).

**Sparsify (Downsample Keyframes)**
Reduce the density of keyframes (e.g., keep 10% of frames) to make manual editing easier.
```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/video_tools.py sparsify \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID> \
  --track-id auto-track-0 \
  --start-frame 1000 \
  --end-frame 2000 \
  --ratio 0.1
'
```

**Swap IDs (Fix Identity Switches)**
Move a segment of tracking history from one object ID to another (e.g., if the tracker swapped "Person A" to "Person B").
```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/video_tools.py swap-ids \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID> \
  --source-track-id auto-track-0 \
  --target-track-id auto-track-1 \
  --start-frame 500 \
  --end-frame 600
'
```

**Trim Tail (Delete Trailing Frames)**
Delete all keyframes for a specific track after a certain cutoff frame (e.g., when an object leaves the view).
```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/video_tools.py trim-tail \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID> \
  --track-id auto-track-0 \
  --cutoff-frame 1500
'
```

**Smooth (Stabilize Jitter)**
Apply a moving average filter to smooth out shaky bounding boxes.
```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/video_tools.py smooth \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID> \
  --track-id auto-track-0 \
  --window 5
'
```

**Pad (Expand Bounding Boxes)**
Inflate bounding boxes by a percentage (e.g., 10%) over a specific frame range. Useful if the tracker is consistently too tight.
```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/video_tools.py pad \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID> \
  --track-id auto-track-0 \
  --percent 0.10 \
  --start-frame 0 \
  --end-frame 1000
'
```

### 5. Prediction Validation (`validate_prediction.py`)
**Use case:** Validate a prediction JSON file against your project configuration without uploading.

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/validate_prediction.py \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --task <TASK_ID> \
  --prediction-file prediction.json \
  --upload
'
```

### 6. Export Utilities (`export_interpolated_annotation.py`)
**Use case:** Download a single annotation JSON with *all* interpolated video frames included (not just keyframes). This is critical for getting the full frame-by-frame tracking data out of Label Studio.

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/export_interpolated_annotation.py \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <PROJECT_ID> \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID> \
  --output-dir /app/exports
'
```

**Alternative (Bash script):**
If you prefer a shell script (requires `curl`, `jq`, `unzip`):
```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

/app/export_interpolated_annotation.sh \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <PROJECT_ID> \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID> \
  --output /app/exports/annotation.json
'
```

### 7. Deletion Utilities (`delete_annotation_or_prediction.py`)
**Use case:** surgically delete a specific annotation or prediction by ID. Useful for cleanup scripts or resetting a task state.

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

# Delete an annotation
python /app/delete_annotation_or_prediction.py \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <PROJECT_ID> \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID>

# OR delete a prediction
python /app/delete_annotation_or_prediction.py \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <PROJECT_ID> \
  --task <TASK_ID> \
  --prediction <PREDICTION_ID>
'
```

### 8. Merge Video Regions (`mergevideoregions.py`)
**Use case:** Consolidate fragmented tracks that share the same text ID (e.g., `id:31` in `meta.text`) into single continuous track objects. Useful after manual labeling or ReID where multiple regions represent the same object.

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/mergevideoregions.py \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <PROJECT_ID> \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID>
'
```

### 9. Bounding Box Refinement (`adjust_bboxes_sam2.py`)
**Use case:** Tighten or adjust existing bounding boxes using SAM2's segmentation capabilities. It uses the box as a prompt, generates a mask, and replaces the box with the mask's bounding box.

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/adjust_bboxes_sam2.py \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <PROJECT_ID> \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID> \
  --search-scale 1.2
'
```
*   `--search-scale`: How much to expand the search region around the original box (default 1.2 = 20% larger).

### 10. Re-Identification (`complete_reid.py`)
**Use case:** Automatically suggest identity matches for broken tracks. Uses appearance features (color, geometry, or SAM2 embeddings) to find likely matches between "candidate" tracks (no ID) and "reference" tracks (confirmed ID).

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST="https://app.heartex.com"
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/complete_reid.py \
  --ls-url "$LABEL_STUDIO_HOST" \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project <PROJECT_ID> \
  --task <TASK_ID> \
  --annotation <ANNOTATION_ID> \
  --profile uav \
  --feature-backend sam2
'
```
*   `--profile`: Preset for weighting features (`uav`, `ugv`).
*   `--feature-backend`: `classic` (color/geometry histograms) or `sam2` (neural embeddings).

## Configuration

### Core Environment Variables
- `LABEL_STUDIO_HOST` / `LABEL_STUDIO_URL`: The URL of your Label Studio instance (e.g., `https://app.heartex.com`).
- `LABEL_STUDIO_API_KEY`: Your Label Studio API key.
- `DEVICE`: Computing device (recommended: `cuda`).
- `MODEL_CONFIG`: SAM2 model config path (default: `configs/sam2.1/sam2.1_hiera_l.yaml`).
- `MODEL_CHECKPOINT`: SAM2 checkpoint filename (default: `sam2.1_hiera_large.pt`).

### Tracking Performance & Memory
- `MAX_FRAMES_TO_TRACK`: Hard limit for how many frames to track from the first keyframe (default: `1000`).
  - **Note**: This environment variable controls the limit for the serving model (`model.py` / `cli.py`). The `initial_seeding_video_boxes.py` script uses its own `--max-frames-to-track` CLI argument (default `300`).
  - Set to `0` for no limit (tracks to end of video).
  - **Warning**: High values (e.g. >2000) require significant RAM/VRAM.
- `TRACK_FPS`: Temporal downsampling target FPS (default: `0` = disabled).
  - Example: Set `TRACK_FPS=5` to track only 5 frames per second, reducing memory usage and processing time.

### Server Configuration (Gunicorn/Docker)
- `PORT`: Port to listen on (default: `9090`).
- `WORKERS`: Number of worker processes (default: `1`).
  - **Note**: SAM2 is memory-intensive. Increasing workers increases RAM usage linearly.
- `THREADS`: Threads per worker (default: `4`; `docker-compose.yml` sets this to `8`).
- `LOG_LEVEL`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- **Docker `shm_size`**: SAM2 requires shared memory. Ensure your `docker-compose.yml` sets `shm_size: '32g'` or higher.
- **Docker `mem_limit`**: Recommended to set a memory limit (e.g., `mem_limit: 48g`) to avoid system instability.

### Advanced Tool Configuration

**Re-Identification (`complete_reid.py`):**
- `REID_PROFILE`: Preset profile (`uav` or `ugv`). Default: `uav`.
- `REID_FEATURE_BACKEND`: Feature extractor (`classic` or `sam2`). Default: `classic`.

**Initial seeding (`initial_seeding_video.py`):**
- `GROUNDINGDINO_REPO_PATH`: Path to Grounding DINO repo (default: `/GroundingDINO`).
- `GROUNDING_DINO_CONFIG`: Config filename (default: `GroundingDINO_SwinT_OGC.py`).
- `GROUNDING_DINO_WEIGHTS`: Weights filename (default: `gdino_swint_darpa-ir-v1-1k_20_1.pth`).
- `GROUNDING_DINO_DEVICE`: Device for Grounding DINO inference (default: `cuda` if available, else `cpu`).
- `GROUNDING_DINO_PROMPT` or `GROUNDING_DINO_LABELS`: Class prompt for detection.
- `GROUNDING_DINO_BOX_THRESHOLD` (default 0.35) / `GROUNDING_DINO_TEXT_THRESHOLD` (default 0.25).
- `GROUNDING_DINO_NMS_IOU`: NMS IoU threshold (default `0.5`).
- `GROUNDING_DINO_BASE_SIZE`: Input image resize short side (default `800`).
- `GROUNDING_DINO_MAX_SIZE`: Input image resize long side (default `1333`).
- `CACHE_DIR`: Joblib cache directory for SAM2 embeddings.
- `EMBED_BATCH`: Embedding batch size (default: `8`).
- `MAX_SEGMENT_FRAMES`: Max frames per segment for seeding (default: `1024`).
- `FRAME_JPEG_QUALITY`: Quality of extracted frames (default: `95`).
- `SPARSE_SEQUENCE`: Set to `true` to enable sparse sequence generation (can be overridden by CLI flags).

**Stitching & Sparsification Tuning (`initial_seeding_video.py` only):**
- `SPARSE_IOU_THRESH`: IoU threshold for removing redundant frames (default: `0.2`).
- `SPARSE_MAX_INTERVAL`: Max frame interval for sparsification (default: `0` = disabled).
- `STITCH_REQUIRE_VISIBLE_AT_END`: Require object to be visible at segment end to match (default: `true`).
- `STITCH_IOU_MIN`: Min IoU to consider matching tracks (default: `0.3`).
- `STITCH_DIST_MAX`: Max normalized distance for matching (default: `0.15`).
- `STITCH_AREA_RATIO_MAX`: Max area ratio difference (default: `2.5`).
- `STITCH_MAX_COST`: Max total cost for Hungarian matching (default: `1.2`).
- `STITCH_W_IOU` / `STITCH_W_DIST` / `STITCH_W_SIZE`: Weights for cost calculation (defaults: `1.0`, `1.0`, `0.2`).

## Known limitations
- SAM2 is designed to run on GPU servers; CPU execution is not recommended for practical video workloads.
- Currently, we do not support video segmentation (only bounding boxes).
- For very long videos (40,000+ frames), tracking may take significant time. Consider using `--max-frames` to process in chunks.

If you want to contribute to this repository to help with some of these limitations, you can submit a PR.

## Customization

The ML backend can be customized by adding your own models and logic inside the `./segment_anything_2_video` directory. 
