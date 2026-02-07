<!--
---
title: SAM3 with Videos
type: guide
tier: all
order: 15
hide_menu: true
hide_frontmatter_title: true
meta_title: Using SAM3 with Label Studio for Video Annotation
categories:
    - Computer Vision
    - Video Annotation
    - Object Detection
    - Segment Anything Model
image: "/tutorials/sam2-video.png"
---
-->

# Using SAM3 with Label Studio for Video Annotation

This guide describes how to use **Segment Anything 3** (SAM3) with Label Studio for video object tracking.

SAM3 is loaded via HuggingFace Transformers (`from_pretrained`), replacing the previous SAM2 standalone repo + custom CUDA ops + Grounding DINO stack. Video decoding uses PyAV (no disk-based frame extraction).

This repository is specifically for working with object tracking in videos. For working with images,
see the [segment_anything_2_image repository](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/segment_anything_2_image).

## What changed from SAM2

| Area | SAM2 | SAM3 |
|------|------|------|
| Model source | Meta repo clone + custom CUDA build | `transformers.from_pretrained()` |
| Object detection | Grounding DINO (separate repo + weights) | `Sam3VideoModel` with text prompts (`HINTS=true`) |
| Instance tracking | `build_sam2_video_predictor` + keypoints hack | `Sam3TrackerVideoModel` with native box prompts |
| Video decoding | OpenCV `cv2.VideoCapture` + JPEG extraction to disk | PyAV in-memory streaming |
| Dockerfile | 104 lines, `devel` base, CUDA compilation | 19 lines, `runtime` base |
| Prompt format | 5 synthetic keypoints from bbox | Native xyxy bounding boxes |
| Processing modes | Single (all frames to disk) | `streaming` (constant memory) or `chunked_batch` (bidirectional context) |

## Before you begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart).

This tutorial uses the `segment_anything_3_video` example.

## Running with Docker (recommended)

```bash
cd label_studio_ml/examples/segment_anything_3_video

# Set your credentials
export LABEL_STUDIO_API_KEY="your-api-key"
export HF_TOKEN="your-huggingface-token"  # if model is gated

# Build and start
docker compose up --build
```

The model weights are downloaded automatically on first startup via `from_pretrained()`.

## Running from source

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend
pip install -e .
cd label_studio_ml/examples/segment_anything_3_video
pip install -r requirements.txt
```

No separate model repo or checkpoint download is required. Weights are fetched automatically by HuggingFace Transformers on first import.

2. Export environment variables:

```bash
export LABEL_STUDIO_URL="https://your-label-studio-instance.com"
export LABEL_STUDIO_API_KEY="your-api-key"
export MODEL_NAME="facebook/sam3"
export HINTS=false           # or true for text-based detection
export PROCESSING_MODE=streaming  # or chunked_batch
```

3. Start the ML backend:

```bash
cd ../
label-studio-ml start ./segment_anything_3_video
```

4. Connect the running ML backend to Label Studio: go to your project **Settings > Machine Learning > Add Model** and specify `http://localhost:9090` as the URL.

## Labeling Config

For your project, you can use any labeling config with video properties. Here's a basic one to get you started:

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

## Model Variants

### Tracker mode (`HINTS=false`, default)

Uses `Sam3TrackerVideoModel` + `Sam3TrackerVideoProcessor`. Requires user-drawn bounding boxes in Label Studio as tracking prompts. Best for interactive annotation where you draw a box on the first frame and the model tracks it forward.

### PCS / hints mode (`HINTS=true`)

Uses `Sam3VideoModel` + `Sam3VideoProcessor`. Replaces Grounding DINO with SAM3's built-in text-based detection. Set `PROMPT_TEXT` to specify what to detect (e.g., `person`, `car`). No drawn bounding boxes required.

## Processing Modes

### Streaming (`PROCESSING_MODE=streaming`, default)

Decodes frames one at a time via PyAV. Constant memory usage regardless of video length. Best for long videos and production use.

### Chunked batch (`PROCESSING_MODE=chunked_batch`)

Decodes all frames in `[start_frame, end_frame]` into memory at once. Provides bidirectional temporal context for better tracking quality. Use for shorter clips or when you have ample GPU memory.

## Configuration

### Core Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Computing device (`cuda` or `cpu`) |
| `MODEL_NAME` | `facebook/sam3` | HuggingFace model identifier |
| `HINTS` | `false` | `false` = Tracker (box prompts), `true` = PCS (text detection) |
| `PROCESSING_MODE` | `streaming` | `streaming` or `chunked_batch` |
| `PROMPT_TEXT` | `person` | Text prompt for PCS mode (`HINTS=true`) |
| `HF_TOKEN` | (empty) | HuggingFace token for gated models |
| `LABEL_STUDIO_HOST` | | URL of your Label Studio instance |
| `LABEL_STUDIO_API_KEY` | | Your Label Studio API key |

### Tracking Performance & Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FRAMES_TO_TRACK` | `1000` | Hard limit for frames to track from first keyframe. `0` = no limit |
| `TRACK_FPS` | `0` | Temporal downsampling target FPS. `0` = use original FPS |

- **Example**: Set `TRACK_FPS=2` to track only 2 frames per second, reducing processing time for high-FPS videos.
- **Warning**: In `chunked_batch` mode, high frame counts require significant RAM/VRAM since all frames are held in memory.

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `9090` | Server listen port |
| `WORKERS` | `1` | Gunicorn worker processes |
| `THREADS` | `4` | Threads per worker |
| `LOG_LEVEL` | `DEBUG` | Logging verbosity |

- **Docker `shm_size`**: Set `shm_size: '32g'` or higher in `docker-compose.yml` for GPU operations.
- **Docker `mem_limit`**: Recommended `48g` to avoid system instability.

## CLI Tools

This directory contains several CLI tools for video tracking automation. All commands are run inside the Docker container via `docker compose exec`.

> **Migration status**: All CLI tools have been **migrated to SAM3**. They use the same HuggingFace Transformers-based SAM3 models as the ML backend. Video decoding uses PyAV (no cv2/OpenCV dependency). All commands should be run in the `segment_anything_3_video` container.

### 1. Automatic Object Detection & Tracking (`initial_seeding_video.py`)

**Status**: Migrated to SAM3

**Use case:** You have a raw video and want to automatically find and track all objects of a certain class (e.g., "person", "car").
**Input:** Raw video task. No existing annotations or bounding boxes are required.
**Method:** Uses SAM3 text-based detection (`Sam3VideoModel`) to find objects at keyframes, then `Sam3TrackerVideoModel` to track them, and stitches the results into complete tracks. Seed boxes are refined using text+box prompts before tracking (configurable).

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/initial_seeding_video.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --prompt "person" --keyframe-frac 0.1
```

**Arguments:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID |
| `--prompt` | No | `None` | Text prompt for detection (e.g., "person", "car") |
| `--keyframe-frac` | No | `0.1` | Fraction of frames to use as keyframes (0.1 = 10%) |
| `--min-spacing` | No | `30` | Minimum spacing between keyframes |
| `--embedding-batch` | No | `8` | Batch size for SAM3 embedding computation |
| `--cache-dir` | No | `./cache_dir/joblib` | Cache directory for embeddings |
| `--stitch-mode` | No | `teacher` | Stitching mode: `teacher` (embedding similarity) or `hungarian` (IoU + distance) |
| `--merge-threshold` | No | `0.6` | Cosine similarity threshold for teacher stitching (higher = stricter) |
| `--sparse-sequence` | No | `None` | Enable sparse sequence generation |
| `--no-sparse-sequence` | No | — | Disable sparse sequence generation |
| `--no-refine-seeds` | No | — | Disable seed box refinement (enabled by default) |
| `--refine-search-scale` | No | `1.3` | Search scale for refinement (1.3 = 30% expansion) |
| `--dry-run` | No | `False` | Save prediction to JSON file instead of uploading |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

### 2. Track Existing Manual Keyframes (`initial_seeding_video_boxes.py`)

**Status**: Migrated to SAM3

**Use case:** You have already drawn some bounding boxes (keyframes) in Label Studio and want SAM3 to track them forward and backward to fill the gaps.
**Input:** Video task with at least a few manual bounding boxes (keyframes).
**Method:** Uses your existing boxes as anchors, generates bidirectional tracklets using `Sam3TrackerVideoModel`, and uses the Hungarian algorithm to stitch them robustly. Seed boxes are refined using text+box prompts before tracking (configurable).

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/initial_seeding_video_boxes.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --global-start 0 --global-end 2000 --max-frames-to-track 300
```

**Arguments:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID |
| `--prompt` | No | `None` | Label override for all tracked objects |
| `--global-start` | No | `0` | Starting frame index (0-based inclusive) |
| `--global-end` | No | last frame | Ending frame index (0-based inclusive) |
| `--max-frames-to-track` | No | `300` | Max frames to track in each direction from keyframe |
| `--frame-stride` | No | `1` | Sample every N frames (1 = no downsampling) |
| `--no-refine-seeds` | No | — | Disable seed box refinement (enabled by default) |
| `--refine-search-scale` | No | `1.3` | Search scale for refinement (1.3 = 30% expansion) |
| `--dry-run` | No | `False` | Print prediction JSON instead of uploading |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

### 2a. Manual Merge Tracking (`initial_seeding_video_boxes_manual_merge.py`)

**Status**: Migrated to SAM3

**Use case:** You want bidirectional tracking per seed box, but you prefer to merge tracks manually using `mergevideoregions.py` and meta text IDs.
**Input:** Video task with manual keyframes. Optional track-id filtering lets you target specific seed regions per iteration.
**Method:** Builds forward+backward tracks per seed using `Sam3TrackerVideoModel` and keeps them in the same region; no automatic cross-seed merging. Each output region gets `meta.text="id:"` to ease manual ID assignment.

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/initial_seeding_video_boxes_manual_merge.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 225664 --task 245750672 --annotation 85260070 --global-start 1000 --global-end 1800 --max-frames-to-track 300
```

**With track filtering:**
```bash
docker compose exec segment_anything_3_video python /app/initial_seeding_video_boxes_manual_merge.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --track-id auto-track-14,auto-track-15 --global-start 1000 --global-end 1800
```

**Arguments:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID |
| `--prompt` | No | `None` | Label override for all tracked objects |
| `--track-id` | No | `None` | Comma-separated region IDs to use as anchors (omit for all) |
| `--global-start` | No | `None` | Starting frame index (0-based inclusive) |
| `--global-end` | No | `None` | Ending frame index (0-based inclusive) |
| `--max-frames-to-track` | No | `300` | Max frames to track in each direction from keyframe |
| `--frame-stride` | No | `1` | Sample every N frames (1 = no downsampling) |
| `--overlap-mode` | No | `iou-weighted` | Overlap resolution: `iou-weighted`, `weighted`, `winner` |
| `--overlap-iou-threshold` | No | `0.3` | IoU threshold for iou-weighted mode |
| `--no-refine-seeds` | No | — | Disable seed box refinement (enabled by default) |
| `--refine-search-scale` | No | `1.3` | Search scale for refinement (1.3 = 30% expansion) |
| `--dump-payload` | No | `None` | Path to write submission payload JSON before upload |
| `--no-progress` | No | `False` | Disable progress bars |
| `--dry-run` | No | `False` | Print prediction JSON instead of uploading |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

**Submission behavior:** If `--track-id`, `--global-start`, and `--global-end` are all provided, the script patches the existing annotation; otherwise it creates a new prediction.

### 3. Simple Forward Tracking (`cli.py`)

**Status**: Migrated to SAM3

**Use case:** Simple "predict" functionality similar to the UI button. Tracks from start to finish linearly.
**Input:** Video task with at least one manual keyframe (bounding box) to start tracking from.
**Method:** Uses SAM3 via model.py to load the entire video and propagate all keyframes at once. Best for short clips.

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/cli.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789
```

**Arguments:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID with keyframes |
| `--max-frames` | No | `0` | Max frames to track (0 = unlimited) |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

### 4. Post-Processing Tools (`video_tools.py`)

**Status**: No model dependency -- works in either container.

**Use case:** Clean up tracking results (sparsify dense frames, swap IDs, smooth jitter).

**Sparsify (Downsample Keyframes)**

Reduce the density of keyframes (e.g., keep 10% of frames) to make manual editing easier.

```bash
docker compose exec segment_anything_3_video python /app/video_tools.py sparsify --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --task 456 --annotation 789 --track-id auto-track-0 --start-frame 1000 --end-frame 2000 --ratio 0.1
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--track-id` | Yes | — | Region track ID (e.g., auto-track-0) |
| `--start-frame` | Yes | — | Start frame (1-based) |
| `--end-frame` | Yes | — | End frame (1-based) |
| `--ratio` | Yes | — | Fraction of frames to keep (0,1] |

**Swap IDs (Fix Identity Switches)**

Move a segment of tracking history from one object ID to another (e.g., if the tracker swapped "Person A" to "Person B").

```bash
docker compose exec segment_anything_3_video python /app/video_tools.py swap-ids --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --task 456 --annotation 789 --source-track-id auto-track-0 --target-track-id auto-track-1 --start-frame 500 --end-frame 600
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--source-track-id` | Yes | — | Source region track ID |
| `--target-track-id` | Yes | — | Target region track ID |
| `--start-frame` | Yes | — | Start frame (1-based) |
| `--end-frame` | Yes | — | End frame (1-based) |

**Trim Tail (Delete Trailing Frames)**

Delete all keyframes for a specific track after a certain cutoff frame (e.g., when an object leaves the view).

```bash
docker compose exec segment_anything_3_video python /app/video_tools.py trim-tail --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --task 456 --annotation 789 --track-id auto-track-0 --cutoff-frame 1500
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--track-id` | Yes | — | Region track ID |
| `--cutoff-frame` | Yes | — | Delete all frames after this (1-based) |

**Smooth (Stabilize Jitter)**

Apply a moving average filter to smooth out shaky bounding boxes.

```bash
docker compose exec segment_anything_3_video python /app/video_tools.py smooth --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --task 456 --annotation 789 --track-id auto-track-0 --window 5
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--track-id` | Yes | — | Region track ID |
| `--window` | No | `5` | Moving average window size |

**Pad (Expand Bounding Boxes)**

Inflate bounding boxes by a percentage (e.g., 10%) over a specific frame range. Useful if the tracker is consistently too tight.

```bash
docker compose exec segment_anything_3_video python /app/video_tools.py pad --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --task 456 --annotation 789 --track-id auto-track-0 --percent 0.10 --start-frame 0 --end-frame 1000
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--track-id` | Yes | — | Region track ID |
| `--percent` | Yes | — | Expansion percentage (e.g., 0.10 = 10%) |
| `--start-frame` | Yes | — | Start frame (1-based) |
| `--end-frame` | Yes | — | End frame (1-based) |

**Common arguments for all video_tools.py commands:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | No | env `LABEL_STUDIO_URL` | Label Studio URL |
| `--ls-api-key` | No | env `LABEL_STUDIO_API_KEY` | Label Studio API key |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID |
| `--dry-run` | No | `False` | Write updated JSON to file instead of PATCH |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

### 5. Prediction Validation (`validate_prediction.py`)

**Status**: No model dependency -- works in either container.

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/validate_prediction.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --task 456 --prediction-file prediction.json --upload
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--task` | Yes | — | Task ID |
| `--prediction-file` | Yes | — | Path to prediction JSON file |
| `--upload` | No | `False` | Upload prediction to Label Studio |

### 6. Export Utilities (`export_interpolated_annotation.py`)

**Status**: No model dependency -- works in either container.

**Use case:** Download a single annotation JSON with *all* interpolated video frames included (not just keyframes). This is critical for getting the full frame-by-frame tracking data out of Label Studio.

**Example (Python):**
```bash
docker compose exec segment_anything_3_video python /app/export_interpolated_annotation.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --output-dir /app/exports
```

**Example (Bash script):**
```bash
docker compose exec segment_anything_3_video /app/export_interpolated_annotation.sh --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --output /app/exports/annotation.json
```

**Summary output:** The bash script also writes a per-casualty summary JSON next to the exported annotation. If the output is `annotation.json`, the summary is written to `annotation.summary.json`. The summary includes frame/time ranges per `meta.text` ID.

### 6a. Generate Casualty Snippets (integrated into `export_interpolated_annotation.sh`)

**Use case:** Export the interpolated annotation, create summary JSON, and generate per-casualty snippets plus per-snippet bbox JSON outputs in one step.

**Example:**
```bash
docker compose exec segment_anything_3_video /app/export_interpolated_annotation.sh --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --output /app/exports/annotation.json --snippets --person-id 31 --min-seconds 2 --fps 10
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--snippets` | No | `False` | Enable snippet generation |
| `--snippets-dir` | No | auto-generated | Output directory for snippets |
| `--person-id` | No | all | Generate snippets for specific person ID only |
| `--min-frames` | No | — | Skip ranges shorter than N frames (mutually exclusive with `--min-seconds`) |
| `--min-seconds` | No | — | Skip ranges shorter than N seconds (mutually exclusive with `--min-frames`) |
| `--fps` | No | original | Output FPS (omit to use original with stream-copy) |

**Output files:**
- `casualty_<id>_f<start>-<end>_fps<fpsInt>.mp4`
- `casualty_<id>_f<start>-<end>_fps<fpsInt>.json` with frame-level bbox entries
- `README.txt` in the output folder capturing parameters used

### 6b. Overlay BBoxes on Snippets (`overlay_snippet_bboxes.sh`)

**Use case:** Draw bbox overlays on an existing snippet using its bbox JSON.

**Example:**
```bash
docker compose exec segment_anything_3_video /app/overlay_snippet_bboxes.sh --snippet /app/exports/casualty_31_f1000-2400_fps10.mp4 --bbox-json /app/exports/casualty_31_f1000-2400_fps10.json
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--snippet` | Yes | — | Path to snippet video |
| `--bbox-json` | Yes | — | Path to bbox JSON file |
| `--output` | No | `<snippet>_bbox_overlaid.mp4` | Output video path |
| `--chunk-size` | No | `1000` | Frames per chunk (for large snippets) |

### 7. Deletion Utilities (`delete_annotation_or_prediction.py`)

**Status**: No model dependency -- works in either container.

**Delete an annotation:**
```bash
docker compose exec segment_anything_3_video python /app/delete_annotation_or_prediction.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789
```

**Delete a prediction:**
```bash
docker compose exec segment_anything_3_video python /app/delete_annotation_or_prediction.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --prediction 555
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Either | — | Annotation ID to delete (mutually exclusive with `--prediction`) |
| `--prediction` | Either | — | Prediction ID to delete (mutually exclusive with `--annotation`) |

### 8. Merge Video Regions (`mergevideoregions.py`)

**Status**: No model dependency -- works in either container.

**Use case:** Consolidate fragmented tracks that share the same text ID (e.g., `id:31` in `meta.text`) into single continuous track objects. Useful after manual labeling or ReID where multiple regions represent the same object.

**Example (from annotation):**
```bash
docker compose exec segment_anything_3_video python /app/mergevideoregions.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789
```

**Example (from prediction):**
```bash
docker compose exec segment_anything_3_video python /app/mergevideoregions.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --prediction 555
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Either | — | Annotation ID as source (mutually exclusive with `--prediction`) |
| `--prediction` | Either | — | Prediction ID as source (mutually exclusive with `--annotation`) |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

### 9. Bounding Box Refinement (`adjust_bboxes_sam3.py`)

**Status**: Migrated to SAM3

**Use case:** Tighten or adjust existing bounding boxes that have drifted due to tracker instability (camera movement, zoom, person motion). Works for boxes that are too large OR too small.

**Method:** Uses SAM3's combined text+box prompt capability:
- The **text prompt** (from track label or `--default-label`) tells SAM3 WHAT to segment (e.g., "person")
- The **expanded box prompt** tells SAM3 WHERE to look

This approach handles bidirectional drift - even if the original box doesn't fully contain the target, the expanded search region should, and the text prompt ensures SAM3 finds the right object.

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/adjust_bboxes_sam3.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --search-scale 1.3 --default-label person
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID whose bboxes will be refined |
| `--search-scale` | No | `1.3` | Search region expansion (1.3 = 30% larger). Increase for more drift tolerance |
| `--default-label` | No | `person` | Text prompt when track has no label |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

### 10. Re-Identification (`complete_reid.py`)

**Status**: Migrated to SAM3

**Use case:** Automatically suggest identity matches for broken tracks. Uses appearance features (color, geometry, or SAM3 embeddings) to find likely matches between "candidate" tracks (no ID) and "reference" tracks (confirmed ID).

**Example:**
```bash
docker compose exec segment_anything_3_video python /app/complete_reid.py --ls-url "$LABEL_STUDIO_HOST" --ls-api-key "$LABEL_STUDIO_API_KEY" --project 123 --task 456 --annotation 789 --profile uav --feature-backend sam3
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--ls-url` | Yes | — | Label Studio URL |
| `--ls-api-key` | Yes | — | Label Studio API key |
| `--project` | Yes | — | Project ID |
| `--task` | Yes | — | Task ID |
| `--annotation` | Yes | — | Annotation ID as source of tracks |
| `--profile` | No | `uav` | Feature weighting preset: `uav`, `ugv` |
| `--feature-backend` | No | `classic` | `classic` (color/geometry via numpy/scipy) or `sam3` (neural embeddings) |
| `--sam3-padding-fraction` | No | `0.1` | Padding around boxes for SAM3 embedding extraction |
| `--log-level` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

## CLI Migration Status Summary

| Script | Status | Notes |
|--------|--------|-------|
| `model.py` (ML backend) | **Migrated** | SAM3 via HuggingFace Transformers |
| `cli.py` | **Migrated** | Uses SAM3 via model.py |
| `initial_seeding_video.py` | **Migrated** | Sam3VideoModel (text detection) + Sam3TrackerVideoModel |
| `initial_seeding_video_boxes.py` | **Migrated** | Sam3TrackerVideoModel with bidirectional tracking |
| `initial_seeding_video_boxes_manual_merge.py` | **Migrated** | Sam3TrackerVideoModel, no cross-seed merging |
| `seeding_common.py` | **Migrated** | Lazy-loaded SAM3 singletons, PyAV video I/O |
| `adjust_bboxes_sam3.py` | **Migrated** | Sam3Model for box-prompted segmentation |
| `complete_reid.py` | **Migrated** | SAM3 embeddings backend, numpy/scipy for classic features |
| `video_tools.py` | No model dependency | Post-processing utilities |
| `export_interpolated_annotation.py` | No model dependency | Export with interpolation |
| `export_interpolated_annotation.sh` | No model dependency | Bash export + snippets |
| `overlay_snippet_bboxes.sh` | No model dependency | BBox overlay on snippets |
| `validate_prediction.py` | No model dependency | Prediction validation |
| `delete_annotation_or_prediction.py` | No model dependency | Deletion utility |
| `mergevideoregions.py` | No model dependency | Track merging |
| `update_video_paths.py` | No model dependency | Path updates |

All CLI tools now run in the `segment_anything_3_video` container. OpenCV (cv2) has been completely removed; video decoding uses PyAV and image processing uses PIL/numpy/scipy.

## Known Limitations

- SAM3 is designed to run on GPU servers; CPU execution is not recommended for practical video workloads.
- Currently, we do not support video segmentation (only bounding boxes).
- For very long videos (40,000+ frames), tracking may take significant time. Consider using `MAX_FRAMES_TO_TRACK` to process in chunks.

## Customization

The ML backend can be customized by adding your own models and logic inside the `./segment_anything_3_video` directory.
