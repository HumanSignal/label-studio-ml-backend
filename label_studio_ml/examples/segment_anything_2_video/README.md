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

## CLI Usage for Batch Processing

You can use the CLI to run SAM2 tracking on tasks with existing annotations (keyframes). This is useful for processing long videos in parallel segments.

### Arguments

- `--ls-url`: Label Studio URL (required)
- `--ls-api-key`: API key (required)
- `--project`: Project ID (required)
- `--task`: Task ID (required)
- `--annotation`: Annotation ID with keyframes (required)
- `--global-start`: Start frame index (0-based inclusive, default: 0)
- `--global-end`: End frame index (0-based inclusive, default: last frame)
- `--max-frames-to-track`: Max frames to track forward/backward from each keyframe (default: 300)
- `--prompt`: Optional label override (e.g., "Person")
- `--dry-run`: Print JSON output instead of uploading

### Example Command

To run tracking on a specific segment (frames 0-2000):

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST=https://app.heartex.com
# API key can also be passed as argument
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"
 
python /app/initial_seeding_video_boxes.py \
  --ls-url https://app.heartex.com \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project 198563 \
  --task 226454007 \
  --annotation 79598308 \
  --global-start 0 \
  --global-end 2000 \
  --max-frames-to-track 300
'
```

### Processing Strategy

For very long videos (e.g., 1 hour), you should:
1. Divide the video into overlapping segments (e.g., 0-2000, 1900-3900...)
2. Run this script for each segment in parallel (using different `--global-start`/`--global-end`)
3. The script will:
   - Extract only the needed frames to a temp directory
   - Filter keyframes relevant to this segment
   - Track objects bidirectionally
   - Stitch tracks using Hungarian matching
   - Upload results back to Label Studio

Note: Tracks are uploaded as new "videorectangle" regions. You may want to merge them later or use `video_tools.py` to clean up results.

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_HOST=https://app.heartex.com
export LABEL_STUDIO_URL=https://app.heartex.com
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/cli.py \
  --ls-url https://app.heartex.com \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project 198563 \
  --task 227350954 \
  --annotation 12345'
```

### CLI Parameters:
- `--ls-url`: Label Studio URL (e.g., https://app.heartex.com)
- `--ls-api-key`: Your Label Studio API key
- `--project`: Project ID
- `--task`: Task ID to process
- `--annotation`: Annotation ID containing keyframes to track
- `--max-frames`: (Optional) Limit tracking to N frames (default: tracks full video)

### Tracking Strategy:
- **Multiple keyframes (recommended for long videos)**: Draw boxes at key moments (start, turns, occlusions). SAM2 uses them as guidance points for better tracking accuracy.
- **Single keyframe**: Draw one box per person at video start. SAM2 tracks forward from there.
- The model tracks from the first keyframe to the end of the video (or `--max-frames` limit).
- Supports multi-person tracking: annotate multiple people and track them all simultaneously.

## Video annotation toolbox (`video_tools.py`)

If you need to clean up or adjust existing video tracks after tracking, use `video_tools.py`. It fetches an existing task annotation from Label Studio, modifies the region `result` locally, and patches the annotation back to Label Studio.

Track selection requirements:
- Tracks are selected by the region `result[i]["id"]` (for example `auto-track-0`).
- The region `id` is case-sensitive and is expected to be unique within the annotation.

Common parameters:
- `--task`, `--annotation`: Identify the task and annotation to modify.
- `--ls-url`, `--ls-api-key`: Or use `LABEL_STUDIO_URL` / `LABEL_STUDIO_HOST` and `LABEL_STUDIO_API_KEY`.
- `--dry-run`: Write the updated annotation JSON locally instead of uploading.

Commands:
- `sparsify`: Uniformly downsample keyframes in a frame range (keep a ratio).
- `swap-ids`: Move a segment of video history from one track id to another.
- `trim-tail`: Delete all keyframes after a cutoff frame.
- `smooth`: Moving-average smoothing over `x`, `y`, `width`, `height`.
- `pad`: Inflate boxes by a percentage over a frame range (clamped to 0-100%).

Example (inside Docker):

```bash
docker compose exec segment_anything_2_video bash -lc '
export LABEL_STUDIO_URL=https://app.heartex.com
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/video_tools.py sparsify \
  --task 227350954 --annotation 12345 \
  --track-id auto-track-0 \
  --start-frame 1000 --end-frame 2000 --ratio 0.1
'
```

Note: For large updates, Label Studio may return `504 Gateway Timeout` on `PATCH` even though the update succeeds. `video_tools.py` treats HTTP 504 as success and logs a warning.

## Automatic initial seeding (Grounding DINO + SAM2)

If you want to bootstrap a task **without drawing any keyframes**, you can generate an initial set of tracks using:

- Grounding DINO (open-vocabulary box detection on selected keyframes)
- SAM2 (keyframe selection via embeddings + per-frame mask-to-box tracking)

Run it (from source or inside the container) using `initial_seeding.py`:

```bash
python initial_seeding.py \
  --ls-url https://app.heartex.com \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project 198563 \
  --task 227350954 \
  --annotation 12345 \
  --dry-run
```

Notes:
- `--annotation` is currently required for validation/logging, even though the seeding pipeline does not use keyframe regions from it.
- Use `--dry-run` first to write `prediction_task_<TASK_ID>.json` locally, then validate (and optionally upload) with:

```bash
python validate_prediction.py \
  --ls-url https://app.heartex.com \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --task 227350954 \
  --prediction-file prediction_task_227350954.json \
  --upload
```

Key parameters:
- `--keyframe-frac`: Fraction of video frames to treat as keyframes (default: `0.1`)
- `--min-spacing`: Minimum spacing between high-change keyframes (default: `30`)
- `--embedding-batch`: SAM2 embedding batch size (default: `8`)
- `--num-workers`: Parallel workers for keyframe-pair tracking (default: `4`)

## Configuration

### Environment Variables:
- `DEVICE`: Computing device (recommended: `cuda`)
- `MODEL_CONFIG`: SAM2 model config path.
  - In `docker-compose.yml` this is set to `configs/sam2.1/sam2.1_hiera_l.yaml`
- `MODEL_CHECKPOINT`: SAM2 checkpoint filename.
  - In `docker-compose.yml` this is set to `sam2.1_hiera_large.pt`
- `MAX_FRAMES_TO_TRACK`: Hard limit for how many frames to track from the first keyframe.
  - Set to `0` for no limit.
  - In `model.py` the default is `1000` if not set.
- `TRACK_FPS`: Temporal downsampling target FPS for tracking (used by `model.py`).
  - Set to `0` to disable downsampling (default: `0`).
- `LABEL_STUDIO_HOST`, `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_KEY`: Used to resolve and download video assets via `get_local_path`.

Initial seeding (`initial_seeding.py`) also uses:
- `GROUNDINGDINO_REPO_PATH`: Path to Grounding DINO repo (docker default: `/GroundingDINO`)
- `GROUNDING_DINO_CONFIG`, `GROUNDING_DINO_WEIGHTS`: Config and weights name under `${GROUNDINGDINO_REPO_PATH}`
- `GROUNDING_DINO_PROMPT` or `GROUNDING_DINO_LABELS`: Class prompt for detection
- `GROUNDING_DINO_BOX_THRESHOLD`, `GROUNDING_DINO_TEXT_THRESHOLD`: Detection thresholds
- `CACHE_DIR`: Joblib cache directory for SAM2 embeddings (default: `./cache_dir/joblib`)
- `EMBED_BATCH`: Embedding batch size (default: `8`)
- `SAM2_NUM_WORKERS`: Parallel workers for SAM2 tracking pairs (default: `4`)

## Known limitations
- SAM2 is designed to run on GPU servers; CPU execution is not recommended for practical video workloads.
- Currently, we do not support video segmentation (only bounding boxes).
- For very long videos (40,000+ frames), tracking may take significant time. Consider using `--max-frames` to process in chunks.

If you want to contribute to this repository to help with some of these limitations, you can submit a PR.

## Customization

The ML backend can be customized by adding your own models and logic inside the `./segment_anything_2_video` directory. 
