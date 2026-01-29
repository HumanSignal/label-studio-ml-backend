# Grounding DINO + ByteTrack Video ML Backend

This ML backend provides **zero-shot object detection and multi-object tracking** for video annotation in Label Studio using:

- **Grounding DINO** (SwinT) for text-prompted object detection
- **ByteTrack** (via `supervision.ByteTrack`) for multi-object tracking
- **Composable tracking presets** for different video scenarios (UAV, thermal, crowded scenes, etc.)
- **CUDA/FP16 acceleration** for faster frame-by-frame processing

## Table of Contents

1. [Quick Start](#quick-start)
2. [Tracking Presets](#tracking-presets)
3. [Performance Notes](#performance-notes)
4. [CLI Reference](#cli-reference)
5. [Configuration Reference](#configuration-reference)
6. [Debugging](#debugging)
7. [FPS Synchronization Utility](#fps-synchronization-utility)
8. [Development](#development)

---

## Quick Start

### 1. Configure Environment

Edit `docker-compose.yml` with your Label Studio credentials:

```yaml
environment:
  LABEL_STUDIO_URL: https://app.heartex.com
  LABEL_STUDIO_API_KEY: ${LABEL_STUDIO_API_KEY}
  GROUNDING_DINO_PROMPT: person
```

### 2. Run with Docker

```bash
docker compose up --build
```

### 3. Run Predictions via CLI

```bash
# Basic usage
docker compose exec grounding_dino_video bash -lc '
python /app/cli.py --project 123 --tasks 456,457,458'

# With tracking preset for UAV footage (applies env thresholds)
docker compose exec grounding_dino_video bash -lc '
python /app/cli.py --preset uav+long_video --project 123 --tasks 456'

# With debugging output (saves annotated frames)
docker compose exec grounding_dino_video bash -lc '
python /app/cli.py --preset uav --save-frames --output-dir ./debug --project 123 --tasks 456'
```

---

## Tracking Presets

Presets provide pre-configured detection and tracking parameters optimized for different scenarios. **Presets are composable** — combine multiple layers with `+` to address multiple concerns.

### Available Preset Layers

| Category | Layers | Description |
|----------|--------|-------------|
| **Platform** | `uav`, `ugv`, `handheld`, `fixed` | Camera/vehicle type |
| **Scene** | `crowded`, `sparse`, `cluttered` | Scene characteristics |
| **Motion** | `fast_motion`, `slow_motion`, `erratic` | Subject motion patterns |
| **Duration** | `long_video`, `short_clip` | Video length considerations |
| **Modality** | `thermal`, `lowlight`, `hdr` | Sensor/image type |
| **Quality** | `high_precision`, `high_recall` | Detection quality tuning |

### Preset Examples

```bash
# UAV footage with fast-moving subjects in a long video
docker compose exec grounding_dino_video bash -lc '
python /app/cli.py --preset uav+fast_motion+long_video --project 123 --tasks 456'

# Thermal imagery with crowded scene, prioritizing precision
docker compose exec grounding_dino_video bash -lc '
python /app/cli.py --preset thermal+crowded+high_precision --project 123 --tasks 456'

# Ground vehicle with slow-moving subjects
docker compose exec grounding_dino_video bash -lc '
python /app/cli.py --preset ugv+slow_motion --project 123 --tasks 456'
```

### Preset Commands

```bash
# List all available preset layers
docker compose exec grounding_dino_video bash -lc 'python /app/cli.py --list-presets'

# Show computed parameter values for a preset combination
docker compose exec grounding_dino_video bash -lc 'python /app/cli.py --describe-preset uav+fast_motion+long_video'
```

### Example Output: `--describe-preset uav+fast_motion+long_video`

```
Preset: uav+fast_motion+long_video
Description: Aerial/drone: small subjects, fast relative motion + High-speed subjects + Long videos (>10 min)
Layers: uav, fast_motion, long_video

Detection Parameters:                    Value   Valid Range
  box_threshold:                         0.35 [0.05-0.95]
  text_threshold:                        0.20 [0.05-0.95]
  model_score_threshold:                 0.65 [0.05-0.95]

Tracking Parameters:
  track_activation_threshold:            0.55 [0.05-0.95]
  lost_track_buffer:                      400 [1-1800] frames
  minimum_matching_threshold:            0.15 [0.05-0.95]
  minimum_consecutive_frames:              17 [1-100]

All values are validated and clamped to valid ranges automatically.
```

### Preset Layer Details

#### Platform Layers

| Layer | Use Case | Key Adjustments |
|-------|----------|-----------------|
| `uav` | Aerial/drone footage | Higher thresholds, longer lost buffer, lower IoU matching |
| `ugv` | Ground vehicle | Balanced defaults |
| `handheld` | Handheld camera | Lower IoU matching for shake |
| `fixed` | Stationary camera | Stricter matching, shorter buffer |

#### Scene Layers

| Layer | Use Case | Key Adjustments |
|-------|----------|-----------------|
| `crowded` | Many subjects, occlusions | Higher thresholds, 50% longer buffer |
| `sparse` | Few subjects | Lower thresholds, shorter buffer |
| `cluttered` | Complex backgrounds | Higher thresholds, more consecutive frames |

#### Motion Layers

| Layer | Use Case | Key Adjustments |
|-------|----------|-----------------|
| `fast_motion` | High-speed subjects | Very permissive IoU (0.15), lower thresholds |
| `slow_motion` | Slow-moving subjects | Strict IoU (0.55), shorter buffer |
| `erratic` | Unpredictable motion | Lower IoU, longer buffer |

#### Duration Layers

| Layer | Use Case | Key Adjustments |
|-------|----------|-----------------|
| `long_video` | Videos >10 min | 2x buffer, higher thresholds, more consecutive frames |
| `short_clip` | Clips <1 min | 0.5x buffer, lower thresholds |

#### Modality Layers

| Layer | Use Case | Key Adjustments |
|-------|----------|-----------------|
| `thermal` | Thermal/IR imagery | Lower thresholds (0.20), longer buffer |
| `lowlight` | Low-light conditions | Lower box threshold, more consecutive frames |
| `hdr` | High dynamic range | Slightly lower thresholds |

#### Quality Layers

| Layer | Use Case | Key Adjustments |
|-------|----------|-----------------|
| `high_precision` | Minimize false positives | +0.15 to all thresholds, +10 consecutive frames |
| `high_recall` | Minimize missed detections | -0.10 from thresholds, -3 consecutive frames |

---

## Performance Notes

- Frames are processed sequentially; parallelism is not used in detection.
- **CUDA + FP16:** Mixed precision is enabled automatically on CUDA for speedups with minimal accuracy impact.
- **Resolution controls:** Use `GROUNDING_DINO_BASE_SIZE` and `GROUNDING_DINO_MAX_SIZE` to tune input resolution (defaults: 800x1333).
- **Logging cadence:** Adjust progress logs with `GROUNDING_DINO_PROGRESS_EVERY` (default: every 25 frames).

---

## CLI Reference

### Basic Usage

```bash
python cli.py [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--ls-url` | `$LABEL_STUDIO_URL` | Label Studio URL |
| `--ls-api-key` | `$LABEL_STUDIO_API_KEY` | Label Studio API key |
| `--project` | `1` | Project ID |
| `--tasks` | `tasks.json` | Task IDs (comma-separated) or JSON file |
| `--preset` | `$TRACKING_PRESET` | Tracking preset(s), combine with `+` |
| `--save-frames` | `false` | Save annotated frames for debugging |
| `--output-dir` | `./output_frames` | Directory for saved frames |
| `--list-presets` | - | List all preset layers and exit |
| `--describe-preset` | - | Show computed values for a preset |

### Examples

```bash
# Process specific tasks with UAV preset
docker compose exec grounding_dino_video bash -lc '
python /app/cli.py --preset uav --project 123 --tasks 456,457,458'

# Compose multiple presets
docker compose exec grounding_dino_video bash -lc '
python /app/cli.py --preset uav+fast_motion+long_video --project 123 --tasks 456'

# Debug with saved frames
docker compose exec grounding_dino_video bash -lc '
python /app/cli.py --preset thermal --save-frames --output-dir ./debug --project 123 --tasks 456'
```

---

## Configuration Reference

### docker-compose.yml

```yaml
environment:
  # Label Studio connection
  LABEL_STUDIO_URL: https://app.heartex.com
  LABEL_STUDIO_API_KEY: ${LABEL_STUDIO_API_KEY}

  # Grounding DINO detection
  GROUNDING_DINO_PROMPT: person
  GROUNDING_DINO_BOX_THRESHOLD: "0.20"
  GROUNDING_DINO_TEXT_THRESHOLD: "0.25"
  GROUNDING_DINO_DEVICE: cuda
  MODEL_SCORE_THRESHOLD: "0.5"

  # Tracking preset (recommended)
  # TRACKING_PRESET: "uav+long_video"
```

### Parameter Bounds

All preset parameters are validated and clamped to sensible bounds:

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| `box_threshold` | 0.05 | 0.95 | Detection box confidence |
| `text_threshold` | 0.05 | 0.95 | Text-to-region matching |
| `model_score_threshold` | 0.05 | 0.95 | Final prediction score |
| `track_activation_threshold` | 0.05 | 0.95 | Track activation confidence |
| `lost_track_buffer` | 1 | 1800 | Frames to keep lost track (1-60s at 30fps) |
| `minimum_matching_threshold` | 0.05 | 0.95 | IoU for track matching |
| `minimum_consecutive_frames` | 1 | 100 | Frames before track confirmation |

---

## Debugging

### Save Annotated Frames

```bash
docker compose exec grounding_dino_video bash -lc '
python /app/cli.py --save-frames --output-dir ./debug --project 123 --tasks 456'
```

Each frame is saved as `frame_XXXXXX.jpg` with:
- Bounding boxes (color-coded by track)
- Track IDs (e.g., "#12")
- Class labels and confidence scores
- Frame numbers

### Logging

```bash
# Enable debug logging
docker compose exec grounding_dino_video bash -lc '
LOG_LEVEL=DEBUG python /app/cli.py --project 123 --tasks 456'
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Too many tracks | Use `high_precision` or increase `track_activation_threshold` |
| Tracks fragmenting | Use `long_video` or increase `lost_track_buffer` |
| Missing detections | Use `high_recall` or decrease thresholds |
| GPU OOM | Lower input resolution (`GROUNDING_DINO_BASE_SIZE`, `GROUNDING_DINO_MAX_SIZE`) or shorten clips |

---

## Label Studio Integration

### Video Object Tracking

Use the `<VideoRectangle>` control tag for video object tracking:

```xml
<View>
  <Video name="video" value="$video"/>
  <VideoRectangle name="box" toName="video"/>
  <Labels name="label" toName="video">
    <Label value="person" background="red"/>
  </Labels>
</View>
```

### Connecting the Model

1. From the **Model** page in project settings, [connect the model](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio)
2. Default URL: `http://localhost:9090`
3. Add videos to Label Studio
4. Open any task to see predictions

---

## Architecture

### Detection Pipeline

1. **Frame Extraction** — Video frames are read using OpenCV
2. **Detection** — Frames are processed on GPU with Grounding DINO
3. **Tracking** — ByteTrack associates detections across frames
4. **Output** — Track annotations are uploaded to Label Studio

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| CLI | `cli.py` | Command-line interface for task processing |
| Presets | `tracking_presets.py` | Composable preset system |
| Detection | `utils/grounding.py` | Grounding DINO inference for detection/tracking |
| Tracking | `control_models/video_rectangle.py` | ByteTrack integration |
| FPS Sync | `update_fps.py` | Utility to synchronize FPS values for video tasks |

---

## FPS Synchronization Utility

The `update_fps.py` utility synchronizes accurate FPS values for Label Studio video tasks by reading video metadata and updating task data.

### Usage

```bash
# Update specific tasks (comma-separated IDs)
docker compose exec grounding_dino_video bash -lc '
export LABEL_STUDIO_HOST=https://app.heartex.com
export LABEL_STUDIO_URL=https://app.heartex.com
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/update_fps.py \
  --ls-url https://app.heartex.com \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project 123 \
  --tasks 10,11,12'

# Update specific tasks (JSON file)
docker compose exec grounding_dino_video bash -lc '
export LABEL_STUDIO_HOST=https://app.heartex.com
export LABEL_STUDIO_URL=https://app.heartex.com
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/update_fps.py \
  --ls-url https://app.heartex.com \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project 123 \
  --tasks my_tasks.json'

# Update ALL tasks in a project
docker compose exec grounding_dino_video bash -lc '
export LABEL_STUDIO_HOST=https://app.heartex.com
export LABEL_STUDIO_URL=https://app.heartex.com
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/update_fps.py \
  --ls-url https://app.heartex.com \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project 123'

# With overwrite flag (updates existing FPS values)
docker compose exec grounding_dino_video bash -lc '
export LABEL_STUDIO_HOST=https://app.heartex.com
export LABEL_STUDIO_URL=https://app.heartex.com
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/update_fps.py \
  --ls-url https://app.heartex.com \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project 123 \
  --overwrite'

# With explicit data key
docker compose exec grounding_dino_video bash -lc '
export LABEL_STUDIO_HOST=https://app.heartex.com
export LABEL_STUDIO_URL=https://app.heartex.com
export LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY"

python /app/update_fps.py \
  --ls-url https://app.heartex.com \
  --ls-api-key "$LABEL_STUDIO_API_KEY" \
  --project 123 \
  --data-key video_path'
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--ls-url` | `$LABEL_STUDIO_URL` | Label Studio URL |
| `--ls-api-key` | `$LABEL_STUDIO_API_KEY` | Label Studio API key |
| `--project` | `None` | Project ID (required if --tasks not provided) |
| `--tasks` | `None` | Task IDs (comma-separated) or JSON file |
| `--data-key` | `None` | Explicit key in task['data'] storing video path |
| `--overwrite` | `false` | Overwrite existing FPS values |
| `--quiet` | `false` | Reduce log verbosity |

### Notes

- If `--tasks` is not provided, the script requires `--project` and will fetch all tasks from that project
- FPS values are extracted using OpenCV and rounded to 6 decimal places
- Existing FPS values are preserved unless `--overwrite` is specified
- Supports various video formats: MP4, MOV, AVI, MKV, WebM, M4V, MPG, MPEG

---

## Development

### Running Without Docker

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the ML backend
label-studio-ml start .
```

### Testing Presets

```bash
# List all presets
docker compose exec grounding_dino_video bash -lc 'python -c "from tracking_presets import list_presets; print(list_presets())"'

# Describe a preset
docker compose exec grounding_dino_video bash -lc 'python -c "from tracking_presets import describe_preset; print(describe_preset(\"uav+long_video\"))"'
```
