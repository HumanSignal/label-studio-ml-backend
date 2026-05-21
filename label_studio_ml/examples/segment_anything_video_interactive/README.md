# Segment Anything — Video Interactive (SAM2) ML Backend

An interactive ML backend that mirrors the in-browser BYOM SAM tag in Label
Studio Enterprise. Accepts click / box / multi-frame prompts, returns
bitmask / rectangle / polygon / video-rectangle results via the standard
`/predict` endpoint.

## What's different from `segment_anything_2_video`

- **Dual mode** — image and video under one backend, dispatched on the label
  config's object tag.
- **Prewarm + sticky frame cache** — frames are encoded in a window around the
  user's current frame in the navigation direction. Nothing is re-encoded
  unless a memory cap forces eviction. Spatial eviction drops frames furthest
  from the current frame first.
- **Per-task state** — no process-wide globals for inference state, so
  concurrent requests across tasks don't clobber each other.
- **Standard `PredictionValue` output** — no custom wire protocol; the Label
  Studio tag consumes the exact shape any other interactive ML backend emits.

## Wire protocol

Two modes multiplex through `/predict`, selected by `context.event`:

### Prewarm

```json
{
  "tasks": [{"id": 1, "data": {"video": "..."}}],
  "params": {
    "context": {
      "event": "prewarm",
      "frame": 42,
      "window": 20,
      "direction": "forward"
    }
  }
}
```

Response `results[0].value`: `{"status": "ok", "cached": [...], "pending": [...], "frame_count": N}`.

### Predict

```json
{
  "tasks": [{"id": 1, "data": {"video": "..."}}],
  "params": {
    "context": {
      "frame": 42,
      "result": [
        {"type": "keypointlabels", "value": {"x": 45.2, "y": 30.1, "positive": true}},
        {"type": "keypointlabels", "value": {"x": 60.0, "y": 55.0, "positive": false}}
      ]
    }
  }
}
```

Response: standard `PredictionValue` with `result[0].value` shaped for the
project's control tag.

## Configuration (env)

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cuda` | `cuda`, `mps`, or `cpu` |
| `MODEL_CONFIG` | `sam2_hiera_l.yaml` | SAM2 config name inside the SAM2 repo |
| `MODEL_CHECKPOINT` | `sam2_hiera_large.pt` | SAM2 checkpoint filename |
| `WINDOW_SIZE` | `20` | default prewarm window if client omits `window` |
| `MAX_CACHED_FRAMES_PER_TASK` | `500` | hard frame-count ceiling per task |
| `MAX_TASK_CACHE_MB` | `2048` | hard byte ceiling per task |
| `MAX_GLOBAL_CACHE_MB` | `8192` | hard byte ceiling across all tasks |
| `TASK_CACHE_TTL_SECONDS` | `1800` | drop idle task caches after this long |
| `LABEL_STUDIO_URL` / `LABEL_STUDIO_API_KEY` | — | needed to fetch task assets from LS |

## Running locally (no Docker)

1. Clone SAM2 next to this example:
   ```bash
   cd label_studio_ml/examples/segment_anything_video_interactive
   git clone https://github.com/facebookresearch/segment-anything-2.git
   (cd segment-anything-2 && pip install -e . && cd checkpoints && ./download_ckpts.sh)
   ```
2. Install backend requirements (from the repo root):
   ```bash
   pip install -e .
   pip install -r label_studio_ml/examples/segment_anything_video_interactive/requirements.txt
   ```
3. Start the dev server:
   ```bash
   cd label_studio_ml/examples/segment_anything_video_interactive
   export DEVICE=mps            # on Apple Silicon, or keep cuda on NVIDIA
   export LABEL_STUDIO_URL=http://host.docker.internal:8080
   export LABEL_STUDIO_API_KEY=<your LSE API key>
   python _wsgi.py -p 9090
   ```

## Running via Docker

```bash
cd label_studio_ml/examples/segment_anything_video_interactive
# fill in LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY in docker-compose.yml first
docker compose up --build
```

The service listens on `:9090`.

## Connecting to Label Studio Enterprise

1. In LSE → **Settings → Model** → **Connect Model**.
2. URL: `http://host.docker.internal:9090` (or wherever the backend is
   reachable from the LSE container).
3. Toggle **Interactive preannotations** ON.
4. Save. LSE will POST `/setup` and `/health` to verify.

## Verifying

```bash
curl -s localhost:9090/health
# {"status":"UP","model_class":"SamVideoInteractive"}

curl -s localhost:9090/predict -H 'content-type: application/json' -d '{
  "tasks":[{"id":1,"data":{"video":"https://..."}}],
  "project":"1.1700000000",
  "label_config":"<View><Video name=\"video\" value=\"$video\"/><VideoRectangle name=\"box\" toName=\"video\"/></View>",
  "params":{"context":{"event":"prewarm","frame":0,"window":10,"direction":"forward"}}
}' | jq
```

## Known gaps

- **SAM2 video predictor integration** — the prewarm pipeline currently caches
  per-frame image embeddings via the SAM2 image predictor. True video
  tracking with memory propagation is in the next iteration.
- **MPS device** — SAM2 has partial MPS support; fall back to `DEVICE=cpu` on
  Apple Silicon if you hit op-not-implemented errors.
