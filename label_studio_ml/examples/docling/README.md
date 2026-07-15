# Docling Serve backend (IBM Docling Workbench)

## What this is for

This backend connects Label Studio to **IBM Docling SaaS** using the Python **`DoclingServiceClient`** from the **`docling`** package (`from docling.service_client import DoclingServiceClient`). **Conversion runs on Docling’s servers**, not inside this container. For each task it resolves the file (usually via Label Studio–hosted storage), calls **`client.convert(source=…)`** with a local **`Path`** or an **`https://` URL string**, then maps **`result.document`** into Label Studio predictions for the annotator.

By default predictions are emitted as **canonical Label Studio result envelopes** (`type: "rectanglelabels"` / `type: "polygonlabels"`) matching the **HumanSignal Interfaces** Docling annotator at `docling-ls-implementation/docling_interface.jsx`. Set `DOCLING_RESULT_FORMAT=reactcode` if the project still uses the legacy ReactCode XML labeling config.

Use the **exact service URL** your tenant gives you (Integrate / Python snippet), including the path segment ending in **`/v1`**—for example  
`https://api.aws-c1.dcls.saas.ibm.com/<instance>/v1`.

The **`docling`** `DoclingServiceClient` builds paths like **`/v1/convert/...`** on top of its `url=` argument. IBM’s URL already ends with **`/v1`**, which would otherwise produce **`…/v1/v1/…`** requests (404/400). This example **strips one trailing `/v1`** from **`DOCLING_SERVICE_URL`** before creating the client—keep pasting the Workbench value unchanged.

Typical workflow:

1. Tasks include a **file URL** (PDF, image, etc.)—often an upload or storage URL managed by Label Studio.
2. Annotators run predictions (or batch predict); this ML backend fetches the file (unless you use remote-URL-only mode), calls **`DoclingServiceClient.convert`**, and returns layout as reactcode regions.
3. Reviewers adjust regions or labels on top of Docling’s structure.

You need the **full SaaS service URL** and API key from Workbench. Separately, the backend must often **download task files** through Label Studio when URLs point at your instance—see **Label Studio URL and API key** below.

## Label Studio URL and API key

Set **`LABEL_STUDIO_URL`** and **`LABEL_STUDIO_API_KEY`** in `docker-compose.yml` (or your shell) whenever tasks reference **files hosted by Label Studio**—uploads, cloud storage integrations, or other URLs that Label Studio resolves for the ML backend.

By default it downloads to a cache path and passes a **`Path`** into **`convert`**. Set **`DOCLING_CONVERT_REMOTE_URL_ONLY=true`** to pass the task’s **`https://` URL** directly to SaaS (works only for URLs the Docling service can fetch without Label Studio auth).

Practical notes:

- **`LABEL_STUDIO_URL`** must be reachable **from where the ML backend runs**. From Docker on your laptop, **`http://localhost:8080`** usually does **not** work inside the container; use your machine’s hostname/IP, **`http://host.docker.internal:8080`** (Docker Desktop), or another URL the container can route to. This compose file includes `extra_hosts` for `host.docker.internal` on macOS/Linux-friendly setups.
- **`LABEL_STUDIO_API_KEY`** should be a **Personal Access Token** (or equivalent) for a user that can read the project’s tasks and attachments.

Always include **`http://` or `https://`** in `LABEL_STUDIO_URL`. More background is in the repository [README](../../../README.md) under allowing the ML backend to access Label Studio data.

## Prerequisites

1. **`DOCLING_SERVICE_URL`** — full URL ending in **`/v1`** from IBM Docling Workbench (same as `DoclingServiceClient(url=…)`).
2. **`DOCLING_SERVE_API_KEY`** — API key for `X-Api-Key` (name kept for backward compatibility).
3. **`LABEL_STUDIO_URL`** / **`LABEL_STUDIO_API_KEY`** when tasks use Label Studio–hosted files (typical for uploads).

## Quick start (Docker)

```bash
cd label_studio_ml/examples/docling
# Set DOCLING_SERVICE_URL, DOCLING_SERVE_API_KEY, LABEL_STUDIO_URL, LABEL_STUDIO_API_KEY in docker-compose.yml
docker compose up --build
```

The ML backend listens on **`http://localhost:9090`**. Register that URL in your Label Studio project’s machine learning settings.

## Docling SaaS configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `DOCLING_SERVICE_URL` | Yes | Full **`DoclingServiceClient`** URL including path to **`/v1`** (fallback env name: `DOCLING_SERVE_URL`). |
| `DOCLING_SERVE_API_KEY` | Often | API key (`X-Api-Key`). Alias: `DOCLING_API_KEY`. |
| `DOCLING_CONVERT_REMOTE_URL_ONLY` | No | If `true`, pass the task **`https://` URL** as `convert(source=url)` instead of downloading via Label Studio first. |
| `DOCLING_CONVERT_SOURCE_HEADERS_JSON` | No | Extra HTTP headers (JSON object) merged into **`convert`** when using remote URLs / headers the client supports. |
| `DOCLING_SERVE_TIMEOUT` | No | Job / read timeout in seconds (default `600`). |
| `DOCLING_HTTP_CONNECT_TIMEOUT` | No | Connect timeout (default `30`). |

Optional tuning: `DOCLING_PAGE_NO`, `DOCLING_PREDICT_READING_ORDER`, `DOCLING_READING_ORDER_LEVEL`, `DOCLING_CONTENT_LAYERS`, `DOCLING_REACTCODE_FROM_NAME`, `DOCLING_REACTCODE_TO_NAME`, `DOCLING_TASK_DATA_KEY`, `DOCLING_RESULT_FORMAT`.

`DOCLING_RESULT_FORMAT` controls the prediction shape. **You usually don't need to set it** — the backend auto-detects from the project's labeling config:

- A `<ReactCode>` tag in the config → legacy `reactcode` envelope (`type: "reactcode"`, `value: {"reactcode": <payload>}`).
- Anything else (including the near-empty `<View></View>` used by HumanSignal Interfaces projects) → canonical Label Studio envelopes (`type: "rectanglelabels"` / `type: "polygonlabels"`), matching `docling-ls-implementation/docling_interface.jsx`'s `parseResults`.

Explicit values override detection: set `DOCLING_RESULT_FORMAT=canonical` or `DOCLING_RESULT_FORMAT=reactcode` to force a format. The chosen format is logged at the start of every predict batch with its source (env var vs. auto-detected) so you can confirm what's being sent.

The **`docling`** PyPI package (**≥2.90**) provides **`DoclingServiceClient`**; behavior follows **your SaaS tenant**, not necessarily open-source Docling docs.

## Label Studio configuration

| Variable | Description |
|----------|-------------|
| `LABEL_STUDIO_URL` | Base URL of Label Studio, reachable from this backend (see above). |
| `LABEL_STUDIO_API_KEY` | Token so the backend can download task attachments when needed. |

Predictions default to **canonical** shape — `type: "rectanglelabels"` for layout regions and `type: "polygonlabels"` for reading-order polylines, with percent coordinates — matching the HumanSignal Interfaces Docling annotator (`docling-ls-implementation/docling_interface.jsx`). Switch to `DOCLING_RESULT_FORMAT=reactcode` for legacy ReactCode XML projects (see `docling_labeling_config.xml` in this folder).

## Running locally (without Docker)

```bash
pip install -r requirements-base.txt -r requirements.txt
export DOCLING_SERVICE_URL=https://api.aws-c1.dcls.saas.ibm.com/your-instance/v1
export DOCLING_SERVE_API_KEY=your-api-key
export LABEL_STUDIO_URL=http://host.docker.internal:8080
export LABEL_STUDIO_API_KEY=your-label-studio-token
python _wsgi.py -p 9090
```

Adjust `LABEL_STUDIO_URL` if Label Studio runs on the same machine without Docker (for example `http://127.0.0.1:8080`).

## Validate

```bash
curl http://localhost:9090/
```

Expected: `{"status":"UP"}`.

## Troubleshooting

### Wrong SaaS URL

**`DOCLING_SERVICE_URL`** must match the URL Workbench gives you (through **`/v1`**). The backend normalizes it so routes are not doubled—see the note above if you see **`/v1/v1/`** in logs.

### No predictions / “nothing happens” (no errors in the UI)

Label Studio often shows **no message** when the ML backend returns **empty `results`** (HTTP 200 with an empty list). Check **Docker logs** for this container:

```bash
docker compose logs -f docling
```

You should see a line like **`Docling predict: N task(s)`** whenever you run predictions. If you see **`Docling produced zero predictions`**, scroll up in the same log for **`No file URL found`** or Docling **`API error`** lines.

Common fixes:

1. **Placeholder URL** — Replace **`YOUR_INSTANCE_SEGMENT`** in **`DOCLING_SERVICE_URL`** with the real path from Workbench.
2. **Wrong task field** — Tasks must expose a **file URL** under the key your labeling config expects. The default is **`image`** (matches `docling_interface.jsx`); the backend then falls back through `image`, `url`, `ocr`, `$undefined$`, `$undefined`, `undefined`, `pdf`, `document`, `file`. Override with **`DOCLING_TASK_DATA_KEY`** if needed.
3. **`LOG_LEVEL`** — Defaults to **`INFO`** in `_wsgi.py` when unset.
4. **Upload / `/storage-data/` URLs** — `model.py` downloads via **`label_studio_sdk`** using **`LABEL_STUDIO_URL`** (same **scheme + host + port** as in your browser; wrong host breaks auth headers), **`LABEL_STUDIO_API_KEY`**, and network reachability from this container (`host.docker.internal` instead of `localhost` on Docker Desktop). Self-signed HTTPS: set **`VERIFY_SSL=false`** on the ML backend. Logs now include **HTTP status / snippet** when the download fails.

Sanity checks:

```bash
curl -s http://localhost:9090/health
curl -s http://localhost:9090/
```

Both should return JSON including **`"status":"UP"`**.

### Empty or tiny downloaded files

Check **`LABEL_STUDIO_URL`** / **`LABEL_STUDIO_API_KEY`** and logs for `Docling task … local_path=… size=…`. A size of **0** or failed stat (`-1` in logs) usually means the file did not download correctly before conversion.

## Layout of this example

Like other backends under `label_studio_ml/examples/` (for example `easyocr/`), this directory includes `_wsgi.py`, `model.py`, `requirements-base.txt`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, and tests. **`docker-compose.yml`** bind-mounts `./data/server` and `./data/.file-cache` for runtime caches; Docker creates those paths on the host when you first run Compose—they are not checked into git (see `.gitignore`).
