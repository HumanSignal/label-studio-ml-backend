# SAM2 image backend on Nebius Serverless

Run the existing [SAM2 image backend](../../label_studio_ml/examples/segment_anything_2_image)
on one Nebius Serverless GPU Endpoint. Label Studio and its database/storage stay on
your existing infrastructure. This example supports one project, authenticated Label
Studio uploads and interactive point/box prompts that return brush masks.

**Validation (2026-09-12):** real SAM2.1 tiny inference passed on a Nebius L40S
through native token authentication and the external NGINX proxy. Label Studio 1.23.0
connected, generated point/box masks through its API, and saved/reloaded annotations.
A browser smart-point session also generated, accepted, saved and reopened a mask.
The image builds for Linux amd64 and 39 API/NGINX tests pass. See
[LIVE_VALIDATION.md](LIVE_VALIDATION.md) for measurements, the TLS fix, provider
limitations and untested failure cases. This is a bounded smoke test for trusted
annotation workloads, not a production inference SLA.

## Architecture and prerequisites

```text
Label Studio -- HTTPS + Basic --> your external NGINX proxy
             -- HTTPS + Bearer --> Nebius managed ingress --> SAM2:9090
SAM2 -- separate Label Studio API credential --> Label Studio uploaded media
```

Label Studio's standard ML connection exposes Basic authentication; Nebius native
Endpoint authentication expects Bearer. The small proxy translates between these
schemes, without modifying Label Studio core. The GPU backend has Basic auth disabled
and must be reachable only through the native-token ingress and trusted private networks.
Do not put the Endpoint token in the Label Studio Basic-password field.

You need:

- An existing Label Studio instance with a reachable HTTPS origin, an annotation
  project, and a dedicated media credential with the least access your edition supports.
  The project's numeric ID is configured explicitly; use this Endpoint for one project.
- An external host with Docker Compose and an existing HTTPS ingress/certificate for
  the proxy. Its hostname must be accepted by Label Studio's outbound URL policy.
  Do not disable SSRF protection globally to make a local proxy URL work.
- A configured Nebius CLI profile and an explicit project, subnet, GPU quota and regular
  one-GPU preset. The subnet needs outbound access to Label Studio and the registry.
- A Linux amd64 image builder and a registry you are authorized to publish to. Allow
  several GB for the CUDA runtime. The live smoke test used one L40S; measure GPU memory/latency
  for your own image sizes and workload.
- An approved cloud-spend/time limit and an operator responsible for the cleanup receipt.

Control-plane identity, Endpoint Bearer token, proxy Basic password and Label Studio
media credential are four separate credentials. None belongs in the Docker build,
source tree or PR evidence. Follow the current [Endpoint management guide](https://docs.nebius.com/serverless/endpoints/manage)
and [lifecycle documentation](https://docs.nebius.com/serverless/lifecycle).

## 1. Build and record the image

Run from this repository root:

```sh
docker build --platform linux/amd64 -f deploy/nebius/Dockerfile.sam2 \
  -t label-studio-nebius:sam2 .
```

The Dockerfile pins a CUDA 12.8 / PyTorch 2.7.1 / TorchVision 0.22.1 runtime digest,
SAM2 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`, SDK commit
`58fe284b668ef767698917093032e87cdcc343ea`, Python dependencies, and the SAM2.1 tiny
checkpoint checksum. It copies the model from this checkout. Record this checkout's
commit and dirty state, then record the final registry digest after your authorized
image push. Prefer deployment by digest where the provider accepts it. OS package mirrors can change;
this is a pinned application stack, not a bit-for-bit reproducible OS build.

**Observed private-registry limitation (CLI 0.12.265, eu-north1, 2026-09-12):** both
validation and creation rejected a fully qualified Nebius registry digest reference
with `Labels: label value length (149) exceeds maximum value (64)`. A public registry
digest reference passed. The live test used a short, commit-specific private tag under
64 characters, with its digest checked before and after deployment. Use a unique tag
that no other process will overwrite and record the resolved digest; this workaround
does not make a tag immutable. Treat long-lived deployment by immutable digest as an
open provider-compatibility check. Use the Docker namespace shown by your registry;
it may differ from its API resource ID.

The optional SAM2 connected-component CUDA extension is disabled to avoid requiring
NVCC during the build. Model inference still uses CUDA. This can affect optional mask
postprocessing; validate output for your use case. The checkpoint is baked into the
image; do not mount storage over `/sam2/checkpoints`.

The runtime uses one non-root Gunicorn **sync** worker, one thread, a 75-second worker
timeout and no preload. SAM2's global predictor mutates image state, so increasing
threads without changing that implementation is unsafe. Request-driven cold starts,
autoscaling, training, video and automatic batch pre-annotation are outside this example.

## 2. Create runtime secrets and the Endpoint

Use your secret-management workflow to create:

| Secret | Payload key | Used by |
|---|---|---|
| Random 64-hex-character Endpoint token (`openssl rand -hex 32`) | `AUTH_TOKEN` | Native ingress and external proxy |
| Label Studio personal refresh token (or legacy key if enabled) | `LABEL_STUDIO_API_KEY` | SAM2's Label Studio SDK/media helper |
| Optional registry credentials | CLI-specific registry keys | Private image pull |

Keep a protected copy of the Endpoint token for the proxy. Use the current CLI's
SecretStash selector syntax; raw protobuf field names still use MysteryBox terminology.
Check registry-secret key mapping against your CLI version. The live run used CLI `0.12.265`, the environment/secret options below and a
50 GiB boot disk. Adjust the example disk size to your image and cache requirements.

Substitute the placeholders; this command creates billable resources:

```sh
nebius --profile <profile> --retries 1 ai endpoint create \
  --parent-id <project-id> --name <unique-display-name> \
  --image <tested-image-reference> \
  --platform <gpu-platform> --preset <one-gpu-preset> \
  --subnet-id <subnet-with-required-egress> \
  --container-port 9090 --working-dir /sam2 \
  --disk-size 250Gi --shm-size 16Gi \
  --auth token --token-secret <endpoint-token-secret> \
  --env LABEL_STUDIO_URL=https://<label-studio-host> \
  --env LABEL_STUDIO_PROJECT_ID=<numeric-label-studio-project-id> \
  --env-secret LABEL_STUDIO_API_KEY=<media-secret> \
  --async > create.receipt.txt
```

Add `--registry-secret <selector>` if required. This example intentionally omits a
public workload IP and preemptibility; managed HTTPS exposure is separate from the
workload IP. Verify the subnet has the required egress and that raw port 9090 is not
publicly reachable. Do not expose a second port: native-token auth requires one HTTP port.

CLI 0.12.265 returned a text `Endpoint ID:` receipt for asynchronous creation even with
`--format json`. Preserve that text, extract the returned ID, and use `endpoint get
--id <id> --format json` for a structured resource receipt. Do not assume the create
output is JSON. `--dry-run` is only request validation, not a live compatibility test.

Before submission, record project, display name, image digest and creation time. Keep
returned operation/resource IDs in a non-secret receipt. If the response is ambiguous,
inspect operation status and **all pages** of matching resources; do not automatically
retry creation or assume display names are unique. A CLI timeout does not cancel the
Endpoint. Track operation completion as well as resource state with `nebius ai endpoint
operation get` and `nebius ai endpoint get`; consult their current `--help` for ID arguments.

Wait for creation to complete. Copy the managed HTTPS URL from `status.public_endpoints`.
The proxy takes its **hostname only**, without scheme/path. Only use this recipe with
an origin-style managed URL. Do not connect Label Studio until authenticated `/health`
and `/setup` succeed; `RUNNING` alone is not a successful prediction check.

## 3. Configure the external proxy

On your existing proxy host, create `deploy/nebius/proxy/secrets/` with mode 0700.
Place the Endpoint token in `secrets/endpoint_token` and an NGINX-compatible htpasswd
entry in `secrets/htpasswd`, both mode 0600. Use an interactive `htpasswd` tool to
create the Basic password entry; avoid passwords in shell history or process arguments.
These paths are ignored by Git and excluded from the model build context.

```sh
cd deploy/nebius/proxy
export NEBIUS_ENDPOINT_HOST=<managed-endpoint-hostname>
docker compose up -d
```

Forward an existing **HTTPS** virtual host to `http://127.0.0.1:9091`. Keep that loopback
binding; do not expose plaintext Basic auth on a public interface. Configure your
outer ingress for requests up to 16 MiB and a timeout greater than 100 seconds. Label Studio
sends the task's saved annotations, drafts and predictions (brush masks) with each request.
Do not log Authorization headers, request bodies or response bodies at either ingress.

The proxy authenticates Basic, replaces Authorization with the Endpoint Bearer token,
removes Cookie, checks upstream TLS certificates and SNI with verification depth 3
(to support the managed ingress's two-intermediate chain), and forwards only `/health`,
`/setup` and `/predict`. Training/webhook/management routes return 404. It permits one
active authenticated prediction and returns 429 for overlapping predictions. It does not
retry or cache inference responses. The startup script accepts only a DNS-style hostname
and a 64-hex Endpoint token. Rendered configuration contains the token; do not publish `nginx -T` output.

After rotating a secret, recreate the proxy container to reload the file. Endpoint-side
secret rotation/restart behavior requires validation; changing a file on the proxy does
not rotate the native ingress token. Do not keep the backend's `BASIC_AUTH_USER/PASS` set.

## 4. Connect Label Studio and annotate

1. Configure the project using [fixtures/label-config.xml](fixtures/label-config.xml),
   or adapt the existing SAM2 labeling example. Enable interactive predictions.
2. Add a model using the proxy's HTTPS URL and its dedicated Basic username/password.
   Health is `GET /health`; setup is `POST /setup` with `project` and `schema`.
3. **Check the project's Webhooks settings and disable any ML webhook** created for
   this connection. The automatic webhook path does not inherit the model connection's Basic credentials. This example performs
   no training and deliberately does not expose `/webhook`.
4. Upload a trusted small PNG/JPEG through Label Studio and enable **Auto-Annotation**.
   In the **Auto-Detect** tool group, select the smart keypoint or smart rectangle
   variant, then select its matching label (point: Object shortcut 2; box: Object
   shortcut 3 in the fixture). Scroll to reveal the labels if needed. Place a positive
   point or draw a box. The regular drawing tools create manual regions.
5. Confirm the mask suggestion on the same image and accept it with the checkmark
   when Auto-Accept Suggestions is off. Submit, reload and reopen the completed task
   to verify Label Studio remains the system of record. For programmatic checks,
   inspect the prediction body as well as HTTP status: Label Studio 1.23.0 returned
   HTTP 200 without `data` when the backend rejected revoked credentials. Require
   the expected brush-mask result; HTTP 200 alone does not establish a prediction.

The worker fetches media using `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY`.
Credentials in `/setup` do not configure that environment. Only relative or same-origin
HTTPS `/data/upload/` and `/storage-data/uploaded/` URLs are accepted. This recipe does
not support arbitrary URLs, local host paths or cloud-storage URIs. Label Studio
1.23.0 disables legacy tokens by default; use a personal refresh token. The pinned SDK
exchanged that token successfully in live validation. An expired access token returned
401 and refresh restored authenticated media access. Live refresh-token revocation
also denied a newly uploaded, uncached image: the SDK's renewal returned 401 and the
backend returned its sanitized prediction error. This backend creates a model/SDK
client for each prediction request, so that denial was immediate. An already-issued
access JWT remained valid immediately after revocation; its observed lifetime was
300 seconds. A separate CPU-only control confirmed that JWT was denied after expiry
while a valid session read a newly uploaded image. Test your deployed release/edition
and token-lifetime configuration.

[fixtures/predict.json](fixtures/predict.json) illustrates the HTTP payload. Replace its
project ID, task ID, image URL and dimensions with actual task values before using it.
`params.context.result` contains point/box information; the synchronous response contains
`results`, each with a Label Studio `result` array, model version and score. An empty
interactive context returns no predictions. Exactly one task is required.

Use inputs of at most 4,194,304 pixels without EXIF rotation; the backend rejects prompts
whose dimensions differ from the stored pixels, which includes rotated phone photos.
Enforce a small upload-byte limit on Label Studio (for example 10 MiB). The current SDK
downloader has no explicit request timeout or streaming byte cap. The worker watchdog is a final failure bound, not an efficient
large-file downloader. This is why the recipe is limited to trusted uploads.

### Deadlines, failure handling and persistence

- Suggested warm-session LS settings: `ML_CONNECTION_TIMEOUT=5`, `ML_TIMEOUT_HEALTH=10`,
  `ML_TIMEOUT_SETUP=15`, `ML_TIMEOUT_PREDICT=100`. Restart the LS process after changing
  its environment. The model connection's `timeout` field does not replace every
  route-specific timeout. Hosted Label Studio may require operator assistance.
- The proxy's prediction read timeout is 90 seconds; Gunicorn's sync worker timeout is
  75 seconds. Read timeouts are inactivity limits, not a total deadline. Confirm Nebius's
  managed-ingress ceiling; no numerical provider request/body limit is asserted here.
- Load the model before annotation sessions. A watchdog kill reloads the worker/model;
  it is not graceful cancellation. Disconnecting a client does not prove GPU work stopped.
  Do not automatically retry a timed-out prediction or hold bulk tasks in an HTTP request.
- Local SQLite metadata and media caches are disposable. Startup/stop/start/replacement
  must tolerate losing them. Do not mount an S3 bucket as a SQLite database filesystem.
  The model checkpoint is immutable and there is no trained state to export.
- Cached media can be reused without rechecking access. Purge the disposable cache
  when project access/credentials change; revocation does not erase already cached bytes.
  The project guard is not a substitute for media permissions or tenant isolation.
- Logs can contain diagnostics from upstream model code. Avoid DEBUG logging, redact
  logs before sharing, and never publish private task data. The wrapper suppresses
  media exception details in prediction responses to avoid exposing signed URLs.

## 5. Stop and remove resources

Use the **recorded Endpoint ID**, not a name inferred later:

```sh
nebius --profile <profile> ai endpoint stop --id <endpoint-id>
# To resume later, wait for stop operation completion before starting:
nebius --profile <profile> ai endpoint start --id <endpoint-id>
# When the integration is retired:
nebius --profile <profile> ai endpoint delete --id <endpoint-id>
```

For an asynchronous stop/start, preserve the returned operation ID and poll
`nebius ai endpoint operation get --id <operation-id> --format json`. In CLI 0.12.265,
completed operation receipts contain `finished_at` and `status`; they do not use a
`done` boolean. Check for failure as well as completion. Wait for the stop operation
to finish before requesting start, then wait for the start operation to finish and
verify authenticated health plus a real Label Studio prediction. Resource state can
advance ahead of operation completion. Confirm existing annotations are unchanged.

For retirement, first stop annotation traffic, disable/remove the exact webhook and
model connection, and run `docker compose down` on the proxy host. Then delete the
Endpoint and wait for its operation to complete. Verify the Endpoint and its recorded
underlying VM/boot disk are absent. Do not claim cleanup from a disconnected LS model,
a terminated terminal, or the first `STOPPED` state alone.

Nebius documents removal of the managed VM/boot disk on Endpoint deletion. Separately
owned storage, registry images and secrets remain your responsibility. Remove only
resources created for this example and no longer referenced. Stopped compute is not
charged, but attached storage may still be billed. Leave the subnet and shared resources
intact. If the launcher disappears, another operator must finish the saved receipt;
this example does not install an orphan-resource watchdog.

## Tests and live-validation checklist

Local API tests and real NGINX tests need Python 3.11, Docker and OpenSSL. From the root:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r deploy/nebius/requirements.lock pytest==9.1.1
pip install --no-deps -e .
PYTHONPATH=deploy/nebius MODEL_DIR=/tmp/ls-nebius-test RUN_PROXY_TESTS=1 \
  python -m pytest deploy/nebius/tests/test_app.py deploy/nebius/tests/test_proxy.py -q
```

Proxy tests create temporary local Docker containers/network and test credentials,
then remove their containers/network. They use a fake TLS/Bearer ingress, not Nebius.
The dedicated CI workflow runs these tests without provider credentials or GPU use.

An opt-in **real-model CPU smoke test** runs inside the built GPU image without network:

```sh
docker run --rm --platform linux/amd64 --network none \
  -e DEVICE=cpu -e LABEL_STUDIO_URL=https://label.example \
  -e LABEL_STUDIO_PROJECT_ID=1 -e LABEL_STUDIO_API_KEY=cpu-test-credential \
  -e PYTHONPATH=/app \
  -v "$PWD/deploy/nebius/tests:/tests:ro" \
  -v "$PWD/deploy/nebius/fixtures:/fixtures:ro" \
  label-studio-nebius:sam2 python /tests/smoke_cpu.py
```

This checks the real model, upload-helper auth header, HTTP envelope and decoded brush
mask, but mocks the media HTTP response. It does not prove CUDA, live token renewal or
an annotator's complete UI workflow.

Use the dated [live results](LIVE_VALIDATION.md) as a smoke-test baseline. For your
deployment, record and complete the remaining cases:

- Exact LS release, CLI version, image digest, checkpoint hash, GPU/driver and project/preset.
- Native ingress rejection for missing/wrong token; no public raw-port auth bypass.
- Real LS health/setup and positive-point/box annotation through the external HTTPS proxy.
- Real authenticated upload, renewal/expiry/revocation with a fresh cache, denied media,
  label-config changes, invalid context and overload. A success response must display
  a mask on the correct task and persist correctly in Label Studio.
- Cold/warm timings, memory and request/response size/deadline behavior. Do not turn a
  slow `/predict` into a Nebius Job and return a Job ID as a prediction.
- Missing checkpoint/image-pull failure, worker reload, stop/start/reconnect and complete
  teardown, with independent Endpoint/VM/disk absence checks.
- An optional separately authenticated webhook probe, followed by disabling/removing it.
  A 201 from SAM2's inherited no-op `fit` is not evidence of training.

For larger batch processing, design a separate durable Jobs bridge that imports completed
predictions into Label Studio; it is not part of this synchronous deployment example.
