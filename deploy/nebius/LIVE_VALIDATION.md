# Live validation — September 12, 2026

A bounded smoke test ran in Nebius eu-north1 using synthetic data and disposable
credentials. Label Studio and the HTTPS proxy ran on separate CPU Endpoints in the
same test project; the SAM2 worker ran on one `gpu-l40s-a` / `1gpu-8vcpu-32gb`
Endpoint. Each had a 50 GiB boot disk; the GPU used 16 GiB shared memory. No VM public
IP was configured. This verifies the managed HTTPS path and resource configuration,
not a comprehensive network penetration test or production SLA.

## Tested stack

- Label Studio 1.23.0; Nebius CLI 0.12.265.
- NVIDIA L40S, driver 580.173.02, PyTorch 2.7.1+cu128, CUDA runtime 12.8.
- SAM2.1 tiny and the SDK revisions pinned in the Dockerfile/requirements.
- Built Linux amd64 image manifest:
  `sha256:75b6c8ee5bad07c00b827e322475baddf0b14990b9201e0cd3fac8157d7e4c2f`.
- One sync Gunicorn worker/thread. A test-only startup hook asserted CUDA availability
  and logged device and memory diagnostics. Model/application image contents were
  unchanged; the proxy used the TLS-depth fix in this example.

## Results

| Check | Observed result |
|---|---|
| Native authentication | Missing token, wrong token and Basic auth: 401; correct Bearer token: health 200 |
| Proxy authentication/routing | Missing/wrong Basic: 401; authenticated health/setup: 200; webhook/train/delete: 404 |
| Request admission | 257 KiB body: 413 (the limit was later raised to 16 MiB); malformed JSON: 400; five concurrent predictions: one 200, four 429 |
| First GPU prediction | HTTP 200 in 0.731 s; peak PyTorch allocated memory 583,890,944 bytes (557 MiB), reserved 679,477,248 bytes |
| Warm point predictions | 0.239, 0.189, 0.190 s through the HTTPS proxy |
| Warm box predictions | 0.189, 0.199, 0.211 s through the HTTPS proxy |
| Output | 1,089 nonzero alpha pixels matching the synthetic 33×33 square in a 64×64 image; score 0.98828125 |
| Label Studio | Model connection 201; interactive request 200 in 0.257 s; annotation create 201 and exact API reload 200 |
| Browser | Smart point → visible mask suggestion → accept → submit → reload/reopen; stored brush mask retained score 0.99 |
| Media credential | Expired access token: 401; refresh: 200; authenticated upload: 200 with matching bytes; anonymous upload: 401 |
| Webhook | One automatically created ML webhook was found and disabled |
| Regression tests | 39 passed, including the real NGINX full-chain test; one existing upstream Pydantic deprecation warning |

Timings are individual client-observed samples, not percentiles or throughput claims.
The first prediction is a first-request measurement after readiness, not an end-to-end
cold deployment latency. The GPU Endpoint took about 7 minutes from creation to its
first successful authenticated health response; image pull and startup are included.
Larger images and production annotation accuracy were not evaluated.

## Compatibility fixes and limitations

The managed TLS chain contained two intermediates (Let's Encrypt YR1 and cross-signed
ISRG Root YR) before trusted ISRG Root X1. NGINX's default verification depth rejected
it with 502. `proxy_ssl_verify_depth 3` fixed the connection while retaining CA,
hostname and SNI verification. The regression fixture now reproduces that chain depth
and still checks untrusted roots and wrong hostnames.

Private-registry creation rejected a 149-character digest reference because a generated
label exceeded 64 characters. The run used a short commit-specific tag; the manifest
digest was checked before deployment and again before image cleanup. This is an
observed compatibility workaround, not immutable tag semantics. Public digest-reference
dry runs passed; those dry runs do not establish successful deployment.

Asynchronous CLI create/delete calls returned text despite `--format json`. After
GPU stop, resource state reached `STOPPED` before its operation finished; an immediate
start returned an operation conflict. The documented workflow waits for completion.
The stop operation completed in about 62 seconds. Start was accepted after it
finished, but readiness/reconnection was not established within the remaining
bounded test window; teardown began instead. Stop/start/reconnect is therefore
**incomplete in that initial run**; the later focused follow-up below supersedes
that gap. Saved annotations remained visible in
Label Studio while the GPU was stopped.

During proxy replacement, one early unauthenticated request received a transient 404;
the stable final API suite passed. Treat routing readiness separately from resource state.

Browser box drawing, label-config changes, forced
worker timeouts, image/checkpoint failure injection, cancellation, provider ingress
ceilings and sustained load remain unvalidated. Existing local tests cover additional
input guards and denied-media redaction; they do not replace these live checks.

## Teardown

Cleanup completed at 11:46:35 UTC, within the 60-minute Endpoint lifetime limit.
All 17 recorded Endpoint, auxiliary Job, VM and disk IDs returned NotFound, including
resources retained in the receipt across proxy replacement and GPU restart. All five
compute deletion operations reported completion. Three test registry artifacts were
removed; five test secrets were soft-deleted and returned NotFound (not a physical
purge guarantee). Shared resources and unrelated concurrent experiments were preserved.


## Focused restart and revocation follow-up

A later run on the same date used the same image digest, LS release and L40S preset.
The stop operation completed in 63.37 seconds (observed after 76.81 seconds with
polling). The start operation completed in 436.67 seconds; the existing Label Studio
connection returned a real mask 445.21 seconds after the start request. Its managed
URL was unchanged. That first request took 1.031 seconds; two subsequent requests
took 0.275 and 0.223 seconds and returned 1,089-pixel brush masks. The saved annotation
was identical while stopped and after restart. Worker logs show shutdown, a new
workload image download, and CUDA initialization on the L40S.

For media revocation, a newly uploaded task had not been requested by the backend.
A direct authenticated fetch confirmed its bytes before revocation. Blacklisting the
personal refresh token returned 204. Both an explicit renewal and the backend's next
renewal returned 401 with `Token is blacklisted`; the fresh-image prediction returned
a sanitized 500 at the backend. This establishes live denial of uncached media after
revocation. The backend constructs a model/SDK client for each prediction request,
so a cached SDK access token did not carry over to this next request.

An already-issued access JWT still fetched media immediately after blacklist; its
observed lifetime was 300 seconds. A separate
CPU-only LS 1.23.0 control verified the full expiry boundary: access to media was 200
immediately after blacklist, then 401 after the JWT's 300-second lifetime. A new image
uploaded after expiry was readable with a valid login session (exact bytes matched)
and denied to the expired JWT; the revoked refresh token still returned 401. The LS
session/API remained healthy. GPU denial and JWT expiry were tested in separate
sessions; the GPU denial occurred immediately and was not a post-expiry GPU request.

Label Studio's interactive API returned HTTP 200 without `data` for the failed GPU
prediction. The harness initially treated that status as success and stopped when it
could not decode a mask. Correlated Label Studio logs (revoked SDK renewal: 401) and
worker logs (sanitized prediction failure: 500) establish the actual negative result;
the raw HTTP-200 record is not evidence of a successful mask. Programmatic validation
must inspect the body. The raw failed assertion is retained in the private evidence.

The follow-up's first provisioning attempt also hit a harness race while the proxy's
managed URL was still unassigned. Its compute was removed before retrying with an
explicit URL-readiness wait. Cleanup finished at 18:29:24 UTC,
within the original 60-minute window ending at 18:35:33 UTC. All 21 recorded
Endpoint/VM/disk IDs were absent and all seven Endpoint deletion operations finished.
The single test image artifact was removed and all seven test secrets returned
NotFound after soft deletion. The final Endpoint inventory contained no test IDs;
shared resources were preserved.
