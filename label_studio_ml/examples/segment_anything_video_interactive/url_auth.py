"""Decide whether to attach Label Studio auth to a media URL.

LS-hosted asset URLs (``/data/upload/...``, ``/upload/...``) are guarded and
return **404** (not 401) to an unauthenticated request — which is exactly how a
streaming ffprobe call fails when the auth header is missing.

The header must only ever go to the LS host: task data can carry arbitrary
external media URLs (and presigned cloud-storage URLs carry their own auth in
the query string), so attaching ``Authorization: Token`` to anything else would
leak the LS credential to a third party. We therefore attach auth iff the URL's
host matches a *known* LS host — never on a path heuristic for an unknown host.

Dependency-free (stdlib only) so it can be unit-tested without torch/cv2/sam2,
mirroring ``ls_auth`` / ``mask_encoding`` / ``frame_resolve``.
"""

from __future__ import annotations

from urllib.parse import urlparse


def _host(url: str) -> str:
    return urlparse(url).netloc.split("@")[-1].split(":")[0].lower()


def should_attach_ls_auth(raw_url: str, ls_url: str, has_token: bool) -> bool:
    """True iff the LS Authorization header should be attached to ``raw_url``.

    Attaches only when we have a token AND ``raw_url``'s host matches the known
    LS host (``ls_url``), compared scheme/port-insensitively. When ``ls_url`` is
    unknown we cannot safely identify an LS asset, so we do not attach — the
    caller falls back to an authenticated SDK download instead. This never sends
    the token to a non-LS host (presigned cloud URLs, external media, etc.).
    """
    if not has_token or not ls_url:
        return False
    return _host(raw_url) == _host(ls_url)


if __name__ == "__main__":
    # Self-check: the reported 404 case + the token-leak guard.
    # Same host as configured ls_url, scheme/port-insensitive: attach.
    assert should_attach_ls_auth(
        "https://app.humansignal.com/upload/266842/x.mp4",
        "http://app.humansignal.com:80", True)
    assert should_attach_ls_auth(
        "https://app.humansignal.com/data/upload/1/x.mp4",
        "https://app.humansignal.com", True)
    # Arbitrary third-party host with an LS-looking path: must NOT attach.
    assert not should_attach_ls_auth(
        "https://evil.example/upload/video.mp4", "https://app.humansignal.com", True)
    # Presigned cloud storage (different host): never attach.
    assert not should_attach_ls_auth(
        "https://bucket.s3.amazonaws.com/x.mp4?X-Amz-Signature=abc",
        "https://app.humansignal.com", True)
    # LS host unknown: cannot identify an LS asset, don't attach.
    assert not should_attach_ls_auth(
        "https://app.humansignal.com/upload/1/x.mp4", "", True)
    # No token configured: never attach.
    assert not should_attach_ls_auth(
        "https://app.humansignal.com/upload/1/x.mp4", "https://app.humansignal.com", False)
    print("url_auth self-check OK")
