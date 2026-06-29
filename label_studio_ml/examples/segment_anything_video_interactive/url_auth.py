"""Decide whether to attach Label Studio auth to a media URL.

LS-hosted asset URLs (``/data/upload/...``, ``/upload/...``) are guarded and
return **404** (not 401) to an unauthenticated request — which is exactly how a
streaming ffprobe call fails when the auth header is missing. Cloud-storage
*presigned* URLs, on the other hand, carry their auth in the query string;
attaching an ``Authorization: Token`` header to those leaks the LS token to a
third party and can break the signed request.

So: attach auth iff the URL is LS-hosted, never for presigned cloud URLs.

Dependency-free (stdlib only) so it can be unit-tested without torch/cv2/sam2,
mirroring ``ls_auth`` / ``mask_encoding`` / ``frame_resolve``.
"""

from __future__ import annotations

from urllib.parse import urlparse

# Query-string markers of a presigned cloud-storage URL (S3 / GCS / Azure SAS).
# Lower-cased substring match against the full URL.
_PRESIGNED_HINTS = (
    "x-amz-signature", "x-amz-credential",   # AWS S3
    "x-goog-signature", "goog-credential",   # GCS
    "googleaccessid",                         # GCS (legacy)
    "sig=",                                   # Azure SAS
)

# Path prefixes LS uses to serve uploaded/local media. Not used by cloud
# presigned URLs (those address a bucket path), so they're a safe signal that
# an unknown-host URL is LS-hosted.
_LS_ASSET_PATHS = ("/data/", "/upload/")


def _host(url: str) -> str:
    return urlparse(url).netloc.split("@")[-1].split(":")[0].lower()


def _is_presigned_cloud_url(url: str) -> bool:
    low = url.lower()
    return any(hint in low for hint in _PRESIGNED_HINTS)


def _looks_like_ls_asset(url: str) -> bool:
    return urlparse(url).path.lower().startswith(_LS_ASSET_PATHS)


def should_attach_ls_auth(raw_url: str, ls_url: str, has_token: bool) -> bool:
    """True iff the LS Authorization header should be attached to ``raw_url``.

    * Same host as a configured ``ls_url`` → LS-hosted, attach (strongest signal).
    * Presigned cloud-storage URL → never attach (token leak / breaks signature).
    * Otherwise, when ``ls_url`` is unset/mismatched, fall back to the LS asset
      path heuristic so an absolute LS URL still gets authenticated.
    """
    if not has_token:
        return False
    if ls_url and _host(raw_url) == _host(ls_url):
        return True
    if _is_presigned_cloud_url(raw_url):
        return False
    return _looks_like_ls_asset(raw_url)


if __name__ == "__main__":
    # Self-check: the reported 404 case + the leak guard.
    # LS-hosted, ls_url unset (the bug): must attach.
    assert should_attach_ls_auth(
        "https://app.humansignal.com/upload/266842/x.mp4", "", True)
    # Same host as configured ls_url, scheme/port-insensitive: attach.
    assert should_attach_ls_auth(
        "https://app.humansignal.com/data/upload/1/x.mp4",
        "http://app.humansignal.com:80", True)
    # Presigned S3/GCS: never attach (would leak token).
    assert not should_attach_ls_auth(
        "https://bucket.s3.amazonaws.com/x.mp4?X-Amz-Signature=abc", "", True)
    assert not should_attach_ls_auth(
        "https://storage.googleapis.com/b/x.mp4?X-Goog-Signature=abc", "", True)
    # Unknown host, non-LS path, no presign markers: don't blindly attach.
    assert not should_attach_ls_auth("https://example.com/video.mp4", "", True)
    # No token configured: never attach.
    assert not should_attach_ls_auth(
        "https://app.humansignal.com/upload/1/x.mp4", "", False)
    print("url_auth self-check OK")
