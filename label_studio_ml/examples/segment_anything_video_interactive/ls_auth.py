"""Label Studio API authentication helpers for streaming video assets.

Most ML backends can delegate asset auth entirely to ``get_local_path()``. This
backend is different because ffprobe/ffmpeg stream LS-hosted videos directly, so
we need headers outside of the SDK downloader. Keep the token semantics aligned
with the SDK instead of reimplementing JWT parsing / refresh logic here:

* ``TokensClientExt.resolve_x_api_key_header_value`` normalizes refresh JWTs
  (PATs) into access JWTs and leaves legacy/access tokens unchanged.
* ``io._build_headers`` applies the same Authorization scheme selection used by
  ``get_local_path`` (JWT => Bearer, opaque token => Token) and only attaches it
  for matching LS hosts.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Tuple

from label_studio_sdk import LabelStudio
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import _build_headers

from url_auth import should_attach_ls_auth

logger = logging.getLogger(__name__)

_CLIENT_LOCK = threading.Lock()
_SDK_CLIENT_CACHE: Dict[Tuple[str, str], LabelStudio] = {}


def _sdk_client(ls_url: Optional[str], token: Optional[str]) -> Optional[LabelStudio]:
    """Return a cached SDK client so token refresh/caching stays inside SDK.

    The client wrapper owns ``TokensClientExt``; using it here means the
    streaming path shares the SDK's JWT token-type detection, refresh exchange,
    expiry handling, and locking instead of duplicating that code locally.
    """
    if not ls_url or not token:
        return None
    key = (ls_url.rstrip("/"), token)
    with _CLIENT_LOCK:
        client = _SDK_CLIENT_CACHE.get(key)
        if client is None:
            client = LabelStudio(base_url=key[0], api_key=token)
            _SDK_CLIENT_CACHE[key] = client
        return client


def _resolve_token_with_sdk(ls_url: Optional[str], token: str) -> str:
    """Normalize a token using the SDK token client when available.

    Legacy tokens and access JWTs are returned unchanged. Refresh JWTs are
    exchanged for access JWTs by the SDK. If SDK normalization is unavailable or
    fails, return the original token so callers preserve the previous fallback
    behavior and surface the eventual 401 from LS rather than hiding it here.
    """
    try:
        client = _sdk_client(ls_url, token)
        if client is None:
            return token
        tokens_client = client._client_wrapper._tokens_client
        return tokens_client.resolve_x_api_key_header_value(token)
    except Exception as e:
        logger.warning("SDK token normalization failed (host=%s): %s", ls_url, e)
        return token


def ls_token_for_sdk(ls_url: Optional[str], token: Optional[str]) -> Optional[str]:
    """Return the token value to hand to ``get_local_path``.

    ``get_local_path`` already knows how to choose ``Token`` vs ``Bearer`` for a
    token value, but older downloader code does not exchange refresh JWTs first.
    Passing the SDK-normalized value lets download fallback and streaming use the
    same credential.
    """
    if not token:
        return None
    return _resolve_token_with_sdk(ls_url, token.strip())


def ls_auth_headers(
    ls_url: Optional[str],
    token: Optional[str],
    target_url: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Build SDK-compatible auth headers for an LS-hosted streaming URL.

    ``target_url`` is optional for existing callers, but should be supplied for
    streaming so SDK header construction can enforce the same host match as
    ``get_local_path``. Host/token leakage prevention still lives in
    ``url_auth.should_attach_ls_auth`` before this helper is called.
    """
    wire_token = ls_token_for_sdk(ls_url, token)
    if not wire_token:
        return None

    if ls_url and target_url and not should_attach_ls_auth(target_url, ls_url, True):
        headers = _build_headers(target_url, ls_url, wire_token)
        return headers or None

    # `should_attach_ls_auth` is intentionally scheme/port-insensitive so the
    # operator can configure `LABEL_STUDIO_URL=http://host:80` while LS returns
    # `https://host/...` asset URLs. The SDK's `_build_headers` uses exact
    # netloc equality, so after our guard says the target is safe, build against
    # `ls_url` itself to reuse the SDK's Token-vs-Bearer choice without losing
    # auth on benign scheme/port differences.
    if ls_url:
        headers = _build_headers(ls_url, ls_url, wire_token)
        return headers or None

    return None
