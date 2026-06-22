"""Label Studio API authentication helpers.

Label Studio Enterprise has two token types and they authenticate differently:

  * Legacy token → an opaque ~40-char string, sent as ``Token <token>``
    (DRF ``TokenAuthentication``).
  * Personal Access Token (PAT) → a JWT *refresh* token. It can't be used
    directly; it must be exchanged at ``/api/token/refresh`` for a short-lived
    access JWT, which is then sent as ``Bearer <access>``.

``ls_auth_headers`` hides that difference: hand it the LS host + whatever token
the operator configured, and it returns the right ``Authorization`` header,
minting and caching access tokens from a PAT as needed.

Dependency-free apart from ``requests`` so it can be unit-tested without torch
/ cv2 / sam2 (mirrors ``control_detect`` / ``mask_encoding``).
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Access tokens minted from a PAT, cached per (host, pat) until they near
# expiry.
_PAT_LOCK = threading.Lock()
_PAT_ACCESS_CACHE: Dict[Tuple[str, str], Tuple[str, float]] = {}
# Refresh this many seconds before the access token's `exp` to avoid racing
# expiry mid-request.
_PAT_REFRESH_MARGIN = 60.0
# If an access token's `exp` can't be read, cache it this long so we don't
# hammer the refresh endpoint.
_PAT_FALLBACK_TTL = 300.0
# Timeout for the token-refresh HTTP call.
_EXCHANGE_TIMEOUT = 10.0


def _looks_like_jwt(token: str) -> bool:
    """A PAT is a JWT: three base64url segments joined by dots, whose header
    base64-encodes to a leading ``eyJ``. Legacy LS tokens are opaque strings
    with no dots, so this cleanly distinguishes the two."""
    return token.count(".") == 2 and token.startswith("eyJ")


def _jwt_exp(token: str) -> Optional[float]:
    """Best-effort read of a JWT's ``exp`` claim (epoch seconds) WITHOUT
    verifying the signature — we only need the expiry to schedule a refresh.
    Returns None if the token can't be parsed."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)  # restore base64 padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        return float(exp) if exp is not None else None
    except Exception:
        return None


def _exchange_pat(ls_url: str, pat: str) -> Optional[str]:
    """Exchange a PAT (JWT refresh token) for a short-lived access JWT via
    LSE's ``/api/token/refresh``. Returns the access token, or None on
    failure."""
    try:
        resp = requests.post(
            f"{ls_url}/api/token/refresh", json={"refresh": pat},
            timeout=_EXCHANGE_TIMEOUT,
        )
        resp.raise_for_status()
        access = (resp.json() or {}).get("access")
        if not access:
            logger.warning("PAT exchange returned no access token (host=%s)", ls_url)
            return None
        return access
    except Exception as e:
        logger.warning("PAT exchange failed (host=%s): %s", ls_url, e)
        return None


def _access_token_for_pat(ls_url: str, pat: str) -> Optional[str]:
    """Return a valid access JWT for ``pat``, minting or refreshing as needed.
    Cached per (host, pat) and reused until shortly before it expires."""
    key = (ls_url, pat)
    now = time.time()
    with _PAT_LOCK:
        cached = _PAT_ACCESS_CACHE.get(key)
        if cached is not None and now < cached[1] - _PAT_REFRESH_MARGIN:
            return cached[0]
    # Exchange outside the lock (network I/O); a rare concurrent double-exchange
    # is harmless — last writer wins and both tokens are valid.
    access = _exchange_pat(ls_url, pat)
    if access is None:
        return None
    exp = _jwt_exp(access)
    expires_at = exp if exp is not None else now + _PAT_FALLBACK_TTL
    with _PAT_LOCK:
        _PAT_ACCESS_CACHE[key] = (access, expires_at)
    return access


def ls_auth_headers(ls_url: Optional[str], token: Optional[str]) -> Optional[Dict[str, str]]:
    """Build the ``Authorization`` header for an LS-hosted asset fetch.

    * Legacy token → ``Token <token>``.
    * PAT (JWT) → exchanged for a short-lived access JWT, sent as
      ``Bearer <access>``. Falls back to ``Token <pat>`` if the exchange fails
      or the LS host is unknown, so legacy deployments still behave as before.

    Returns None when no token is configured.
    """
    if not token:
        return None
    if ls_url and _looks_like_jwt(token):
        access = _access_token_for_pat(ls_url, token)
        if access:
            return {"Authorization": f"Bearer {access}"}
        logger.warning("PAT exchange unavailable — falling back to Token scheme")
    return {"Authorization": f"Token {token}"}
