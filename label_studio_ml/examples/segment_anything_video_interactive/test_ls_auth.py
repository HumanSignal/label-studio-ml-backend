"""Unit tests for LS API auth header building (legacy token vs PAT).

These run without torch/cv2: the auth logic lives in the dependency-free
``ls_auth`` module. ``requests`` is stubbed so no network is touched.

Run with: pytest test_ls_auth.py -v
"""

import base64
import json
import time
from unittest import mock

import pytest

import ls_auth


def _make_jwt(exp=None, token_type="refresh"):
    """Build a syntactically valid (unsigned) JWT for testing detection and
    `exp` parsing. Signature is a dummy — nothing here verifies it."""
    def b64(obj):
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    header = b64({"alg": "HS256", "typ": "JWT"})
    payload_obj = {"token_type": token_type, "user_id": "1"}
    if exp is not None:
        payload_obj["exp"] = exp
    payload = b64(payload_obj)
    return f"{header}.{payload}.{'sig'}"


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with an empty PAT access-token cache."""
    ls_auth._PAT_ACCESS_CACHE.clear()
    yield
    ls_auth._PAT_ACCESS_CACHE.clear()


# --- token-type detection -------------------------------------------------

def test_legacy_token_not_detected_as_jwt():
    assert ls_auth._looks_like_jwt("0123456789abcdef0123456789abcdef01234567") is False


def test_pat_detected_as_jwt():
    assert ls_auth._looks_like_jwt(_make_jwt(exp=9999999999)) is True


def test_jwt_exp_parsed():
    assert ls_auth._jwt_exp(_make_jwt(exp=123456)) == 123456.0


def test_jwt_exp_missing_returns_none():
    assert ls_auth._jwt_exp(_make_jwt(exp=None)) is None


def test_jwt_exp_garbage_returns_none():
    assert ls_auth._jwt_exp("not-a-jwt") is None


# --- header building ------------------------------------------------------

def test_no_token_returns_none():
    assert ls_auth.ls_auth_headers("http://ls", "") is None
    assert ls_auth.ls_auth_headers("http://ls", None) is None


def test_legacy_token_uses_token_scheme():
    headers = ls_auth.ls_auth_headers("http://ls", "legacytoken123")
    assert headers == {"Authorization": "Token legacytoken123"}


def test_pat_is_exchanged_for_bearer():
    pat = _make_jwt(exp=9999999999)
    access = _make_jwt(exp=int(time.time()) + 3600)
    resp = mock.Mock()
    resp.json.return_value = {"access": access}
    resp.raise_for_status.return_value = None
    with mock.patch.object(ls_auth.requests, "post", return_value=resp) as post:
        headers = ls_auth.ls_auth_headers("http://ls", pat)
    assert headers == {"Authorization": f"Bearer {access}"}
    post.assert_called_once_with(
        "http://ls/api/token/refresh", json={"refresh": pat},
        timeout=ls_auth._EXCHANGE_TIMEOUT,
    )


def test_pat_access_token_is_cached():
    pat = _make_jwt(exp=9999999999)
    access = _make_jwt(exp=int(time.time()) + 3600)
    resp = mock.Mock()
    resp.json.return_value = {"access": access}
    resp.raise_for_status.return_value = None
    with mock.patch.object(ls_auth.requests, "post", return_value=resp) as post:
        ls_auth.ls_auth_headers("http://ls", pat)
        ls_auth.ls_auth_headers("http://ls", pat)  # second call hits cache
    post.assert_called_once()  # only one network exchange


def test_expired_cached_access_token_is_refreshed():
    pat = _make_jwt(exp=9999999999)
    # Access token already past the refresh margin → must re-exchange.
    stale = _make_jwt(exp=int(time.time()) + 10)
    fresh = _make_jwt(exp=int(time.time()) + 3600)
    responses = [mock.Mock(), mock.Mock()]
    responses[0].json.return_value = {"access": stale}
    responses[1].json.return_value = {"access": fresh}
    for r in responses:
        r.raise_for_status.return_value = None
    with mock.patch.object(ls_auth.requests, "post", side_effect=responses) as post:
        first = ls_auth.ls_auth_headers("http://ls", pat)
        second = ls_auth.ls_auth_headers("http://ls", pat)
    assert first == {"Authorization": f"Bearer {stale}"}
    assert second == {"Authorization": f"Bearer {fresh}"}
    assert post.call_count == 2


def test_pat_falls_back_to_token_when_exchange_fails():
    pat = _make_jwt(exp=9999999999)
    with mock.patch.object(ls_auth.requests, "post", side_effect=Exception("boom")):
        headers = ls_auth.ls_auth_headers("http://ls", pat)
    # Degrade to legacy scheme rather than dropping auth entirely.
    assert headers == {"Authorization": f"Token {pat}"}


def test_pat_without_host_falls_back_to_token():
    pat = _make_jwt(exp=9999999999)
    # No LS host → can't reach the refresh endpoint; don't attempt exchange.
    with mock.patch.object(ls_auth.requests, "post") as post:
        headers = ls_auth.ls_auth_headers("", pat)
    post.assert_not_called()
    assert headers == {"Authorization": f"Token {pat}"}
