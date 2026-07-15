"""Unit tests for LS SDK-backed auth header building.

These run without torch/cv2. Refresh-token behavior is stubbed at the SDK
normalization boundary so no network is touched.

Run with: pytest test_ls_auth.py -v
"""

import base64
import json
import time
from unittest import mock

import pytest

import ls_auth


def _make_jwt(exp=None, token_type="refresh"):
    """Build a syntactically valid unsigned JWT for SDK parsing tests."""

    def b64(obj):
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    header = b64({"alg": "HS256", "typ": "JWT"})
    payload_obj = {"token_type": token_type, "user_id": "1"}
    if exp is not None:
        payload_obj["exp"] = exp
    payload = b64(payload_obj)
    signature = base64.urlsafe_b64encode(b"sig").rstrip(b"=").decode()
    return f"{header}.{payload}.{signature}"


class _FakeTokensClient:
    def __init__(self, resolved):
        self.resolved = resolved
        self.calls = []

    def resolve_x_api_key_header_value(self, token):
        self.calls.append(token)
        return self.resolved


class _FakeSdkClient:
    def __init__(self, resolved):
        self._client_wrapper = mock.Mock()
        self._client_wrapper._tokens_client = _FakeTokensClient(resolved)


@pytest.fixture(autouse=True)
def _clear_sdk_client_cache():
    ls_auth._SDK_CLIENT_CACHE.clear()
    yield
    ls_auth._SDK_CLIENT_CACHE.clear()


# --- header building ------------------------------------------------------


def test_no_token_returns_none():
    assert ls_auth.ls_auth_headers("http://ls", "") is None
    assert ls_auth.ls_auth_headers("http://ls", None) is None


def test_legacy_token_uses_sdk_token_scheme():
    headers = ls_auth.ls_auth_headers(
        "http://ls", "legacytoken123", target_url="http://ls/data/upload/1/video.mp4"
    )
    assert headers["Authorization"] == "Token legacytoken123"


def test_access_jwt_uses_sdk_bearer_scheme_without_refresh():
    access = _make_jwt(exp=int(time.time()) + 3600, token_type="access")

    with mock.patch.object(ls_auth, "_sdk_client", return_value=_FakeSdkClient(access)) as sdk_client:
        headers = ls_auth.ls_auth_headers(
            "http://ls", access, target_url="http://ls/data/upload/1/video.mp4"
        )

    assert headers["Authorization"] == f"Bearer {access}"
    sdk_client.assert_called_once_with("http://ls", access)


def test_refresh_jwt_is_normalized_by_sdk_before_streaming_headers():
    refresh = _make_jwt(exp=int(time.time()) + 3600, token_type="refresh")
    access = _make_jwt(exp=int(time.time()) + 3600, token_type="access")

    with mock.patch.object(ls_auth, "_sdk_client", return_value=_FakeSdkClient(access)) as sdk_client:
        headers = ls_auth.ls_auth_headers(
            "http://ls", refresh, target_url="http://ls/data/upload/1/video.mp4"
        )

    assert headers["Authorization"] == f"Bearer {access}"
    sdk_client.assert_called_once_with("http://ls", refresh)


def test_refresh_jwt_is_normalized_by_sdk_before_downloader_fallback():
    refresh = _make_jwt(exp=int(time.time()) + 3600, token_type="refresh")
    access = _make_jwt(exp=int(time.time()) + 3600, token_type="access")

    with mock.patch.object(ls_auth, "_sdk_client", return_value=_FakeSdkClient(access)):
        token = ls_auth.ls_token_for_sdk("http://ls", refresh)

    assert token == access


def test_sdk_normalization_failure_preserves_original_token():
    refresh = _make_jwt(exp=int(time.time()) + 3600, token_type="refresh")
    fake = mock.Mock()
    fake._client_wrapper._tokens_client.resolve_x_api_key_header_value.side_effect = RuntimeError("boom")

    with mock.patch.object(ls_auth, "_sdk_client", return_value=fake):
        token = ls_auth.ls_token_for_sdk("http://ls", refresh)

    assert token == refresh


def test_sdk_client_construction_failure_preserves_original_token():
    refresh = _make_jwt(exp=int(time.time()) + 3600, token_type="refresh")

    with mock.patch.object(ls_auth, "_sdk_client", side_effect=RuntimeError("boom")):
        token = ls_auth.ls_token_for_sdk("http://ls", refresh)

    assert token == refresh


def test_no_ls_host_preserves_token_for_sdk_downloader_fallback():
    token = "legacytoken123"
    assert ls_auth.ls_token_for_sdk(None, token) == token


def test_headers_attach_for_ls_host_with_scheme_port_difference():
    headers = ls_auth.ls_auth_headers(
        "http://app.humansignal.com:80",
        "legacytoken123",
        target_url="https://app.humansignal.com/upload/266842/x.mp4",
    )
    assert headers["Authorization"] == "Token legacytoken123"


def test_sdk_builder_does_not_attach_auth_to_mismatched_host():
    headers = ls_auth.ls_auth_headers(
        "http://ls", "legacytoken123", target_url="http://example.com/video.mp4"
    )
    assert "Authorization" not in headers
