"""End-to-end smoke tests hitting the running backend over HTTP.

Run with: pytest test_api.py -v
Requires the server to be running on $ML_BACKEND_URL (default http://localhost:9090).
These tests are illustrative; they skip automatically if the backend is not
reachable so they can live in the repo without breaking unrelated CI.
"""

import os

import pytest
import requests

BASE_URL = os.getenv("ML_BACKEND_URL", "http://localhost:9090")


def _reachable() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="backend not running")


def _predict(context, tasks=None, label_config=None):
    payload = {
        "tasks": tasks or [{"id": 1, "data": {"image": "https://example.com/img.jpg"}}],
        "project": "1.1700000000",
        "label_config": label_config or _IMAGE_BRUSH_CONFIG,
        "params": {"context": context},
    }
    return requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)


_IMAGE_BRUSH_CONFIG = """
<View>
  <Image name="image" value="$image"/>
  <BrushLabels name="tag" toName="image">
    <Label value="object"/>
  </BrushLabels>
</View>
""".strip()


def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200
    assert r.json().get("status") == "UP"


def test_image_prewarm_returns_ack():
    r = _predict({"event": "prewarm", "frame": 0, "window": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["results"], body
