"""Smoke tests for the Docling ML backend API."""

import pytest


@pytest.fixture
def client():
    from label_studio_ml.api import init_app

    from model import Docling

    app = init_app(model_class=Docling)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_server_status(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.get_json()
    assert body is not None
    assert body.get("status") == "UP"
