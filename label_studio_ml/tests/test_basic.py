import requests


def test_basic_health_check():
    response = requests.get("http://127.0.0.1:9090")
    assert response.status_code == 200

    response = requests.get("http://127.0.0.1:9090/health")
    assert response.status_code == 200