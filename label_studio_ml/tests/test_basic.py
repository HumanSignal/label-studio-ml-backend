import requests


def test_basic_health_check():
    response = requests.get("http://127.0.0.1:9090")
    assert response.status_code == 200

    response = requests.get("http://127.0.0.1:9090/health")
    assert response.status_code == 200


def test_setup():
    data = {
        "project": "1.1654592272",
        'schema': '',
        'hostname': "http://localhost:8080",
        'access_token': '1234567890123456789012345678901234567890'
    }
    response = requests.post("http://127.0.0.1:9090/setup", json=data)
    assert response.status_code == 200
