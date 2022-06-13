import requests


def test_basic_health_check():
    response = requests.get("http://127.0.0.1:9090")
    assert response.status_code == 200

    response = requests.get("http://127.0.0.1:9090/health")
    assert response.status_code == 200


def test_setup():
    data = {
        "project": "1.1654592272",
        'schema': '<View><Text name="text" value="$transcript"/><View style="box-shadow: 2px 2px 5px #999;                padding: 20px; margin-top: 2em;                border-radius: 5px;"><Header value="Choose text sentiment"/><Choices name="sentiment" toName="text" choice="single" showInLine="true"><Choice value="Positive"/><Choice value="Negative"/><Choice value="Neutral"/></Choices></View></View>',
        'hostname': "http://localhost:8080",
        'access_token': '1234567890123456789012345678901234567890'
    }
    response = requests.post("http://127.0.0.1:9090/setup", json=data)
    assert response.status_code == 200


def test_predict():
    pass


def test_webhook_predict():
    pass
