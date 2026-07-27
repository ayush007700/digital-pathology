from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health():

    response = client.get("/")

    assert response.status_code == 200