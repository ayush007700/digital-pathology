from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_predict():

    with open(
        r"C:\Users\ayush\Downloads\pcam.jpg",
        "rb",
    ) as image:

        response = client.post(

            "/predict",

            files={

                "file": image,

            },

        )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert "confidence" in data

    assert "overlay" in data

    assert "report" in data