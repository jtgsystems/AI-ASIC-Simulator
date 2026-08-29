from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_simulation_endpoint() -> None:
    response = client.post(
        "/simulate",
        json={"signal": [0.0, 0.2, 0.4, 0.2, 0.0, -0.2, -0.4, -0.2] * 4},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chip_count"] == 2
    assert body["feature_count"] == 216
    assert len(body["telemetry"]) == 64
