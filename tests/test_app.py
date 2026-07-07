import pytest
from unittest import mock
from fastapi.testclient import TestClient

from service.app import create_app
from service.config import Config


def _fake_config(tmp_path=None):
    m = tmp_path or "/tmp/fake.pt"
    return Config(
        model_path=m,
        device="cpu",
        host="127.0.0.1",
        port=8000,
        max_concurrent=1,
        max_file_size_mb=20,
    )


def test_health_ok_after_model_loaded(tmp_path):
    """/health returns ok once model is loaded."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    fake_model = mock.MagicMock()

    with mock.patch("service.app.load_model", return_value=fake_model), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = fake_model  # simulate lifespan having loaded
        client = TestClient(app)
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert body["model_loaded"] is True
    assert body["max_concurrent"] == 1


def test_health_starting_when_no_model(tmp_path):
    """If lifespan didn't run (no model), health returns 503 + status: starting."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    with mock.patch("service.app.load_config", return_value=cfg), \
         mock.patch("service.app.load_model", return_value=mock.MagicMock()):
        app = create_app()
        # Don't enter the lifespan context: model is None.
        client = TestClient(app)  # No `with` — no lifespan
        response = client.get("/health")

    # Without lifespan, our model is None; we expect 503.
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "starting"
    assert body["model_loaded"] is False