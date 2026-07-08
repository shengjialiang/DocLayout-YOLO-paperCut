"""HTTP tests for /predict/exam endpoints."""
import io
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from service.config import Config
from service.exam import tasks


def _fake_config(tmp_path):
    return Config(
        model_path=str(tmp_path / "fake.pt"),
        device="cpu",
        host="127.0.0.1",
        port=8000,
        max_concurrent=1,
        max_file_size_mb=20,
    )


@pytest.fixture
def client(tmp_path):
    cfg = _fake_config(tmp_path)
    fake_model = mock.MagicMock()
    # Patch run_pipeline so background tasks do nothing — tests only check
    # task_id/queued submission and 422/415/404 responses, not pipeline execution.
    noop_pipeline = mock.MagicMock()
    with mock.patch("service.app.load_model", return_value=fake_model), \
         mock.patch("service.app.load_config", return_value=cfg), \
         mock.patch("service.app.run_pipeline", noop_pipeline):
        app = create_app()
        # Simulate lifespan having loaded a model
        app.state.service_state["model"] = fake_model
        # No `with` so lifespan doesn't run; tests set model manually.
        yield TestClient(app)


@pytest.fixture(autouse=True)
def reset_tasks():
    tasks.reset_for_tests()
    yield
    tasks.reset_for_tests()


def _make_image(filename="test.png"):
    # Minimal valid PNG bytes (1x1 red pixel)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x5c\xcd\xff\x69\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return (filename, png_bytes, "image/png")


def test_submit_returns_task_id_and_queued_status(client):
    r = client.post("/predict/exam", files={"file": _make_image()})
    assert r.status_code == 200
    data = r.json()
    assert "task_id" in data
    assert data["status"] == "queued"
    assert data["submit_url"].startswith("/predict/exam/")


def test_status_returns_404_for_unknown_task(client):
    r = client.get("/predict/exam/nonexistent/status")
    assert r.status_code == 404


def test_status_returns_200_for_known_task(client):
    r = client.post("/predict/exam", files={"file": _make_image()})
    tid = r.json()["task_id"]
    r2 = client.get(f"/predict/exam/{tid}/status")
    assert r2.status_code == 200
    body = r2.json()
    assert body["task_id"] == tid
    assert body["status"] in ("queued", "running", "done")


def test_invalid_expand_mode_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"expand_mode": "bad"})
    assert r.status_code == 422


def test_non_numeric_expand_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"expand_top": "abc"})
    assert r.status_code == 422


def test_negative_pixel_expand_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"expand_top": "-5"})
    assert r.status_code == 422


def test_out_of_range_ratio_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"expand_mode": "ratio", "expand_top": "0.9"})
    assert r.status_code == 422


def test_invalid_regex_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"question_regex": "["})
    assert r.status_code == 422


def test_regex_matching_empty_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"question_regex": ".*"})
    assert r.status_code == 422


def test_non_image_mime_returns_415(client):
    r = client.post("/predict/exam",
                    files={"file": ("notes.txt", b"plain text", "text/plain")})
    assert r.status_code == 415


def test_invalid_conf_for_exam_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"conf": "1.5"})
    assert r.status_code == 422


def test_invalid_imgsz_for_exam_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"imgsz": "100"})
    assert r.status_code == 422


def test_invalid_line_width_for_exam_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"line_width": "50"})
    assert r.status_code == 422


def test_invalid_font_size_for_exam_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"font_size": "100"})
    assert r.status_code == 422


def test_exam_accepts_custom_yolo_and_draw_params(client):
    """All four pass-through params accepted with valid values."""
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"conf": "0.25", "imgsz": "800",
                          "line_width": "10", "font_size": "30"})
    assert r.status_code == 200
    assert "task_id" in r.json()
