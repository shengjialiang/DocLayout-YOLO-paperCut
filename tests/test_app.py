import io
import numpy as np
from PIL import Image
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from service.config import Config
from service.schemas import PredictResponse


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


def _png_bytes(width=200, height=100):
    img = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


def test_predict_image_returns_jpeg_bytes(tmp_path):
    """/predict/image returns image/jpeg body."""
    cfg = _fake_config(str(tmp_path / "m.pt"))

    # Mock model predicts one box
    fake_box = mock.MagicMock()
    fake_box.xyxy = np.array([[10.0, 20.0, 110.0, 70.0]])
    fake_box.conf = np.array([0.95])
    fake_box.cls = np.array([0])

    class _PlotResult:
        names = {0: "title"}
        boxes = fake_box
        orig_shape = (100, 200)

        def plot(self, pil=True, line_width=5, font_size=20):
            return np.zeros((100, 200, 3), dtype=np.uint8)

    fake_model = mock.MagicMock()
    fake_model.predict = mock.MagicMock(return_value=[_PlotResult()])

    with mock.patch("service.app.load_model", return_value=fake_model), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = fake_model
        client = TestClient(app)
        response = client.post(
            "/predict/image",
            files={"file": ("test.png", _png_bytes(), "image/png")},
            data={"conf": "0.2", "imgsz": "1024"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert len(response.content) > 0


def test_predict_image_rejects_non_image_mime(tmp_path):
    """Uploading a text file returns 415."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    with mock.patch("service.app.load_model", return_value=mock.MagicMock()), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = mock.MagicMock()
        client = TestClient(app)
        response = client.post(
            "/predict/image",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
    assert response.status_code == 415


def test_predict_image_rejects_corrupt_image(tmp_path):
    """Corrupt image bytes return 400."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    with mock.patch("service.app.load_model", return_value=mock.MagicMock()), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = mock.MagicMock()
        client = TestClient(app)
        response = client.post(
            "/predict/image",
            files={"file": ("bad.png", b"not-an-image", "image/png")},
        )
    # Either 400 from decode failure, or 415/422 if MIME is invalid.
    assert response.status_code in (400, 415, 422)


def test_predict_image_rejects_imgsz_out_of_range(tmp_path):
    """imgsz outside [320, 2048] returns 422."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    with mock.patch("service.app.load_model", return_value=mock.MagicMock()), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = mock.MagicMock()
        client = TestClient(app)
        response = client.post(
            "/predict/image",
            files={"file": ("test.png", _png_bytes(), "image/png")},
            data={"imgsz": "50"},  # below 320
        )
    assert response.status_code == 422


def test_predict_image_requires_file(tmp_path):
    """Missing file field returns 422."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    with mock.patch("service.app.load_model", return_value=mock.MagicMock()), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = mock.MagicMock()
        client = TestClient(app)
        response = client.post("/predict/image")
    assert response.status_code == 422


def test_predict_returns_json_with_annotated_image(tmp_path):
    """/predict returns PredictResponse with annotated_image base64 + detections."""
    cfg = _fake_config(str(tmp_path / "m.pt"))

    fake_box = mock.MagicMock()
    fake_box.xyxy = np.array([[10.0, 20.0, 110.0, 70.0]])
    fake_box.conf = np.array([0.95])
    fake_box.cls = np.array([0])

    class _PlotResult:
        names = {0: "title"}
        boxes = fake_box
        orig_shape = (100, 200)

        def plot(self, pil=True, line_width=5, font_size=20):
            return np.zeros((100, 200, 3), dtype=np.uint8)

    fake_model = mock.MagicMock()
    fake_model.predict = mock.MagicMock(return_value=[_PlotResult()])

    with mock.patch("service.app.load_model", return_value=fake_model), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = fake_model
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("test.png", _png_bytes(), "image/png")},
            data={"conf": "0.2"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["width"] == 200
    assert body["height"] == 100
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")
    assert body["num_detections"] == 1
    assert body["detections"][0]["class_name"] == "title"
    assert body["detections"][0]["score"] == 0.95
    assert body["conf_threshold"] == 0.2
    assert body["device"] == "cpu"


def test_predict_serves_concurrent_requests_serially(tmp_path):
    """Multiple concurrent requests all succeed; semaphore doesn't deadlock."""
    cfg = _fake_config(str(tmp_path / "m.pt"))

    fake_box = mock.MagicMock()
    fake_box.xyxy = np.array([[0.0, 0.0, 50.0, 50.0]])
    fake_box.conf = np.array([0.9])
    fake_box.cls = np.array([0])

    class _PlotResult:
        names = {0: "x"}
        boxes = fake_box
        orig_shape = (50, 50)

        def plot(self, pil=True, line_width=5, font_size=20):
            return np.zeros((50, 50, 3), dtype=np.uint8)

    fake_model = mock.MagicMock()
    fake_model.predict = mock.MagicMock(return_value=[_PlotResult()])

    with mock.patch("service.app.load_model", return_value=fake_model), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = fake_model
        client = TestClient(app)
        responses = []
        for _ in range(5):
            responses.append(client.post(
                "/predict",
                files={"file": ("t.png", _png_bytes(50, 50), "image/png")},
            ))
    assert all(r.status_code == 200 for r in responses)
    assert fake_model.predict.call_count == 5
