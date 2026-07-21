"""Integration tests for /predict/exam/enhance endpoints."""
from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


def _make_jpeg_bytes(width: int = 200, height: int = 150) -> bytes:
    arr = np.full((height, width, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", arr)
    assert ok
    return bytes(buf)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Build a TestClient with a fake YOLO model + no DocRect path."""
    import os
    model_pt = tmp_path / "model.pt"
    model_pt.write_bytes(b"")
    monkeypatch.setenv("MODEL_PATH", str(model_pt))
    monkeypatch.delenv("DOCRECT_MODEL_PATH", raising=False)

    # Prevent the real YOLO from loading during app construction.
    monkeypatch.setattr(
        "service.inference.load_model",
        lambda *a, **kw: object(),
    )

    from service.app import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_submit_enhance_returns_task_id(client: TestClient):
    raw = _make_jpeg_bytes()
    r = client.post(
        "/predict/exam/enhance",
        files={"file": ("test.jpg", io.BytesIO(raw), "image/jpeg")},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert "task_id" in body
    assert body["status"] == "queued"
    assert body["submit_url"].startswith("/predict/exam/enhance/")
    assert body["submit_url"].endswith("/status")


def test_submit_enhance_rejects_non_image(client: TestClient):
    r = client.post(
        "/predict/exam/enhance",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert r.status_code == 415
    assert r.json()["error_code"] == "BAD_MIME"