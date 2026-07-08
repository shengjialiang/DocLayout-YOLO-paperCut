"""End-to-end test with real PaddleOCR. Marked slow."""
import time

import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from service.exam import tasks


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_tasks():
    tasks.reset_for_tests()
    yield
    tasks.reset_for_tests()


@pytest.mark.slow
def test_real_pipeline_with_real_paddleocr(client):
    """Submit, poll until done, verify final has at least one question."""
    with open("assets/example/exam_paper.jpg", "rb") as f:
        r = client.post(
            "/predict/exam",
            files={"file": ("exam_paper.jpg", f, "image/jpeg")},
            data={
                "expand_mode": "pixel",
                "expand_top": 20, "expand_bottom": 20,
                "expand_left": 20, "expand_right": 20,
            },
        )
    assert r.status_code == 200
    tid = r.json()["task_id"]

    # Poll up to 120 seconds
    deadline = time.time() + 120
    while time.time() < deadline:
        r = client.get(f"/predict/exam/{tid}/status")
        assert r.status_code == 200
        status = r.json()["status"]
        if status == "done":
            body = r.json()
            assert body["final"] is not None
            assert len(body["final"]["ocr_blocks"]) > 0
            return
        if status == "failed":
            pytest.fail(f"task failed: {body['error']}")
        time.sleep(1)
    pytest.fail("task did not complete within 120s")