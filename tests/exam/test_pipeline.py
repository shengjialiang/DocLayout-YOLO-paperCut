"""Pipeline orchestration tests with mocked YOLO + OCR."""
import asyncio
import numpy as np
import pytest

from service.exam import pipeline, tasks
from service.exam.ocr import TextBlock


@pytest.fixture(autouse=True)
def reset_store():
    tasks.reset_for_tests()
    yield
    tasks.reset_for_tests()


def fake_yolo_result(boxes):
    """Build a fake YOLO Results-like object with .boxes and .names."""
    class FakeBoxes:
        def __init__(self, boxes):
            self.xyxy = np.array([b["xyxy"] for b in boxes], dtype=float)
            self.conf = np.array([b["conf"] for b in boxes], dtype=float)
            self.cls = np.array([b["cls"] for b in boxes], dtype=int)
    class FakeResult:
        def __init__(self, boxes):
            self.boxes = FakeBoxes(boxes) if boxes else None
            self.names = {0: "plain text", 1: "figure"}
        def plot(self, **kwargs):
            # Return a small image with shape (H, W, 3) RGB
            return np.zeros((100, 100, 3), dtype=np.uint8)
    return FakeResult(boxes)


def test_pipeline_runs_through_all_stages(monkeypatch):
    # Mock YOLO predict_one
    async def fake_predict_one(**kwargs):
        return {
            "width": 800, "height": 1000,
            "annotated_base64": "data:image/jpeg;base64,fake",
            "detections": [
                {"id": 0, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [50, 100, 750, 200], "bbox_xywh": [400, 150, 700, 100],
                 "score": 0.95},
                {"id": 1, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [50, 300, 750, 400], "bbox_xywh": [400, 350, 700, 100],
                 "score": 0.90},
                {"id": 2, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [50, 500, 750, 950], "bbox_xywh": [400, 725, 700, 450],
                 "score": 0.85},
            ],
            "num_detections": 3,
            "inference_time_ms": 5000,
            "model_imgsz": 1024,
            "conf_threshold": 0.3,
            "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)

    # Mock OCR engine
    class FakeOcr:
        def __init__(self):
            self.calls = 0
        def ocr_image(self, img):
            self.calls += 1
            # First call → "1.", Second → "2.", Third → "3."
            texts = ["1. (计算题) 1+1=?", "2. 阅读理解...", "3. 作文"]
            idx = self.calls - 1
            return [TextBlock(text=texts[idx], bbox=[0, 0, 100, 50], score=0.9)]
    fake_ocr = FakeOcr()
    monkeypatch.setattr(pipeline, "OcrEngine", lambda lang="ch": fake_ocr)

    # Mock image redraw (skip actual rendering)
    def fake_redraw(img_bgr, questions, original_detections):
        return "data:image/jpeg;base64,redrawn"
    monkeypatch.setattr(pipeline, "redraw_with_questions", fake_redraw)

    # Mock decode
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((1000, 800, 3), dtype=np.uint8))

    tid = tasks.create_task()
    params = {
        "expand_mode": "pixel",
        "expand_top": 10, "expand_bottom": 10,
        "expand_left": 10, "expand_right": 10,
        "question_regex": r"^\s*\(?\d+[\.\)](?!\d)",
        "conf": 0.3, "imgsz": 1024,
    }
    pipeline.run_pipeline(tid, b"fake-image-bytes", params, model=None, semaphore=asyncio.Semaphore(1))

    state = tasks.get_task(tid)
    assert state["status"] == "done"
    assert state["error"] is None
    # All 5 stages recorded
    for stage in ("yolo", "expand", "ocr", "filter", "geometry"):
        assert stage in state["stages"], f"stage {stage} missing"
        assert state["stages"][stage]["error"] is None
    # 3 questions found
    assert len(state["final"].questions) == 3
    # Q1 bottom = Q2 top after extension
    q1 = state["final"].questions[0]
    q2 = state["final"].questions[1]
    # After expand: y1 ~ 90, y2 ~ 210 (per pixel expand of [100, 200])
    # Q1 extended bottom = Q2 top ~ 290
    assert q1.bbox_xyxy[3] == pytest.approx(q2.bbox_xyxy[1], abs=2)


def test_pipeline_marks_failed_on_fatal_yolo_error(monkeypatch):
    async def fake_predict_one(**kwargs):
        raise RuntimeError("YOLO crashed")
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((100, 100, 3), dtype=np.uint8))

    tid = tasks.create_task()
    pipeline.run_pipeline(tid, b"x", {}, model=None, semaphore=asyncio.Semaphore(1))
    state = tasks.get_task(tid)
    assert state["status"] == "failed"
    assert "YOLO" in state["error"]


def test_pipeline_preserves_per_box_ocr_error(monkeypatch):
    """If OCR fails on any box, the stage error must be visible in the final
    status (not silently overwritten by the trailing update_stage call)."""
    async def fake_predict_one(**kwargs):
        return {
            "width": 800, "height": 1000,
            "annotated_base64": "data:img",
            "detections": [
                {"id": 0, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [50, 100, 750, 200], "bbox_xywh": [400, 150, 700, 100],
                 "score": 0.9},
                {"id": 1, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [50, 300, 750, 400], "bbox_xywh": [400, 350, 700, 100],
                 "score": 0.9},
            ],
            "num_detections": 2,
            "inference_time_ms": 100,
            "model_imgsz": 1024, "conf_threshold": 0.3, "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)

    class FailingOcr:
        def ocr_image(self, img):
            raise RuntimeError("PaddleOCR exploded on this crop")
    monkeypatch.setattr(pipeline, "OcrEngine", lambda lang="ch": FailingOcr())

    monkeypatch.setattr(pipeline, "redraw_with_questions",
                        lambda *a, **k: "data:image/jpeg;base64,redrawn")
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((1000, 800, 3), dtype=np.uint8))

    tid = tasks.create_task()
    pipeline.run_pipeline(tid, b"fake", {}, model=None, semaphore=asyncio.Semaphore(1))

    state = tasks.get_task(tid)
    assert state["stages"]["ocr"]["error"] is not None
    assert "PaddleOCR exploded" in state["stages"]["ocr"]["error"]


def test_pipeline_zero_questions_still_completes(monkeypatch):
    async def fake_predict_one(**kwargs):
        return {
            "width": 800, "height": 1000,
            "annotated_base64": "data:img",
            "detections": [],
            "num_detections": 0,
            "inference_time_ms": 100,
            "model_imgsz": 1024, "conf_threshold": 0.3, "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((100, 100, 3), dtype=np.uint8))

    tid = tasks.create_task()
    pipeline.run_pipeline(tid, b"x", {}, model=None, semaphore=asyncio.Semaphore(1))
    state = tasks.get_task(tid)
    assert state["status"] == "done"
    assert state["final"].questions == []