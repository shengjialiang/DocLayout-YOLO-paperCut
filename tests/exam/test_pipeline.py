"""Pipeline orchestration tests with mocked YOLO + OCR."""
import asyncio
import concurrent.futures
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


class mock_ocr_empty:
    """OCR stub returning no text (used when YOLO yields empty detections)."""
    def ocr_image(self, img):
        return []


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


def test_pipeline_passes_line_width_and_font_size_to_predict_one(monkeypatch):
    """line_width and font_size in params must reach predict_one, not be
    silently hardcoded."""
    captured: dict = {}

    async def fake_predict_one(**kwargs):
        captured.update(kwargs)
        return {
            "width": 100, "height": 100,
            "annotated_base64": "data:img",
            "detections": [],
            "num_detections": 0,
            "inference_time_ms": 1,
            "model_imgsz": 1024, "conf_threshold": 0.3, "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)
    monkeypatch.setattr(pipeline, "OcrEngine", lambda lang="ch": mock_ocr_empty())
    monkeypatch.setattr(pipeline, "redraw_with_questions",
                        lambda *a, **k: "data:image/jpeg;base64,redrawn")
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((100, 100, 3), dtype=np.uint8))

    tid = tasks.create_task()
    pipeline.run_pipeline(
        tid, b"fake", {
            "expand_mode": "pixel",
            "expand_top": 0, "expand_bottom": 0, "expand_left": 0, "expand_right": 0,
            "conf": 0.25, "imgsz": 800,
            "line_width": 12, "font_size": 30,
        },
        model=None, semaphore=asyncio.Semaphore(1),
    )
    assert captured["line_width"] == 12
    assert captured["font_size"] == 30


def test_pipeline_uses_default_line_width_font_size_when_absent(monkeypatch):
    """When params lacks line_width / font_size, fall back to safe defaults
    instead of crashing."""
    captured: dict = {}

    async def fake_predict_one(**kwargs):
        captured.update(kwargs)
        return {
            "width": 100, "height": 100,
            "annotated_base64": "data:img",
            "detections": [],
            "num_detections": 0,
            "inference_time_ms": 1,
            "model_imgsz": 1024, "conf_threshold": 0.3, "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)
    monkeypatch.setattr(pipeline, "OcrEngine", lambda lang="ch": mock_ocr_empty())
    monkeypatch.setattr(pipeline, "redraw_with_questions",
                        lambda *a, **k: "data:image/jpeg;base64,redrawn")
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((100, 100, 3), dtype=np.uint8))

    tid = tasks.create_task()
    pipeline.run_pipeline(
        tid, b"fake", {
            "expand_mode": "pixel",
            "expand_top": 0, "expand_bottom": 0, "expand_left": 0, "expand_right": 0,
        },
        model=None, semaphore=asyncio.Semaphore(1),
    )
    assert captured["line_width"] == 5
    assert captured["font_size"] == 20


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


def test_pipeline_question_number_not_first_ocr_block(monkeypatch):
    """Regression: PaddleOCR returns blocks top-to-bottom, so a question box
    whose leftmost (x-wise) text is NOT the topmost can be misclassified.
    Real example: 第十题.jpg — OCR returns [mx+3y= (top-right), 10.若...
    (middle-left), 4x+y=9 (bottom-right)]; the question number "10." is at
    x=34 but the first block in PaddleOCR order is at x=705. Concatenating
    in natural order yields "mx+3y= 10.若..." which fails `^\\s*\\(?\\d+...`.
    Pipeline must sort blocks by x before joining so the question number
    ends up at the start of the joined text.
    """
    async def fake_predict_one(**kwargs):
        return {
            "width": 2000, "height": 300,
            "annotated_base64": "data:img",
            "detections": [
                {"id": 0, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [0, 0, 1690, 250], "bbox_xywh": [845, 125, 1690, 250],
                 "score": 0.9},
            ],
            "num_detections": 1,
            "inference_time_ms": 100,
            "model_imgsz": 1024,
            "conf_threshold": 0.3,
            "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)

    class OcrReturnsBlocksInPaddleOrder:
        def ocr_image(self, img):
            # Natural PaddleOCR order: top-to-bottom, left-to-right.
            return [
                TextBlock(text="mx+3y=",   bbox=[705, 12, 885, 57],  score=0.99),
                TextBlock(text="10.若关于x,y的二元一次方程组",
                          bbox=[34, 53, 673, 86], score=0.94),
                TextBlock(text="4x+y=9",   bbox=[721, 79, 894, 123], score=0.98),
            ]
    monkeypatch.setattr(pipeline, "OcrEngine",
                        lambda lang="ch": OcrReturnsBlocksInPaddleOrder())
    monkeypatch.setattr(pipeline, "redraw_with_questions",
                        lambda *a, **k: "data:image/jpeg;base64,redrawn")
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((300, 2000, 3), dtype=np.uint8))

    tid = tasks.create_task()
    pipeline.run_pipeline(
        tid, b"fake", {
            "expand_mode": "pixel",
            "expand_top": 0, "expand_bottom": 0,
            "expand_left": 0, "expand_right": 0,
            "question_regex": r"^\s*\(?\d+[\.。．、\)]",
        },
        model=None, semaphore=asyncio.Semaphore(1),
    )

    state = tasks.get_task(tid)
    assert state["status"] == "done", state.get("error")
    assert len(state["final"].questions) == 1, (
        f"expected the '10.' question box to be detected; "
        f"questions={state['final'].questions}"
    )
    # The question box contains the "10." prefix in at least one OCR block
    # (per-block check, not dependent on join order).
    assert any(
        any(t.startswith("10.") for t in b.block_texts)
        for b in state["final"].ocr_blocks
    ), [b.block_texts for b in state["final"].ocr_blocks]


def test_pipeline_multiline_question_with_offset_lines(monkeypatch):
    """Regression: 第十一题.jpg — two text lines where the second line
    starts a few pixels to the LEFT of the first line (x=32 vs x=34).
    Under the previous x-sort heuristic this would put the body line ahead
    of the question number, breaking the ^-anchored prefix check. Per-block
    `is_question` check is robust to this: any block whose text starts with
    a question number pattern makes the crop a question box.
    """
    async def fake_predict_one(**kwargs):
        return {
            "width": 2000, "height": 200,
            "annotated_base64": "data:img",
            "detections": [
                {"id": 0, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [0, 0, 1810, 162], "bbox_xywh": [905, 81, 1810, 162],
                 "score": 0.9},
            ],
            "num_detections": 1,
            "inference_time_ms": 100,
            "model_imgsz": 1024,
            "conf_threshold": 0.3,
            "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)

    class OcrTwoLinesSecondOffsetLeft:
        def ocr_image(self, img):
            # Line 2 starts 2px to the LEFT of line 1 (中文排版常见).
            return [
                TextBlock(text="11.如图所示，一个四边形纸片",
                          bbox=[34, 14, 632, 50], score=0.96),
                TextBlock(text="在AD边上的B点，AE是折痕",
                          bbox=[32, 111, 632, 148], score=0.98),
            ]
    monkeypatch.setattr(pipeline, "OcrEngine",
                        lambda lang="ch": OcrTwoLinesSecondOffsetLeft())
    monkeypatch.setattr(pipeline, "redraw_with_questions",
                        lambda *a, **k: "data:image/jpeg;base64,redrawn")
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((200, 2000, 3), dtype=np.uint8))

    tid = tasks.create_task()
    pipeline.run_pipeline(
        tid, b"fake", {
            "expand_mode": "pixel",
            "expand_top": 0, "expand_bottom": 0,
            "expand_left": 0, "expand_right": 0,
            "question_regex": r"^\s*\(?\d+[\.。．、\)]",
        },
        model=None, semaphore=asyncio.Semaphore(1),
    )

    state = tasks.get_task(tid)
    assert state["status"] == "done", state.get("error")
    assert len(state["final"].questions) == 1, (
        f"expected the '11.' question box to be detected; "
        f"questions={state['final'].questions}"
    )
    # The matched question box should expose a "11." block in its OCR.
    assert any(
        any(t.startswith("11.") for t in b.block_texts)
        for b in state["final"].ocr_blocks
    ), [b.block_texts for b in state["final"].ocr_blocks]


def test_pipeline_no_question_number_anywhere_is_not_a_question(monkeypatch):
    """Reverse case: a non-question box may still have several OCR blocks
    (e.g., a paragraph or caption), but NONE of them starts with a question
    number pattern. The crop must NOT be marked as a question box."""
    async def fake_predict_one(**kwargs):
        return {
            "width": 600, "height": 200,
            "annotated_base64": "data:img",
            "detections": [
                {"id": 0, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [0, 0, 600, 200], "bbox_xywh": [300, 100, 600, 200],
                 "score": 0.9},
            ],
            "num_detections": 1,
            "inference_time_ms": 100,
            "model_imgsz": 1024,
            "conf_threshold": 0.3,
            "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)

    class OcrParagraphNoQuestion:
        def ocr_image(self, img):
            return [
                TextBlock(text="本题共10分", bbox=[10, 10, 200, 40], score=0.9),
                TextBlock(text="解答如下：", bbox=[10, 50, 200, 80], score=0.9),
            ]
    monkeypatch.setattr(pipeline, "OcrEngine",
                        lambda lang="ch": OcrParagraphNoQuestion())
    monkeypatch.setattr(pipeline, "redraw_with_questions",
                        lambda *a, **k: "data:image/jpeg;base64,redrawn")
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((200, 600, 3), dtype=np.uint8))

    tid = tasks.create_task()
    pipeline.run_pipeline(
        tid, b"fake", {
            "expand_mode": "pixel",
            "expand_top": 0, "expand_bottom": 0,
            "expand_left": 0, "expand_right": 0,
            "question_regex": r"^\s*\(?\d+[\.。．、\)]",
        },
        model=None, semaphore=asyncio.Semaphore(1),
    )

    state = tasks.get_task(tid)
    assert state["status"] == "done", state.get("error")
    assert len(state["final"].questions) == 0
    assert all(not b.is_question for b in state["final"].ocr_blocks)


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

def test_pipeline_ocr_uses_thread_pool(monkeypatch):
    """Stage 3 OCR must dispatch via concurrent.futures.ThreadPoolExecutor."""
    import concurrent.futures as cf

    async def fake_predict_one(**kwargs):
        return {
            "width": 800, "height": 1000,
            "annotated_base64": "data:img",
            "detections": [
                {"id": 0, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [50, 100, 750, 200], "bbox_xywh": [400, 150, 700, 100],
                 "score": 0.9},
            ],
            "num_detections": 1, "inference_time_ms": 100,
            "model_imgsz": 1024, "conf_threshold": 0.3, "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((1000, 800, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline, "OcrEngine", lambda lang="ch": mock_ocr_empty())
    monkeypatch.setattr(pipeline, "redraw_with_questions",
                        lambda *a, **k: "data:image/jpeg;base64,redrawn")

    seen = {"max_workers": None}

    class SpyExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
            seen["max_workers"] = max_workers
            self._real = cf.ThreadPoolExecutor(max_workers)
        def __enter__(self):
            return self._real.__enter__()
        def __exit__(self, *args):
            return self._real.__exit__(*args)
        def map(self, fn, iterable, **kwargs):
            return self._real.map(fn, iterable, **kwargs)

    monkeypatch.setattr(pipeline.concurrent.futures, "ThreadPoolExecutor", SpyExecutor)

    tid = tasks.create_task()
    pipeline.run_pipeline(tid, b"fake", {
        "expand_mode": "pixel", "expand_top": 0, "expand_bottom": 0,
        "expand_left": 0, "expand_right": 0,
    }, model=None, semaphore=asyncio.Semaphore(1))

    state = tasks.get_task(tid)
    assert state["status"] == "done", state.get("error")
    assert seen["max_workers"] is not None and seen["max_workers"] >= 1


def test_pipeline_ocr_block_order_preserved(monkeypatch):
    """pool.map semantics: ocr_blocks must come out in plain_expanded order
    even if individual OCR calls complete out of order."""
    import time

    async def fake_predict_one(**kwargs):
        return {
            "width": 1000, "height": 1000,
            "annotated_base64": "data:img",
            "detections": [
                {"id": i, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [10, 100 + i * 80, 990, 100 + i * 80 + 60],
                 "bbox_xywh": [500, 130 + i * 80, 980, 60],
                 "score": 0.9}
                for i in range(5)
            ],
            "num_detections": 5, "inference_time_ms": 100,
            "model_imgsz": 1024, "conf_threshold": 0.3, "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((1000, 1000, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline, "redraw_with_questions",
                        lambda *a, **k: "data:image/jpeg;base64,redrawn")

    class OcrTagsByCallOrder:
        # Shared counter across instances so we can verify pool.map order
        # preservation even though OcrEngine() is constructed per-worker.
        _shared_n = 0
        def __init__(self):
            pass
        def ocr_image(self, img):
            OcrTagsByCallOrder._shared_n += 1
            order = OcrTagsByCallOrder._shared_n
            # Odd calls finish first (would scramble order without pool.map).
            time.sleep(0.04 if order % 2 == 1 else 0.10)
            return [TextBlock(text=f"box{order}", bbox=[0, 0, 10, 10], score=0.9)]

    # Reset shared counter (other tests may have bumped it).
    OcrTagsByCallOrder._shared_n = 0

    monkeypatch.setattr(pipeline, "OcrEngine",
                        lambda lang="ch": OcrTagsByCallOrder())

    tid = tasks.create_task()
    pipeline.run_pipeline(tid, b"fake", {
        "expand_mode": "pixel", "expand_top": 0, "expand_bottom": 0,
        "expand_left": 0, "expand_right": 0,
    }, model=None, semaphore=asyncio.Semaphore(1))

    state = tasks.get_task(tid)
    assert state["status"] == "done", state.get("error")
    block_texts = [b.block_texts[0] for b in state["final"].ocr_blocks]
    assert block_texts == ["box1", "box2", "box3", "box4", "box5"], block_texts
