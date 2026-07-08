"""OCR engine tests with mocked PaddleOCR (no model download)."""
import numpy as np
import pytest

from service.exam.ocr import OcrEngine, TextBlock


def test_text_block_dataclass():
    block = TextBlock(text="hello", bbox=[0, 0, 100, 50], score=0.95)
    assert block.text == "hello"
    assert block.score == 0.95


def test_ocr_engine_lazy_load(monkeypatch):
    """OcrEngine should not call PaddleOCR at construction."""
    loaded = {"called": False}

    def fake_load():
        loaded["called"] = True
        return "fake-engine"

    monkeypatch.setattr(
        "service.exam.ocr._load_paddleocr", fake_load
    )
    engine = OcrEngine(lang="ch")
    assert loaded["called"] is False


def test_ocr_engine_loads_on_first_use(monkeypatch):
    loaded = {"called": False}

    def fake_load():
        loaded["called"] = True
        return "fake-engine"

    monkeypatch.setattr(
        "service.exam.ocr._load_paddleocr", fake_load
    )
    monkeypatch.setattr(
        "service.exam.ocr._run_ocr",
        lambda engine, img: [TextBlock(text="1.", bbox=[0, 0, 10, 10], score=0.9)],
    )
    engine = OcrEngine(lang="ch")
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    result = engine.ocr_image(img)
    assert loaded["called"] is True
    assert len(result) == 1
    assert result[0].text == "1."


def test_ocr_engine_caches_engine(monkeypatch):
    """PaddleOCR should only load once per OcrEngine instance."""
    call_count = {"n": 0}

    def fake_load():
        call_count["n"] += 1
        return "engine"

    monkeypatch.setattr("service.exam.ocr._load_paddleocr", fake_load)
    monkeypatch.setattr(
        "service.exam.ocr._run_ocr",
        lambda e, img: [],
    )
    engine = OcrEngine(lang="ch")
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    engine.ocr_image(img)
    engine.ocr_image(img)
    assert call_count["n"] == 1


def test_ocr_engine_handles_load_failure(monkeypatch):
    def fake_load():
        raise RuntimeError("model not found")
    monkeypatch.setattr("service.exam.ocr._load_paddleocr", fake_load)
    engine = OcrEngine(lang="ch")
    with pytest.raises(RuntimeError):
        engine.ocr_image(np.zeros((10, 10, 3), dtype=np.uint8))
