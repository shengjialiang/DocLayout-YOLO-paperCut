"""OCR engine tests with mocked PaddleOCR (no model download)."""
import numpy as np
import pytest

from service.exam.ocr import OcrEngine, TextBlock, _leftmost_chunk, ocr_leftmost


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


def test_leftmost_chunk_no_crop_for_normal_aspect():
    """Images within aspect budget pass through unchanged."""
    img = np.zeros((100, 300, 3), dtype=np.uint8)  # aspect 3.0
    out = _leftmost_chunk(img, max_aspect=4.0)
    assert out is img  # no slicing


def test_leftmost_chunk_wide_image_crops_to_width_times_aspect():
    """Wide images get cropped to width = H * max_aspect."""
    H = 100
    W = 1500  # aspect 15
    img = np.zeros((H, W, 3), dtype=np.uint8)
    out = _leftmost_chunk(img, max_aspect=4.0)
    # Expected: H * 4 = 400 columns
    assert out.shape[0] == H
    assert out.shape[1] == H * 4
    # And it's the leftmost 400 columns
    np.testing.assert_array_equal(out, img[:, :H*4])


def test_leftmost_chunk_tall_image_crops_to_height_times_aspect():
    """Tall images get cropped to height = W * max_aspect (topmost rows)."""
    W = 100
    H = 1500  # aspect 15 tall
    img = np.zeros((H, W, 3), dtype=np.uint8)
    out = _leftmost_chunk(img, max_aspect=4.0)
    assert out.shape[1] == W
    assert out.shape[0] == W * 4
    np.testing.assert_array_equal(out, img[:W*4, :])


def test_leftmost_chunk_respects_image_size_when_chunk_larger():
    """If H * max_aspect > W, return the whole image."""
    img = np.zeros((100, 50, 3), dtype=np.uint8)  # aspect 0.5 — but here W=50, H*4=400 so chunk larger
    out = _leftmost_chunk(img, max_aspect=4.0)
    # aspect_w = 0.5 < 4, aspect_h = 2.0 < 4 → no crop
    assert out is img


def test_ocr_leftmost_calls_run_ocr_on_full_image_for_normal_aspect(monkeypatch):
    """For normal aspect, ocr_leftmost runs OCR on the full image."""
    seen_imgs = []

    def fake_run_ocr(engine, img):
        seen_imgs.append(img)
        return [TextBlock(text="5.", bbox=[0, 0, 30, 30], score=0.9)]

    monkeypatch.setattr("service.exam.ocr._run_ocr", fake_run_ocr)
    img = np.zeros((100, 300, 3), dtype=np.uint8)  # aspect 3.0
    blocks = ocr_leftmost("engine", img)
    assert len(seen_imgs) == 1
    assert seen_imgs[0].shape == (100, 300, 3)
    assert blocks[0].text == "5."


def test_ocr_leftmost_calls_run_ocr_on_left_chunk_for_wide_image(monkeypatch):
    """For wide images, only the leftmost chunk should be OCR'd."""
    H = 100
    W = 1500  # aspect 15
    seen_imgs = []

    def fake_run_ocr(engine, img):
        seen_imgs.append(img)
        # Simulate correct detection of the question number prefix.
        return [TextBlock(text="5.", bbox=[0, 0, 30, 30], score=0.95)]

    monkeypatch.setattr("service.exam.ocr._run_ocr", fake_run_ocr)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    blocks = ocr_leftmost("engine", img)
    assert len(seen_imgs) == 1
    # OCR'd input is the leftmost chunk: (H, H*4) = (100, 400)
    np.testing.assert_array_equal(seen_imgs[0], img[:, :H*4])
    # The "5." prefix was detected in the leftmost chunk → caller can decide
    # the original crop is a question box.
    assert blocks[0].text == "5."


def test_ocr_leftmost_handles_chinese_question_prefix(monkeypatch):
    """Regression for the real-world case: a wide question crop with Chinese."""
    def fake_run_ocr(engine, img):
        return [TextBlock(text="5.ʵa", bbox=[0, 0, 200, 80], score=0.95)]

    monkeypatch.setattr("service.exam.ocr._run_ocr", fake_run_ocr)
    img = np.zeros((100, 1500, 3), dtype=np.uint8)
    blocks = ocr_leftmost("engine", img)
    # The downstream is_question() check uses the regex on blocks[0].text,
    # which now starts with "5." and matches.
    assert blocks[0].text.startswith("5.")


def test_ocr_engine_routes_through_leftmost_for_wide_images(monkeypatch):
    """OcrEngine.ocr_image should automatically use leftmost-chunk for wide images."""
    seen_imgs = []

    def fake_run_ocr(engine, img):
        seen_imgs.append(img)
        return [TextBlock(text="1.", bbox=[0, 0, 10, 10], score=0.9)]

    monkeypatch.setattr("service.exam.ocr._load_paddleocr", lambda: "engine")
    monkeypatch.setattr("service.exam.ocr._run_ocr", fake_run_ocr)

    engine = OcrEngine(lang="ch")
    wide_img = np.zeros((100, 1500, 3), dtype=np.uint8)  # aspect 15
    engine.ocr_image(wide_img)
    # Single call, on a cropped chunk — not the original.
    assert len(seen_imgs) == 1
    assert seen_imgs[0].shape[1] < wide_img.shape[1]  # narrower than original
