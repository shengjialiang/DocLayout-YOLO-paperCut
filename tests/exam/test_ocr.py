"""OCR engine tests with mocked PaddleOCR (no model download)."""
import numpy as np
import pytest

from service.exam.ocr import (
    OcrEngine,
    TextBlock,
    _split_extreme_image,
    ocr_with_chunking,
)


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


def test_split_no_split_for_normal_aspect():
    """Images with reasonable aspect ratio should not be split."""
    img = np.zeros((100, 300, 3), dtype=np.uint8)  # aspect 3.0
    chunks = _split_extreme_image(img, max_aspect=4.0)
    assert len(chunks) == 1
    x_off, y_off, crop = chunks[0]
    assert x_off == 0 and y_off == 0
    assert crop is img


def test_split_wide_image_horizontally():
    """Wide images should be split horizontally into overlapping chunks."""
    img = np.zeros((100, 1000, 3), dtype=np.uint8)  # aspect 10
    chunks = _split_extreme_image(img, max_aspect=4.0)
    # 1000 / (100*4) ≈ 2.5 → expect 3 chunks with overlap
    assert len(chunks) >= 2
    # All chunks must be wider than 0 and cover the full width
    reconstructed_widths = []
    for x_off, _, crop in chunks:
        assert crop.shape[0] == 100
        reconstructed_widths.append((x_off, x_off + crop.shape[1]))
    # First chunk starts at 0
    assert reconstructed_widths[0][0] == 0
    # Last chunk ends at full width
    assert reconstructed_widths[-1][1] == 1000
    # Chunks overlap (each subsequent start < previous end)
    for i in range(1, len(reconstructed_widths)):
        assert reconstructed_widths[i][0] < reconstructed_widths[i-1][1], \
            f"Chunks {i-1} and {i} don't overlap"


def test_split_tall_image_vertically():
    """Tall images should be split vertically."""
    img = np.zeros((1000, 100, 3), dtype=np.uint8)  # aspect 10 (tall)
    chunks = _split_extreme_image(img, max_aspect=4.0)
    assert len(chunks) >= 2
    # First chunk starts at y=0
    assert chunks[0][1] == 0
    # Last chunk ends at full height
    assert chunks[-1][1] + chunks[-1][2].shape[0] == 1000


def test_split_chunks_within_aspect_after_split():
    """After splitting, each chunk's aspect should not exceed max_aspect + slack."""
    img = np.zeros((50, 1500, 3), dtype=np.uint8)  # aspect 30
    chunks = _split_extreme_image(img, max_aspect=4.0)
    max_chunk_aspect = max(c[2].shape[1] / c[2].shape[0] for c in chunks)
    # Allow up to 1.4x the limit due to overlap (10% × 2 overlaps = 20% extra width worst case)
    assert max_chunk_aspect < 4.0 * 1.4, \
        f"chunk aspect {max_chunk_aspect:.2f} too high"


def test_ocr_with_chunking_no_split_calls_run_ocr_once(monkeypatch):
    """Normal aspect images should call _run_ocr once and offset by (0,0)."""
    call_log = []

    def fake_run_ocr(engine, img):
        call_log.append(img)
        return [TextBlock(text="hello", bbox=[10, 20, 100, 50], score=0.9)]

    monkeypatch.setattr("service.exam.ocr._run_ocr", fake_run_ocr)
    img = np.zeros((100, 200, 3), dtype=np.uint8)  # aspect 2.0
    blocks = ocr_with_chunking("engine", img)
    assert len(call_log) == 1
    assert len(blocks) == 1
    assert blocks[0].text == "hello"
    assert blocks[0].bbox == [10, 20, 100, 50]


def test_ocr_with_chunking_wide_image_translates_bboxes(monkeypatch):
    """For wide images, block bboxes from each chunk should be offset by chunk x_offset."""
    # Derive actual chunk count from the splitter so the test stays in sync.
    img = np.zeros((100, 1000, 3), dtype=np.uint8)
    expected_chunks = _split_extreme_image(img, max_aspect=4.0)
    chunks_ocr_results = [
        # Each chunk returns exactly one block at local x=10.
        [TextBlock(text=f"chunk{i}", bbox=[10, 0, 50, 30], score=0.9)]
        for i in range(len(expected_chunks))
    ]
    seen_crops = []

    def fake_run_ocr(engine, crop):
        seen_crops.append(crop)
        return chunks_ocr_results[len(seen_crops) - 1]

    monkeypatch.setattr("service.exam.ocr._run_ocr", fake_run_ocr)
    blocks = ocr_with_chunking("engine", img, max_aspect=4.0)
    # Each chunk returned 1 unique bbox, none overlap → kept all.
    assert len(blocks) == len(expected_chunks)
    # First chunk's block has x=10 (no offset)
    assert blocks[0].bbox[0] == 10
    assert blocks[0].text == "chunk0"
    # Subsequent blocks must have been translated by their chunk's x_offset
    for i, chunk in enumerate(expected_chunks):
        assert blocks[i].bbox[0] == 10 + chunk[0]


def test_ocr_with_chunking_dedups_high_iou_overlap(monkeypatch):
    """Overlapping blocks across chunks (high IoU, same coords after translation) get deduped."""
    img = np.zeros((100, 1000, 3), dtype=np.uint8)
    expected_chunks = _split_extreme_image(img, max_aspect=4.0)
    # Each chunk returns a block whose local bbox, after translation by
    # the chunk's x_offset, lands on the SAME global bbox [10, 0, 50, 30].
    chunks_ocr_results = []
    for x_off, _, _ in expected_chunks:
        local_bbox = [10 - x_off, 0, 50 - x_off, 30]
        chunks_ocr_results.append(
            [TextBlock(text="dup", bbox=local_bbox, score=0.9)]
        )
    seen = []

    def fake_run_ocr(engine, crop):
        seen.append(crop)
        return chunks_ocr_results[len(seen) - 1]

    monkeypatch.setattr("service.exam.ocr._run_ocr", fake_run_ocr)
    blocks = ocr_with_chunking("engine", img, max_aspect=4.0)
    # All chunks produced identical-after-translation bboxes → dedup to 1.
    assert len(blocks) == 1


def test_ocr_with_chunking_keeps_higher_score_when_dedup(monkeypatch):
    """When dedup'ing overlapping blocks, the higher score wins."""
    img = np.zeros((100, 1000, 3), dtype=np.uint8)
    expected_chunks = _split_extreme_image(img, max_aspect=4.0)
    # Same translated bbox across chunks; scores vary.
    chunks_ocr_results = []
    for i, (x_off, _, _) in enumerate(expected_chunks):
        local_bbox = [10 - x_off, 0, 50 - x_off, 30]
        chunks_ocr_results.append(
            [TextBlock(text="same", bbox=local_bbox, score=0.5 + i * 0.1)]
        )
    seen = []

    def fake_run_ocr(engine, crop):
        seen.append(crop)
        return chunks_ocr_results[len(seen) - 1]

    monkeypatch.setattr("service.exam.ocr._run_ocr", fake_run_ocr)
    blocks = ocr_with_chunking("engine", img, max_aspect=4.0)
    assert len(blocks) == 1
    # Highest score is the last chunk's 0.5 + (n-1)*0.1
    expected_max = 0.5 + (len(expected_chunks) - 1) * 0.1
    assert blocks[0].score == pytest.approx(expected_max)


def test_ocr_engine_uses_chunking_for_wide_images(monkeypatch):
    """OcrEngine.ocr_image should automatically chunk wide images."""
    seen_crops = []

    def fake_run_ocr(engine, img):
        seen_crops.append(img)
        return [TextBlock(text="detected", bbox=[5, 10, 30, 30], score=0.9)]

    monkeypatch.setattr("service.exam.ocr._load_paddleocr", lambda: "engine")
    monkeypatch.setattr("service.exam.ocr._run_ocr", fake_run_ocr)

    engine = OcrEngine(lang="ch")
    wide_img = np.zeros((100, 1000, 3), dtype=np.uint8)  # aspect 10
    blocks = engine.ocr_image(wide_img)
    # Should have been split into multiple chunks
    assert len(seen_crops) >= 2
    assert all(b.text == "detected" for b in blocks)
