"""PaddleOCR engine wrapper with lazy loading."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Above this aspect ratio, PaddleOCR's DB text detector fails to find most
# text lines (returns ~1 block instead of the full text). Empirically
# determined from real exam crops produced by the YOLO detector.
MAX_ASPECT_RATIO = 4.0


@dataclass
class TextBlock:
    """Single OCR result block."""
    text: str
    bbox: list[float]   # [x1, y1, x2, y2]
    score: float


def _load_paddleocr(lang: str = "ch"):
    """Load PaddleOCR (lazy import to avoid heavy startup cost)."""
    from paddleocr import PaddleOCR  # type: ignore
    return PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)


def _run_ocr(engine, img: np.ndarray) -> list[TextBlock]:
    """Run PaddleOCR and convert results to TextBlock list."""
    raw = engine.ocr(img, cls=False)
    blocks: list[TextBlock] = []
    if not raw or not raw[0]:
        return blocks
    for line in raw[0]:
        # line = [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, score)]
        box_pts = line[0]
        text_score = line[1]
        text = text_score[0]
        score = float(text_score[1])
        xs = [p[0] for p in box_pts]
        ys = [p[1] for p in box_pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        blocks.append(TextBlock(text=text, bbox=[x1, y1, x2, y2], score=score))
    return blocks


def _leftmost_chunk(
    img: np.ndarray,
    max_aspect: float = MAX_ASPECT_RATIO,
) -> np.ndarray:
    """Crop an extreme-aspect image to its leftmost (or topmost) chunk.

    For a wide image, takes the leftmost columns; for a tall image, takes
    the topmost rows. The returned chunk has aspect ratio ≤ ``max_aspect``.

    For images within the aspect budget, returns the input unchanged.
    """
    H, W = img.shape[:2]
    aspect_w = W / H
    aspect_h = H / W
    if aspect_w <= max_aspect and aspect_h <= max_aspect:
        return img
    if aspect_w >= aspect_h:
        chunk_w = min(W, int(H * max_aspect))
        return img[:, :chunk_w]
    else:
        chunk_h = min(H, int(W * max_aspect))
        return img[:chunk_h, :]


def ocr_leftmost(
    engine,
    img: np.ndarray,
    max_aspect: float = MAX_ASPECT_RATIO,
) -> list[TextBlock]:
    """OCR ``img``; for extreme aspect ratios, OCR only the leftmost chunk.

    The leading digits of a question number (e.g. "5.") always sit at the
    leftmost edge of its bounding box, so recognizing just the first chunk
    is enough to decide whether the original crop is a question box — and
    PaddleOCR's detector only works reliably on that one chunk, not on the
    full image when the aspect ratio is extreme.
    """
    crop = _leftmost_chunk(img, max_aspect=max_aspect)
    return _run_ocr(engine, crop)


class OcrEngine:
    """Lazy-loaded PaddleOCR engine.

    PaddleOCR models are not loaded at construction time. The first call to
    `ocr_image` triggers loading. Subsequent calls reuse the loaded engine.
    """

    def __init__(self, lang: str = "ch") -> None:
        self._lang = lang
        self._engine = None

    def ocr_image(self, img: np.ndarray) -> list[TextBlock]:
        if self._engine is None:
            # Call without args: tests monkeypatch with no-arg fakes; the
            # production `_load_paddleocr(lang="ch")` has a default value.
            self._engine = _load_paddleocr()
        return ocr_leftmost(self._engine, img)
