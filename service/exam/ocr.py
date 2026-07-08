"""PaddleOCR engine wrapper with lazy loading."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        return _run_ocr(self._engine, img)
