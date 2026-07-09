"""PaddleOCR engine wrapper with lazy loading."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

# Above this aspect ratio, PaddleOCR's DB text detector fails to find most
# text lines (returns ~1 block instead of the full text). Empirically
# determined from real exam crops produced by the YOLO detector.
MAX_ASPECT_RATIO = 4.0

# Below this absolute height, PaddleOCR's DB detector returns zero (or
# garbled) detections even after aspect-ratio chunking — its feature
# pyramid (stride ~32) collapses to a single row, leaving text instances
# sub-pixel. White-space padding to this height restores detection on
# short crops produced by the YOLO detector. Empirically validated on
# real exam crops where 94-px-tall crops return 0 detections but
# 128-px-tall (padded) crops correctly detect the question number.
MIN_OCR_HEIGHT = 128

# Process-wide singleton state. Double-checked locking in get_engine() so
# first-call concurrent access still triggers exactly one PaddleOCR load.
_engine_lock = threading.Lock()
_engine: "object | None" = None


def get_engine(lang: str = "ch"):
    """Return the process-wide singleton PaddleOCR engine (lazy + thread-safe)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = _load_paddleocr(lang=lang)
    return _engine


def reset_engine_singleton() -> None:
    """Test helper: drop the cached engine so the next get_engine() reloads."""
    global _engine
    with _engine_lock:
        _engine = None


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


def _ensure_min_height(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad ``img`` vertically with white space if below MIN_OCR_HEIGHT.

    Returns ``(padded_img, top_pad)``. ``top_pad`` is the number of
    pixels added above the original image (0 if no padding was needed),
    so callers can translate bbox y-coordinates back to the original
    coordinate system.
    """
    h = img.shape[0]
    if h >= MIN_OCR_HEIGHT:
        return img, 0
    pad_top = (MIN_OCR_HEIGHT - h) // 2
    pad_bottom = MIN_OCR_HEIGHT - h - pad_top
    padded = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, 0, 0,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )
    return padded, pad_top


def _run_ocr(engine, img: np.ndarray) -> list[TextBlock]:
    """Run PaddleOCR and convert results to TextBlock list.

    Pads ``img`` vertically to at least MIN_OCR_HEIGHT before calling
    PaddleOCR (so very short crops don't collapse the DB feature pyramid),
    then translates returned bbox y-coords back to the original
    (unpadded) coordinate system.
    """
    img, pad_top = _ensure_min_height(img)
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
        y1, y2 = min(ys) - pad_top, max(ys) - pad_top
        blocks.append(TextBlock(text=text, bbox=[x1, y1, x2, y2], score=score))
    return blocks


def _split_into_chunks(
    img: np.ndarray,
    max_aspect: float = MAX_ASPECT_RATIO,
) -> list[tuple[int, int, np.ndarray]]:
    """Split an image into multiple overlapping chunks along the longer axis.

    Returns ``[(x_offset, y_offset, crop), ...]`` that together cover the
    entire input image. For normal-aspect images there is a single
    full-image chunk with offset (0, 0); for wide images the chunks tile
    horizontally with overlap; for tall images they tile vertically.

    Each chunk has aspect ratio ≤ ``max_aspect``.
    """
    H, W = img.shape[:2]
    aspect_w = W / H
    aspect_h = H / W

    if aspect_w <= max_aspect and aspect_h <= max_aspect:
        return [(0, 0, img)]

    if aspect_w >= aspect_h:
        # Wide image: tile horizontally.
        chunk_w = int(H * max_aspect)
        overlap = max(1, int(chunk_w * 0.1))
        n_chunks = max(2, -(-(W - overlap) // (chunk_w - overlap)))
        step = max(1, (W - overlap) // n_chunks)
        chunks: list[tuple[int, int, np.ndarray]] = []
        for i in range(n_chunks):
            x1 = i * step
            x2 = (i + 1) * step + overlap if i < n_chunks - 1 else W
            chunks.append((x1, 0, img[:, x1:x2]))
        return chunks
    else:
        # Tall image: tile vertically.
        chunk_h = int(W * max_aspect)
        overlap = max(1, int(chunk_h * 0.1))
        n_chunks = max(2, -(-(H - overlap) // (chunk_h - overlap)))
        step = max(1, (H - overlap) // n_chunks)
        chunks = []
        for i in range(n_chunks):
            y1 = i * step
            y2 = (i + 1) * step + overlap if i < n_chunks - 1 else H
            chunks.append((0, y1, img[y1:y2, :]))
        return chunks


def _leftmost_chunk(
    img: np.ndarray,
    max_aspect: float = MAX_ASPECT_RATIO,
) -> np.ndarray:
    """Return just the first chunk from ``_split_into_chunks``.

    Always picks the (topmost, leftmost) chunk, regardless of whether the
    image is normal or extreme aspect. For normal-aspect images where the
    split collapses to a single full-image chunk, this returns the whole
    image.
    """
    chunks = _split_into_chunks(img, max_aspect=max_aspect)
    return chunks[0][2]


def ocr_leftmost(
    engine,
    img: np.ndarray,
    max_aspect: float = MAX_ASPECT_RATIO,
) -> list[TextBlock]:
    """OCR only the leftmost (or topmost) chunk of ``img``.

    The leading digits of a question number (e.g. "5.") always sit at the
    leftmost edge of its bounding box, so recognizing just that one chunk
    is enough to decide whether the original crop is a question box — and
    PaddleOCR's DB detector only works reliably on chunks whose aspect
    ratio is sane, not on full extreme-aspect crops.

    Always uses the leftmost chunk, even when the aspect ratio is already
    within budget: in that case the chunk collapses to the whole image.
    """
    crop = _leftmost_chunk(img, max_aspect=max_aspect)
    return _run_ocr(engine, crop)


class OcrEngine:
    """Thin wrapper that delegates to the process-wide PaddleOCR singleton.

    Construction is zero-cost: no engine state is held on the instance.
    Backward-compatible with the original per-instance caching API.
    """

    def __init__(self, lang: str = "ch") -> None:
        self._lang = lang

    def ocr_image(self, img: np.ndarray) -> list[TextBlock]:
        return ocr_leftmost(get_engine(self._lang), img)
