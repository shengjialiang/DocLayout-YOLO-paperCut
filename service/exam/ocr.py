"""PaddleOCR engine wrapper with lazy loading."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Above this aspect ratio, PaddleOCR's DB text detector starts missing
# text lines. Empirical threshold based on real-world exam crops that
# come back from YOLO as plain-text bboxes.
MAX_ASPECT_RATIO = 4.0

# Overlap between adjacent chunks when splitting wide images.
# 10% of chunk width on each side so adjacent chunks have ~20% total overlap
# to avoid losing characters at chunk boundaries.
CHUNK_OVERLAP_RATIO = 0.1

# Deduplication threshold: if two blocks have IoU > this, the lower-scored
# one is dropped. Used to remove duplicate detections in chunk overlap regions.
DEDUP_IOU_THRESHOLD = 0.3


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


def _split_extreme_image(
    img: np.ndarray,
    max_aspect: float = MAX_ASPECT_RATIO,
) -> list[tuple[int, int, np.ndarray]]:
    """Split an image whose aspect ratio exceeds ``max_aspect``.

    Returns a list of ``(x_offset, y_offset, crop)`` tuples that, taken
    together, cover the entire input image. The first chunk always starts
    at (0, 0) and the last one ends at the input image's bottom-right
    corner.

    For images with reasonable aspect ratio, returns a single full-image
    chunk with offset (0, 0).
    """
    H, W = img.shape[:2]
    aspect_w = W / H
    aspect_h = H / W

    if aspect_w <= max_aspect and aspect_h <= max_aspect:
        return [(0, 0, img)]

    # Choose split direction based on which dimension is extreme.
    if aspect_w >= aspect_h:
        # Wide image: split horizontally. Each chunk's width is at most
        # max_aspect * H, plus overlap on each side.
        chunk_w = int(H * max_aspect)
        overlap = max(1, int(chunk_w * CHUNK_OVERLAP_RATIO))
        # Compute n_chunks so that chunks cover full width with overlaps.
        # effective_coverage = n * chunk_w - (n-1) * overlap >= W
        #  → n >= (W - overlap) / (chunk_w - overlap)
        n_chunks = max(2, -(-(W - overlap) // (chunk_w - overlap)))  # ceil div
        step = (W - overlap) // n_chunks
        chunks: list[tuple[int, int, np.ndarray]] = []
        for i in range(n_chunks):
            x1 = i * step
            if i < n_chunks - 1:
                x2 = (i + 1) * step + overlap
            else:
                x2 = W  # ensure last chunk reaches the right edge
            chunks.append((x1, 0, img[:, x1:x2]))
        return chunks
    else:
        # Tall image: split vertically.
        chunk_h = int(W * max_aspect)
        overlap = max(1, int(chunk_h * CHUNK_OVERLAP_RATIO))
        n_chunks = max(2, -(-(H - overlap) // (chunk_h - overlap)))
        step = (H - overlap) // n_chunks
        chunks = []
        for i in range(n_chunks):
            y1 = i * step
            if i < n_chunks - 1:
                y2 = (i + 1) * step + overlap
            else:
                y2 = H
            chunks.append((0, y1, img[y1:y2, :]))
        return chunks


def _iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Compute IoU between two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _dedup_blocks(blocks: list[TextBlock], iou_threshold: float = DEDUP_IOU_THRESHOLD) -> list[TextBlock]:
    """Remove duplicate blocks via IoU-based match, keeping the higher score.

    Useful after splitting wide images into overlapping chunks where the
    same text region may be detected twice with slightly different bboxes.
    """
    if not blocks:
        return blocks
    # Highest score first; first-seen (highest score) wins on duplicate IoU.
    ordered = sorted(blocks, key=lambda b: -b.score)
    kept: list[TextBlock] = []
    for b in ordered:
        dup = False
        for k in kept:
            if _iou(b.bbox, k.bbox) > iou_threshold:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def ocr_with_chunking(
    engine,
    img: np.ndarray,
    max_aspect: float = MAX_ASPECT_RATIO,
) -> list[TextBlock]:
    """OCR ``img``, automatically chunking extreme aspect ratios.

    Translates per-chunk bboxes back into the original image coordinate
    system and removes duplicate detections across overlapping chunks.
    """
    chunks = _split_extreme_image(img, max_aspect=max_aspect)
    blocks: list[TextBlock] = []
    for x_off, y_off, crop in chunks:
        for b in _run_ocr(engine, crop):
            blocks.append(TextBlock(
                text=b.text,
                bbox=[b.bbox[0] + x_off, b.bbox[1] + y_off,
                      b.bbox[2] + x_off, b.bbox[3] + y_off],
                score=b.score,
            ))
    return _dedup_blocks(blocks)


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
        return ocr_with_chunking(self._engine, img)
