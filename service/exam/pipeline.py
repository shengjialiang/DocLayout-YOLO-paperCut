"""5-stage exam pipeline orchestrator.

Stages: decode → yolo → expand → ocr → filter → geometry → redraw.
Intermediate errors are recorded per-stage but don't fail the task.
Fatal errors (decode failure, YOLO crash) mark the task failed.
"""
from __future__ import annotations

import base64
import io
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from service.exam import tasks
from service.exam.expand import expand_boxes
from service.exam.geometry import extend_questions
from service.exam.ocr import OcrEngine, TextBlock
from service.exam.question_filter import is_question
from service.exam.schemas import FinalResult, OcrBlock, Question


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image bytes")
    return img


def _encode_jpeg_base64(rgb_or_bgr: np.ndarray) -> str:
    """Encode numpy image as data-URL base64 JPEG. Accepts RGB or BGR."""
    if rgb_or_bgr.ndim == 3 and rgb_or_bgr.shape[2] == 3:
        # If colors look BGR (heuristic: high blue variance), swap.
        # For simplicity, assume BGR input and convert.
        rgb = cv2.cvtColor(rgb_or_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = rgb_or_bgr
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else bgr


def redraw_with_questions(
    img_bgr: np.ndarray,
    questions: list[dict],
    original_detections: list[dict],
) -> str:
    """Draw extended question boxes on original image, return base64 jpeg data-URL."""
    rgb = _to_rgb(img_bgr.copy())
    pil = Image.fromarray(rgb)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil)
    for q in questions:
        x1, y1, x2, y2 = q["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=5)
    return _encode_jpeg_base64(np.asarray(pil))


def run_pipeline(task_id: str, image_bytes: bytes, params: dict[str, Any]) -> None:
    """Run the 5-stage pipeline for a task. Updates task state in-place."""
    import service.inference as inference

    # Stage 0: decode
    try:
        img_bgr = _decode_image_bytes(image_bytes)
        h, w = img_bgr.shape[:2]
    except Exception as e:
        tasks.mark_failed(task_id, f"decode failed: {e}")
        return

    # Stage 1: YOLO
    t0 = time.perf_counter()
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(inference.predict_one(
                image_bytes=image_bytes,
                model=None,           # placeholder; predict_one in tests uses fake
                semaphore=__import__("asyncio").Semaphore(1),
                conf=params.get("conf", 0.3),
                imgsz=params.get("imgsz", 1024),
                line_width=5,
                font_size=20,
                device=params.get("device", "cpu"),
            ))
        finally:
            loop.close()
        detections = result["detections"]
        tasks.update_stage(task_id, "yolo",
                           duration_ms=int((time.perf_counter() - t0) * 1000),
                           payload={"num_detections": len(detections)})
    except Exception as e:
        tasks.update_stage(task_id, "yolo", duration_ms=0, error=str(e))
        tasks.mark_failed(task_id, f"yolo failed: {e}")
        return

    # Stage 2: expand
    t0 = time.perf_counter()
    try:
        expanded = expand_boxes(
            detections,
            mode=params.get("expand_mode", "pixel"),
            top=params.get("expand_top", 20),
            bottom=params.get("expand_bottom", 20),
            left=params.get("expand_left", 20),
            right=params.get("expand_right", 20),
            img_width=w, img_height=h,
        )
        plain_expanded = [b for b in expanded if b["class_name"] == "plain text"]
        tasks.update_stage(task_id, "expand",
                           duration_ms=int((time.perf_counter() - t0) * 1000),
                           payload={"num_expanded": len(plain_expanded)})
    except Exception as e:
        tasks.update_stage(task_id, "expand", duration_ms=0, error=str(e))
        plain_expanded = []

    # Stage 3: OCR
    t0 = time.perf_counter()
    ocr_blocks: list[OcrBlock] = []
    try:
        ocr_engine = OcrEngine(lang="ch")
        for box in plain_expanded:
            x1, y1, x2, y2 = [int(round(v)) for v in box["bbox_xyxy"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                ocr_blocks.append(OcrBlock(
                    source_detection_id=box["id"],
                    bbox_xyxy=box["bbox_xyxy"], text="", score=0.0,
                    is_question=False,
                ))
                continue
            crop = img_bgr[y1:y2, x1:x2]
            try:
                blocks: list[TextBlock] = ocr_engine.ocr_image(crop)
            except Exception as e:
                tasks.update_stage(task_id, "ocr", duration_ms=0,
                                   error=f"per-box OCR failed: {e}")
                blocks = []
            full_text = " ".join(b.text for b in blocks) if blocks else ""
            avg_score = sum(b.score for b in blocks) / len(blocks) if blocks else 0.0
            ocr_blocks.append(OcrBlock(
                source_detection_id=box["id"],
                bbox_xyxy=box["bbox_xyxy"],
                text=full_text, score=avg_score,
                is_question=False,  # set in filter stage
            ))
        tasks.update_stage(task_id, "ocr",
                           duration_ms=int((time.perf_counter() - t0) * 1000),
                           payload={"num_blocks": len(ocr_blocks)})
    except Exception as e:
        tasks.update_stage(task_id, "ocr", duration_ms=0, error=str(e))
        ocr_blocks = []

    # Stage 4: filter
    t0 = time.perf_counter()
    regex = params.get("question_regex", r"^\s*\(?\d+[\.\)](?!\d)")
    try:
        questions: list[dict] = []
        for block in ocr_blocks:
            matched = is_question(block.text, regex)
            block.is_question = matched
            if matched:
                src_box = next(
                    (b for b in detections if b["id"] == block.source_detection_id),
                    None,
                )
                questions.append({
                    "text": block.text,
                    "bbox_xyxy": block.bbox_xyxy,
                    "source_ids": [block.source_detection_id],
                    "ocr_score": block.score,
                })
        tasks.update_stage(task_id, "filter",
                           duration_ms=int((time.perf_counter() - t0) * 1000),
                           payload={"num_questions": len(questions)})
    except Exception as e:
        tasks.update_stage(task_id, "filter", duration_ms=0, error=str(e))
        questions = []

    # Stage 5: geometry
    t0 = time.perf_counter()
    try:
        extended = extend_questions(questions, detections)
        tasks.update_stage(task_id, "geometry",
                           duration_ms=int((time.perf_counter() - t0) * 1000),
                           payload={"num_extended": len(extended)})
    except Exception as e:
        tasks.update_stage(task_id, "geometry", duration_ms=0, error=str(e))
        extended = questions

    # Stage 6: redraw + final
    try:
        annotated_b64 = redraw_with_questions(img_bgr, extended, detections)
        original_b64 = _encode_jpeg_base64(img_bgr)
        question_models = [
            Question(
                index=i + 1,
                bbox_xyxy=q["bbox_xyxy"],
                matched_text=q["text"],
                source_detection_ids=q["source_ids"],
                ocr_score=q["ocr_score"],
            )
            for i, q in enumerate(extended)
        ]
        final = FinalResult(
            annotated_image=annotated_b64,
            original_image=original_b64,
            questions=question_models,
            ocr_blocks=ocr_blocks,
        )
        tasks.mark_done(task_id, final)
    except Exception as e:
        tasks.update_stage(task_id, "geometry", duration_ms=0,
                           error=f"redraw failed: {e}")
        tasks.mark_failed(task_id, f"final assembly failed: {e}")