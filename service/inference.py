"""Model loading and inference execution."""
from __future__ import annotations

import base64
import io
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
from doclayout_yolo import YOLOv10


def parse_detections(result: Any) -> list[dict]:
    """Convert a single doclayout_yolo Results object to a JSON-serializable list.

    Each entry: {id, class_id, class_name, bbox_xyxy, bbox_xywh, score}.
    bbox_xyxy: [x1, y1, x2, y2]  (original image pixel coords)
    bbox_xywh: [cx, cy, w, h]   (center-format)
    """
    if result.boxes is None:
        return []

    xyxy = np.asarray(result.boxes.xyxy, dtype=float)
    conf = np.asarray(result.boxes.conf, dtype=float)
    cls = np.asarray(result.boxes.cls, dtype=int)
    names = result.names  # {int: str} mapping

    if len(xyxy) == 0:
        return []

    detections = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        detections.append({
            "id": i,
            "class_id": int(cls[i]),
            "class_name": names.get(int(cls[i]), str(int(cls[i]))),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": [cx, cy, w, h],
            "score": float(conf[i]),
        })
    return detections


def load_model(model_path: str):
    """Instantiate a YOLOv10 model from path."""
    return YOLOv10(model_path)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to BGR numpy array. Raises ValueError if decode fails."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image bytes (cv2.imdecode returned None)")
    return img


def _encode_jpeg_base64(rgb: np.ndarray) -> str:
    """Encode an RGB numpy uint8 image to data-URL base64 jpeg string."""
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


async def predict_one(
    *,
    image_bytes: bytes,
    model,
    semaphore,
    conf: float,
    imgsz: int,
    line_width: int,
    font_size: int,
    device: str,
    timeout: float = 60.0,
) -> dict:
    """Run inference on one image. Concurrency-limited by semaphore."""
    img_bgr = _decode_image(image_bytes)
    h, w = img_bgr.shape[:2]

    async with semaphore:
        t0 = time.perf_counter()
        det_res = model.predict(
            img_bgr,
            imgsz=imgsz,
            conf=conf,
            device=device,
        )
        annotated_rgb = det_res[0].plot(
            pil=True,
            line_width=line_width,
            font_size=font_size,
        )
        if isinstance(annotated_rgb, Image.Image):
            annotated_rgb = np.asarray(annotated_rgb)
        inference_ms = int((time.perf_counter() - t0) * 1000)

    detections = parse_detections(det_res[0])
    return {
        "width": w,
        "height": h,
        "annotated_base64": _encode_jpeg_base64(annotated_rgb),
        "detections": detections,
        "num_detections": len(detections),
        "inference_time_ms": inference_ms,
        "model_imgsz": imgsz,
        "conf_threshold": conf,
        "device": device,
    }