"""Model loading and inference execution."""
from __future__ import annotations

import io
import base64
from typing import Any

import cv2
import numpy as np
from PIL import Image


def parse_detections(result: Any) -> list[dict]:
    """Convert a single doclayout_yolo Results object to a JSON-serializable list.

    Each entry: {id, class_id, class_name, bbox_xyxy, bbox_xywh, score}.
    bbox_xyxy: [x1, y1, x2, y2]  (original image pixel coords)
    bbox_xywh: [cx, cy, w, h]   (center-format)
    """
    if result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = np.asarray(result.boxes.xyxy, dtype=float)
    conf = np.asarray(result.boxes.conf, dtype=float)
    cls = np.asarray(result.boxes.cls, dtype=int)
    names = result.names  # {int: str} mapping

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
