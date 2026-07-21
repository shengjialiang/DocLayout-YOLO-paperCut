"""5-stage image enhancement pipeline for exam recognition preprocessing.

Stages: decode → edge_crop → deskew → dewarping → clahe_enhance.
Each stage is independently skippable on failure (records error in stages
table but does not abort the pipeline).
"""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to BGR uint8 ndarray. Raises ValueError on failure."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image bytes")
    return img


def _encode_jpeg_base64(rgb_or_bgr: np.ndarray) -> tuple[str, bytes]:
    """Encode image as (data_url, raw_jpeg_bytes). Accepts RGB or BGR.

    Heuristic: if 3-channel, treat as BGR (matches OpenCV convention) and
    convert to RGB before JPEG encoding. Returns the raw JPEG bytes so
    callers can also persist them without re-encoding.
    """
    if rgb_or_bgr.ndim == 3 and rgb_or_bgr.shape[2] == 3:
        rgb = cv2.cvtColor(rgb_or_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = rgb_or_bgr
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    data_url = "data:image/jpeg;base64," + b64
    return data_url, raw
