"""5-stage image enhancement pipeline for exam recognition preprocessing.

Stages: decode → edge_crop → deskew → dewarping → clahe_enhance.
Each stage is independently skippable on failure (records error in stages
table but does not abort the pipeline).
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to BGR uint8 ndarray. Raises ValueError on failure."""
    if not image_bytes:
        raise ValueError("failed to decode image bytes")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("failed to decode image bytes") from exc
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


@dataclass
class StageResult:
    """Outcome of running one enhancement stage."""
    image: np.ndarray
    applied: bool
    duration_ms: int = 0
    error: str | None = None
    payload: dict | None = None


# Minimum allowed output dimension (any smaller is treated as a degenerate warp).
_MIN_CROP_DIM = 64


def _stage_edge_crop(img_bgr: np.ndarray) -> StageResult:
    """Detect the largest 4-point contour and warp the image to its bounding rect.

    Skips (applied=False, error="NO_QUAD_FOUND") when no suitable quad is found
    or when the warped result is below _MIN_CROP_DIM on either side.
    """
    import time
    t0 = time.perf_counter()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return StageResult(
            image=img_bgr, applied=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error="NO_QUAD_FOUND",
        )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    quad = None
    for c in contours[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2)
            break

    if quad is None:
        return StageResult(
            image=img_bgr, applied=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error="NO_QUAD_FOUND",
        )

    # Order points: top-left, top-right, bottom-right, bottom-left.
    s = quad.sum(axis=1)
    diff = np.diff(quad, axis=1).ravel()
    tl, br = quad[np.argmin(s)], quad[np.argmax(s)]
    tr, bl = quad[np.argmin(diff)], quad[np.argmax(diff)]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)

    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bot = np.linalg.norm(ordered[2] - ordered[3])
    height_l = np.linalg.norm(ordered[3] - ordered[0])
    height_r = np.linalg.norm(ordered[2] - ordered[1])
    natural_w = int(max(width_top, width_bot))
    natural_h = int(max(height_l, height_r))
    if natural_w < _MIN_CROP_DIM or natural_h < _MIN_CROP_DIM:
        return StageResult(
            image=img_bgr, applied=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error="NO_QUAD_FOUND",
        )
    out_w, out_h = natural_w, natural_h

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h))

    return StageResult(
        image=warped, applied=True,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        payload={"out_size": {"width": out_w, "height": out_h}},
    )


_MAX_DESKEW_DEG = 15.0


def _stage_deskew(img_bgr: np.ndarray) -> StageResult:
    """Detect dominant text-block rotation and rotate the image to upright.

    Skips when estimated angle exceeds ±15° (degenerate or non-document image).
    """
    import time
    t0 = time.perf_counter()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles: list[float] = []
    for c in contours:
        if cv2.contourArea(c) < 50000:
            continue
        _, _, ang = cv2.minAreaRect(c)
        if ang < -45:
            ang += 90
        elif ang > 45:
            ang -= 90
        angles.append(ang)
    if not angles:
        return StageResult(
            image=img_bgr, applied=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )

    angle = float(np.median(angles))
    # OpenCV positive angle = counter-clockwise. To correct, rotate by -angle.
    if abs(angle) > _MAX_DESKEW_DEG:
        return StageResult(
            image=img_bgr, applied=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            payload={"estimated_angle_deg": angle, "skipped_reason": "ANGLE_OUT_OF_RANGE"},
        )

    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), borderValue=(255, 255, 255))
    return StageResult(
        image=rotated, applied=True,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        payload={"rotated_deg": angle},
    )
