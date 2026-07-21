"""Unit tests for service.exam.enhance helpers and stages."""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from service.exam.enhance import (
    _decode_image_bytes,
    _encode_jpeg_base64,
    _stage_edge_crop,
)


def _make_solid_jpeg_bytes(width: int = 64, height: int = 48, color: tuple[int, int, int] = (200, 200, 200)) -> bytes:
    """Helper: produce a tiny JPEG of a solid-color rectangle."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", arr)
    assert ok
    return bytes(buf)


def test_decode_image_bytes_round_trip():
    raw = _make_solid_jpeg_bytes()
    img = _decode_image_bytes(raw)
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


def test_decode_image_bytes_invalid_raises():
    with pytest.raises(ValueError, match="failed to decode"):
        _decode_image_bytes(b"not an image")


def test_decode_image_bytes_empty_raises():
    with pytest.raises(ValueError, match="failed to decode"):
        _decode_image_bytes(b"")


def test_encode_jpeg_base64_returns_data_url_and_raw_bytes():
    arr = np.full((48, 64, 3), (200, 200, 200), dtype=np.uint8)
    data_url, raw = _encode_jpeg_base64(arr)
    assert data_url.startswith("data:image/jpeg;base64,")
    # raw decodes back to the same bytes advertised in the data URL
    advertised = data_url.split(",", 1)[1]
    assert base64.b64decode(advertised) == raw
    assert raw[:3] == b"\xff\xd8\xff"  # JPEG magic


def _make_quad_test_image() -> np.ndarray:
    """Background-dark image with a white paper quad in the middle."""
    img = np.full((480, 640, 3), 64, dtype=np.uint8)  # dark gray background
    # Quad with slight perspective tilt
    quad = np.array([[120, 100], [540, 80], [560, 400], [100, 420]], dtype=np.int32)
    cv2.fillPoly(img, [quad], (240, 240, 240))
    cv2.putText(img, "HELLO", (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return img


def test_edge_crop_finds_perspective_quad():
    img = _make_quad_test_image()
    result = _stage_edge_crop(img)
    assert result.applied is True, f"edge_crop should detect the quad; got {result.error}"
    h, w = result.image.shape[:2]
    # Warped output should be roughly rectangular and smaller than original.
    assert h < 480
    assert w < 640
    assert h > 100
    assert w > 100


def test_edge_crop_skips_when_no_quad():
    # Solid color, no edges, no quad.
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    result = _stage_edge_crop(img)
    assert result.applied is False
    assert result.error == "NO_QUAD_FOUND"
    assert result.image is img  # passthrough


def test_edge_crop_skips_when_result_too_small():
    # Tiny image where any warp would underflow the size guard.
    img = _make_quad_test_image()
    img = cv2.resize(img, (120, 90))
    result = _stage_edge_crop(img)
    # Either applied=False or applied=True with reasonable size; here expect skip
    # because the quad cannot yield a perspective warp meeting the size threshold.
    assert result.applied is False
