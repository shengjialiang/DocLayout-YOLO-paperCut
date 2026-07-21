"""Unit tests for service.exam.enhance helpers and stages."""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from service.exam.enhance import _decode_image_bytes, _encode_jpeg_base64


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
