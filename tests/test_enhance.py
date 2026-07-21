"""Unit tests for service.exam.enhance helpers and stages."""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from service.exam.enhance import (
    _decode_image_bytes,
    _encode_jpeg_base64,
    _stage_clahe_enhance,
    _stage_deskew,
    _stage_dewarping,
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


def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))


def _estimate_skew_angle_deg(img_bgr: np.ndarray) -> float:
    """Return the dominant text-block angle in degrees (positive = CCW)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        if cv2.contourArea(c) < 50000:
            continue
        _, _, ang = cv2.minAreaRect(c)
        # OpenCV minAreaRect angle: between -90 and 0. Normalize to ±45.
        if ang < -45:
            ang += 90
        elif ang > 45:
            ang -= 90
        angles.append(ang)
    if not angles:
        return 0.0
    # Use median to be robust to outliers.
    return float(np.median(angles))


def test_deskew_corrects_small_rotation():
    base = _make_quad_test_image()
    rotated = _rotate_image(base, 5.0)
    pre_angle = abs(_estimate_skew_angle_deg(rotated))
    assert pre_angle > 1.0, "test sanity: rotated image must report a non-trivial skew"
    result = _stage_deskew(rotated)
    assert result.applied is True
    post_angle = abs(_estimate_skew_angle_deg(result.image))
    assert post_angle < 1.5, f"residual skew {post_angle} should be < 1.5°"


def test_deskew_skips_extreme_angle():
    base = _make_quad_test_image()
    rotated = _rotate_image(base, 25.0)
    result = _stage_deskew(rotated)
    assert result.applied is False


def test_clahe_preserves_shape_and_dtype():
    img = _make_quad_test_image()
    result = _stage_clahe_enhance(img)
    assert result.applied is True
    assert result.image.shape == img.shape
    assert result.image.dtype == img.dtype


def test_clahe_increases_luminance_diversity():
    # Low-contrast image: CLAHE should pull histogram apart.
    img = np.full((200, 300, 3), 128, dtype=np.uint8)
    # Add a few subtle gray patches to give CLAHE something to work with.
    cv2.rectangle(img, (50, 50), (120, 120), (110, 110, 110), -1)
    cv2.rectangle(img, (180, 80), (240, 160), (140, 140, 140), -1)

    pre_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pre_std = float(pre_gray.std())

    result = _stage_clahe_enhance(img)
    post_gray = cv2.cvtColor(result.image, cv2.COLOR_BGR2GRAY)
    post_std = float(post_gray.std())

    assert post_std >= pre_std, (
        f"CLAHE should preserve or increase contrast (pre={pre_std:.2f}, post={post_std:.2f})"
    )


def test_dewarping_skips_when_no_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("service.exam.enhance._docrect_model_path", None)
    img = _make_quad_test_image()
    result = _stage_dewarping(img)
    assert result.applied is False
    assert "DOCRECT_UNAVAILABLE" in (result.error or "")


def test_dewarping_passthrough_when_model_callable(monkeypatch: pytest.MonkeyPatch):
    # Stub model: just return the input unchanged.
    monkeypatch.setattr("service.exam.enhance._docrect_model_path", "/fake/path/model.onnx")

    def fake_apply(img, model_path):
        assert model_path == "/fake/path/model.onnx"
        return img.copy()

    monkeypatch.setattr("service.exam.enhance._apply_docrect", fake_apply)
    img = _make_quad_test_image()
    result = _stage_dewarping(img)
    assert result.applied is True
    assert result.image.shape == img.shape


# --- Orchestrator tests ---

from service.exam.enhance import run_enhance_pipeline
from service.exam import tasks
from service.exam.schemas import EnhancementFinal


def _patch_all_stages_apply(monkeypatch: pytest.MonkeyPatch):
    """Make every stage return the input unchanged but applied=True. Used to
    test the orchestrator wiring without depending on real CV outcomes."""
    def passthrough_stage(img):
        from service.exam.enhance import StageResult
        return StageResult(image=img, applied=True)
    for name in ("_stage_edge_crop", "_stage_deskew", "_stage_clahe_enhance", "_stage_dewarping"):
        monkeypatch.setattr(f"service.exam.enhance.{name}", passthrough_stage)


def test_run_enhance_pipeline_marks_done_with_enhancement_final(monkeypatch: pytest.MonkeyPatch):
    _patch_all_stages_apply(monkeypatch)
    monkeypatch.setattr("service.exam.enhance._docrect_model_path", None)
    tasks.reset_for_tests()
    tid = tasks.create_task()

    raw = _make_solid_jpeg_bytes(width=100, height=80, color=(180, 180, 180))
    run_enhance_pipeline(tid, raw, params={"docrect_model_path": None})

    state = tasks.get_task(tid)
    assert state is not None
    assert state["status"] == "done"
    assert state["final"] is not None
    # final is an EnhancementFinal dataclass
    assert isinstance(state["final"], EnhancementFinal)
    assert state["final"].applied_stages  # at least one stage applied
    assert state["final"].output_size["width"] > 0
    assert state["final"].output_size["height"] > 0
    # Stages table has entries
    assert "decode" in state["stages"]
    assert "edge_crop" in state["stages"]


def test_run_enhance_pipeline_marks_failed_on_decode_error(monkeypatch: pytest.MonkeyPatch):
    tasks.reset_for_tests()
    tid = tasks.create_task()
    run_enhance_pipeline(tid, b"garbage", params={})
    state = tasks.get_task(tid)
    assert state is not None
    assert state["status"] == "failed"
    assert "decode failed" in (state["error"] or "")


def test_run_enhance_pipeline_all_stages_skip_keeps_original(monkeypatch: pytest.MonkeyPatch):
    """When every stage skips, enhanced_image == original_image."""
    def always_skip(img):
        from service.exam.enhance import StageResult
        return StageResult(image=img, applied=False)
    for name in ("_stage_edge_crop", "_stage_deskew", "_stage_clahe_enhance", "_stage_dewarping"):
        monkeypatch.setattr(f"service.exam.enhance.{name}", always_skip)

    tasks.reset_for_tests()
    tid = tasks.create_task()
    raw = _make_solid_jpeg_bytes(width=80, height=60, color=(200, 200, 200))
    run_enhance_pipeline(tid, raw, params={})

    state = tasks.get_task(tid)
    assert state is not None
    assert state["status"] == "done"
    final = state["final"]
    assert isinstance(final, EnhancementFinal)
    assert final.applied_stages == []
    assert final.original_image == final.enhanced_image


# --- DocRect integration tests ---

def test_apply_docrect_prefers_onnx(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """When ONNX file is provided, _load_docrect_session returns an onnxruntime InferenceSession."""
    fake_onnx = tmp_path / "docrect.onnx"
    fake_onnx.write_bytes(b"")

    monkeypatch.setattr("service.exam.enhance._HAS_ONNX", True)

    called = {"ort": False, "torch": False}

    class FakeSession:
        def __init__(self, path):
            assert str(path) == str(fake_onnx)

    def fake_ort_load(path):
        called["ort"] = True
        return FakeSession(path)

    monkeypatch.setattr(
        "service.exam.enhance._load_onnx_session", fake_ort_load
    )
    monkeypatch.setattr(
        "service.exam.enhance._load_torch_session",
        lambda path: called.__setitem__("torch", True) or None,
    )

    from service.exam.enhance import _load_docrect_session
    sess = _load_docrect_session(str(fake_onnx))
    assert called["ort"] is True
    assert called["torch"] is False


def test_apply_docrect_falls_back_to_torch(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """When ONNX is unavailable or missing, falls back to PyTorch .pth."""
    fake_pth = tmp_path / "docrect.pth"
    fake_pth.write_bytes(b"")

    monkeypatch.setattr("service.exam.enhance._HAS_ONNX", False)
    called = {"torch": False}

    def fake_torch_load(path):
        called["torch"] = True
        assert str(path) == str(fake_pth)
        return "torch-model-stub"

    monkeypatch.setattr(
        "service.exam.enhance._load_torch_session", fake_torch_load
    )

    from service.exam.enhance import _load_docrect_session
    sess = _load_docrect_session(str(fake_pth))
    assert called["torch"] is True
    assert sess == "torch-model-stub"


def test_apply_docrect_raises_when_no_backend(monkeypatch: pytest.MonkeyPatch, tmp_path):
    fake = tmp_path / "docrect.onnx"
    fake.write_bytes(b"")
    monkeypatch.setattr("service.exam.enhance._HAS_ONNX", False)

    # Make torch load raise to simulate unavailable backend.
    def raise_load(path):
        raise RuntimeError("torch not installed")

    monkeypatch.setattr("service.exam.enhance._load_torch_session", raise_load)

    from service.exam.enhance import _load_docrect_session
    import pytest
    with pytest.raises(RuntimeError, match="torch not installed"):
        _load_docrect_session(str(fake))
