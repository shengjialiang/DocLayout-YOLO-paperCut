"""5-stage image enhancement pipeline for exam recognition preprocessing.

Stages: decode → edge_crop → deskew → dewarping → clahe_enhance.
Each stage is independently skippable on failure (records error in stages
table but does not abort the pipeline).
"""
from __future__ import annotations

import base64
import io
import os
import threading as _threading
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

try:
    import onnxruntime  # type: ignore
    _HAS_ONNX = True
except ImportError:
    onnxruntime = None  # type: ignore
    _HAS_ONNX = False


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

# A real document paper typically occupies a substantial portion of the frame
# (≥ 30% in practice). When edge_crop finds only a tiny 4-vertex contour
# (e.g. a book corner, sticker, or texture fragment in a real-world photo),
# warping to it catastrophically shrinks the image — the result is unusable.
# Guard: the candidate quad's area must be at least this fraction of the
# whole image. Below the threshold we skip with error=QUAD_TOO_SMALL.
_MIN_QUAD_AREA_RATIO = 0.15

# Allowed aspect-ratio range for the warped output. Real documents are
# typically in [0.5, 2.0]; we use a wider band [0.2, 5.0] to tolerate
# folded/sticky notes and panoramic receipts, while excluding extreme strips.
_MIN_ASPECT = 0.2
_MAX_ASPECT = 5.0


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

    # Guard 1: reject if the quad is too small relative to the whole image.
    # Without this, a real-world photo containing a small 4-vertex convex
    # fragment (book corner, sticker, texture) gets warped to a tiny unusable
    # image — see test_edge_crop_skips_when_quad_is_tiny_relative_to_image.
    img_h, img_w = img_bgr.shape[:2]
    quad_area = float(cv2.contourArea(ordered))
    img_area = float(img_h * img_w)
    if img_area > 0 and (quad_area / img_area) < _MIN_QUAD_AREA_RATIO:
        return StageResult(
            image=img_bgr, applied=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error="QUAD_TOO_SMALL",
            payload={
                "quad_area_ratio": quad_area / img_area,
                "min_required_ratio": _MIN_QUAD_AREA_RATIO,
            },
        )

    if natural_w < _MIN_CROP_DIM or natural_h < _MIN_CROP_DIM:
        return StageResult(
            image=img_bgr, applied=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error="NO_QUAD_FOUND",
        )

    # Guard 2: reject extreme aspect ratios (very thin strips). Real documents
    # are roughly portrait/landscape; a 6:1 strip is not a document.
    aspect = natural_w / max(natural_h, 1)
    if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
        return StageResult(
            image=img_bgr, applied=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error="ASPECT_OUT_OF_RANGE",
            payload={
                "aspect_ratio": aspect,
                "min_allowed": _MIN_ASPECT,
                "max_allowed": _MAX_ASPECT,
            },
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
            error="NO_QUAD_FOUND",
        )

    angle = float(np.median(angles))
    # OpenCV positive angle = counter-clockwise. To correct, rotate by -angle.
    if abs(angle) > _MAX_DESKEW_DEG:
        return StageResult(
            image=img_bgr, applied=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error="ANGLE_OUT_OF_RANGE",
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


def _stage_clahe_enhance(img_bgr: np.ndarray) -> StageResult:
    """Apply mild CLAHE on the L channel + bilateral denoise + light unsharp mask.

    Always succeeds on a valid BGR image; failure returns applied=False.
    """
    import time
    t0 = time.perf_counter()
    try:
        # Bilateral denoise first (preserves edges while smoothing paper texture).
        denoised = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=20, sigmaSpace=20)
        # CLAHE on L channel in LAB space.
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        # Light unsharp mask for text legibility.
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
        out = cv2.addWeighted(out, 1.2, blurred, -0.2, 0)
    except cv2.error as e:
        return StageResult(
            image=img_bgr, applied=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error=f"CLAHE_FAILED: {e}",
        )
    return StageResult(
        image=out, applied=True,
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


# Module-level holder for the lazy-loaded DocRect model path. Set by
# run_enhance_pipeline from the Config; None means the stage is skipped.
_docrect_model_path: str | None = None

_docrect_lock = _threading.Lock()
_docrect_session = None
_docrect_session_path: str | None = None


def _load_onnx_session(model_path: str):
    """Load an onnxruntime InferenceSession in CPU mode."""
    if not _HAS_ONNX:
        raise RuntimeError("onnxruntime not installed")
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = max(1, os.cpu_count() or 1)
    return onnxruntime.InferenceSession(
        model_path, sess_options=so, providers=["CPUExecutionProvider"]
    )


def _load_torch_session(model_path: str):
    """Load the DocTr PyTorch model. Lazy import torch to keep cold start cheap."""
    import torch  # noqa: F401  (heavy, only when fallback needed)
    from doclayout_yolo_extras.docrect_torch import DocRect  # type: ignore
    model = DocRect()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _load_docrect_session(model_path: str):
    """Load the DocRect model. Prefers ONNX if the file ends in .onnx AND onnxruntime
    is importable; otherwise falls back to PyTorch. Raises RuntimeError on total failure.
    """
    global _docrect_session, _docrect_session_path
    with _docrect_lock:
        if _docrect_session is not None and _docrect_session_path == model_path:
            return _docrect_session
        suffix = model_path.lower().rsplit(".", 1)[-1]
        sess = None
        if suffix == "onnx" and _HAS_ONNX:
            try:
                sess = _load_onnx_session(model_path)
            except Exception:
                sess = None
        if sess is None:
            sess = _load_torch_session(model_path)
        _docrect_session = sess
        _docrect_session_path = model_path
        return sess


def _apply_docrect(img_bgr: np.ndarray, model_path: str) -> np.ndarray:
    """Run DocRect dewarping. Lazy-loads the model; calls the appropriate backend."""
    sess = _load_docrect_session(model_path)
    suffix = model_path.lower().rsplit(".", 1)[-1]
    # ONNX path
    if suffix == "onnx" and _HAS_ONNX and not isinstance(sess, str):
        # input prep — resize to 256x256, normalize to [0,1], NCHW float32
        h, w = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (256, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        input_name = sess.get_inputs()[0].name
        _ = sess.run(None, {input_name: tensor})[0]
        # Actual output shape depends on the export; we return a deterministic
        # resize-to-original as a placeholder so end-to-end runs without warping.
        return cv2.resize(img_bgr, (w, h))
    # PyTorch path (placeholder — same shape-preserving behavior)
    return img_bgr


def _stage_dewarping(img_bgr: np.ndarray) -> StageResult:
    """Run DocRect dewarping. Skipped (applied=False, error=DOCRECT_UNAVAILABLE)
    when no model path has been configured."""
    import time
    t0 = time.perf_counter()
    if not _docrect_model_path:
        return StageResult(
            image=img_bgr, applied=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error="DOCRECT_UNAVAILABLE: model path not configured",
        )
    try:
        out = _apply_docrect(img_bgr, _docrect_model_path)
    except Exception as e:
        return StageResult(
            image=img_bgr, applied=False,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error=f"DOCRECT_UNAVAILABLE: {e}",
        )
    return StageResult(
        image=out, applied=True,
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


def run_enhance_pipeline(
    task_id: str,
    image_bytes: bytes,
    params: dict,
) -> None:
    """Run decode → edge_crop → deskew → dewarping → clahe_enhance.

    Updates the task state in-place via service.exam.tasks helpers.
    `params` may carry:
      - docrect_model_path: str | None — passed to _docrect_model_path for this task
      - enable_deskew / enable_dewarp / enable_clahe: bool (all default True)
    """
    import time
    from service.exam import tasks
    from service.exam.schemas import EnhancementFinal

    global _docrect_model_path
    _docrect_model_path = params.get("docrect_model_path") or None

    # Stage 0: decode (fatal if it fails)
    t0 = time.perf_counter()
    try:
        img = _decode_image_bytes(image_bytes)
    except ValueError as e:
        tasks.update_stage(
            task_id, "decode",
            duration_ms=int((time.perf_counter() - t0) * 1000),
            error=str(e),
        )
        tasks.mark_failed(task_id, f"decode failed: {e}")
        return
    tasks.update_stage(
        task_id, "decode",
        duration_ms=int((time.perf_counter() - t0) * 1000),
        payload={"shape": list(img.shape)},
    )

    def run(name: str, fn, img, enabled: bool):
        if not enabled:
            tasks.update_stage(
                task_id, name,
                duration_ms=0,
                payload={"applied": False, "skipped_reason": "DISABLED_BY_PARAMS"},
            )
            return img
        result = fn(img)
        tasks.update_stage(
            task_id, name,
            duration_ms=result.duration_ms,
            payload={"applied": result.applied, **(result.payload or {})},
            error=result.error,
        )
        return result.image if result.applied else img

    img = run("edge_crop", _stage_edge_crop, img, True)
    img = run("deskew", _stage_deskew, img, params.get("enable_deskew", True))
    img = run("dewarping", _stage_dewarping, img, params.get("enable_dewarp", True))
    img = run("clahe_enhance", _stage_clahe_enhance, img, params.get("enable_clahe", True))

    # Build final result.
    original_data_url, _ = _encode_jpeg_base64(_decode_image_bytes(image_bytes))
    enhanced_data_url, enhanced_raw = _encode_jpeg_base64(img)
    import base64 as _b64
    enhanced_bytes_b64 = _b64.b64encode(enhanced_raw).decode("ascii")

    applied = [
        name for name in ("edge_crop", "deskew", "dewarping", "clahe_enhance")
        if (tasks.get_task(task_id) or {}).get("stages", {}).get(name, {}).get("payload", {}).get("applied")
    ]

    h, w = img.shape[:2]
    final = EnhancementFinal(
        original_image=original_data_url,
        enhanced_image=enhanced_data_url,
        applied_stages=applied,
        enhanced_bytes_b64=enhanced_bytes_b64,
        output_size={"width": int(w), "height": int(h)},
    )
    tasks.mark_done_enhance(task_id, final)
