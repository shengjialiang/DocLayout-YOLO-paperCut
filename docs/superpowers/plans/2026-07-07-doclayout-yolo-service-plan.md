# DocLayout-YOLO HTTP Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 DocLayout-YOLO 部署为本地 FastAPI HTTP 服务,接受图片上传,返回标注图片和 JSON 布局数据,并提供简单的浏览器上传 UI。

**Architecture:** 单进程 FastAPI + uvicorn 服务,启动期加载 YOLOv10 模型到全局单例,用 `asyncio.Semaphore` 控制并发推理。模型层、配置层、路由层拆分到独立模块,便于测试。

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, python-multipart, Pillow, opencv-python (依赖已存在), pytest + httpx (测试)。

---

## 文件结构

```
DocLayout-YOLO/
├── service/
│   ├── __init__.py        # 空,标记为 Python 包
│   ├── config.py          # 环境变量加载 (约 30 行)
│   ├── inference.py       # 模型加载 + 推理执行 (约 80 行)
│   ├── schemas.py         # Pydantic 响应模型 (约 50 行)
│   ├── app.py             # FastAPI 应用 + 路由 + 入口 (约 200 行)
│   └── static/
│       └── index.html     # 上传页面 (约 200 行)
├── tests/
│   └── test_service.py    # 单元测试,模型被 mock (约 250 行)
├── pyproject.toml         # 修改: 新增 [service] extras
└── docs/superpowers/specs/2026-07-07-doclayout-yolo-service-design.md  # 已存在
```

**职责切分:**
- `config.py`: 读环境变量,提供强类型 Config 数据类
- `inference.py`: 加载模型 (`load_model`)、执行推理 (`predict_one`)、解析结果 (`parse_detections`)、图像编解码辅助
- `schemas.py`: Pydantic 模型描述 `/predict` 响应,`/health` 响应
- `app.py`: FastAPI 应用工厂,路由,中间件,全局异常处理,启动期 lifespan
- `static/index.html`: 纯 HTML + JS 上传界面

---

## Task 1: 在 pyproject.toml 添加 [service] extras

**Files:**
- Modify: `pyproject.toml:85-98` (在 `[project.optional-dependencies]` 中)

- [ ] **Step 1: 在 pyproject.toml 添加 service 可选依赖**

编辑 `pyproject.toml`,在 `[project.optional-dependencies]` 列表里 `dev` 之前(或合理位置)插入 `service`:

```toml
service = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.23",
    "python-multipart>=0.0.6",
    "httpx>=0.24",
]
```

完整添加到合适位置例如:

```toml
# Optional dependencies ------------------------------------------------------------------------------------------------
[project.optional-dependencies]
service = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.23",
    "python-multipart>=0.0.6",
    "httpx>=0.24",
]
dev = [
    ...
]
```

- [ ] **Step 2: 验证 pyproject.toml 语法合法**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`
Expected: 不报错退出码 0。

- [ ] **Step 3: 提交**

```bash
git add pyproject.toml
git commit -m "build: add [service] extras for fastapi service deps"
```

---

## Task 2: 实现 service/config.py

**Files:**
- Create: `service/__init__.py`
- Create: `service/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 创建 service 包**

创建 `service/__init__.py`(空文件):

```python
"""DocLayout-YOLO HTTP service."""
```

- [ ] **Step 2: 写失败测试**

创建 `tests/test_config.py`:

```python
import pytest
from service.config import Config, load_config


def test_load_config_with_env(monkeypatch, tmp_path):
    """环境变量优先于默认。"""
    model_file = tmp_path / "m.pt"
    model_file.touch()
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("MAX_CONCURRENT", "3")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "10")

    cfg = load_config()

    assert cfg.model_path == str(model_file)
    assert cfg.device == "cpu"
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9000
    assert cfg.max_concurrent == 3
    assert cfg.max_file_size_mb == 10


def test_load_config_defaults(monkeypatch, tmp_path):
    """未设置环境变量时使用默认。"""
    model_file = tmp_path / "m.pt"
    model_file.touch()
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    for k in ("DEVICE", "HOST", "PORT", "MAX_CONCURRENT", "MAX_FILE_SIZE_MB"):
        monkeypatch.delenv(k, raising=False)

    cfg = load_config()

    assert cfg.device is None  # 表示自动检测
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.max_concurrent == 1
    assert cfg.max_file_size_mb == 20


def test_load_config_missing_model_path(monkeypatch, tmp_path):
    """MODEL_PATH 未设置应抛错。"""
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with pytest.raises(ValueError, match="MODEL_PATH"):
        load_config()


def test_load_config_nonexistent_model(tmp_path):
    """模型文件不存在应抛错。"""
    import os
    os.environ["MODEL_PATH"] = str(tmp_path / "nonexistent.pt")
    try:
        with pytest.raises(FileNotFoundError):
            load_config()
    finally:
        del os.environ["MODEL_PATH"]
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'service.config'`

- [ ] **Step 4: 实现 service/config.py**

```python
"""Service configuration loaded from environment variables."""
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model_path: str
    device: str | None  # None means auto-detect
    host: str
    port: int
    max_concurrent: int
    max_file_size_mb: int


def load_config() -> Config:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise ValueError(
            "MODEL_PATH environment variable is required. "
            "Set it to your .pt model file, e.g. "
            "MODEL_PATH=/path/to/doclayout_yolo_docstructbench_imgsz1024.pt"
        )
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = os.environ.get("DEVICE") or None  # empty string -> None

    return Config(
        model_path=model_path,
        device=device,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        max_concurrent=int(os.environ.get("MAX_CONCURRENT", "1")),
        max_file_size_mb=int(os.environ.get("MAX_FILE_SIZE_MB", "20")),
    )
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 6: 提交**

```bash
git add service/__init__.py service/config.py tests/test_config.py
git commit -m "feat(service): add Config dataclass with env-var loading"
```

---

## Task 3: 实现 service/inference.py —— parse_detections

**Files:**
- Modify: `service/inference.py` (create)
- Test: `tests/test_inference.py`

这一步先只做 `parse_detections` 工具函数,模型加载与 `predict_one` 留到下一步。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_inference.py`:

```python
import numpy as np
import pytest

from service.inference import parse_detections


class FakeBoxes:
    """Stand-in for doclayout_yolo.engine.results.Boxes."""

    def __init__(self, xyxy, conf, cls, names):
        self.xyxy = xyxy      # numpy array of shape (N, 4)
        self.conf = conf      # numpy array of shape (N,)
        self.cls = cls        # numpy array of shape (N,) int
        self._names = names

    @property
    def names(self):
        # API: results.boxes.names
        return self._names


class FakeResult:
    def __init__(self, orig_shape, names, boxes):
        self.orig_shape = orig_shape  # (h, w)
        self.names = names
        self.boxes = boxes


def test_parse_detections_empty():
    """Empty boxes returns empty list."""
    boxes = FakeBoxes(
        xyxy=np.zeros((0, 4)),
        conf=np.zeros((0,)),
        cls=np.zeros((0,), dtype=int),
        names={0: "title"},
    )
    res = FakeResult(orig_shape=(100, 200), names={0: "title"}, boxes=boxes)

    detections = parse_detections(res)

    assert detections == []


def test_parse_detections_returns_expected_fields():
    """Each detection has id, class_id, class_name, bbox_xyxy, bbox_xywh, score."""
    boxes = FakeBoxes(
        xyxy=np.array([[10.0, 20.0, 110.0, 70.0], [200.0, 50.0, 300.0, 150.0]]),
        conf=np.array([0.95, 0.80]),
        cls=np.array([0, 1], dtype=int),
        names={0: "title", 1: "text"},
    )
    res = FakeResult(orig_shape=(200, 400), names={0: "title", 1: "text"}, boxes=boxes)

    detections = parse_detections(res)

    assert len(detections) == 2
    assert detections[0]["id"] == 0
    assert detections[0]["class_id"] == 0
    assert detections[0]["class_name"] == "title"
    assert detections[0]["bbox_xyxy"] == [10.0, 20.0, 110.0, 70.0]
    assert detections[0]["bbox_xywh"] == [60.0, 45.0, 100.0, 50.0]  # xc, yc, w, h
    assert detections[0]["score"] == pytest.approx(0.95)
    assert detections[1]["class_name"] == "text"
    assert detections[1]["bbox_xyxy"] == [200.0, 50.0, 300.0, 150.0]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_inference.py -v`
Expected: `ModuleNotFoundError: No module named 'service.inference'`

- [ ] **Step 3: 实现 service/inference.py (仅 parse_detections 部分)**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_inference.py -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add service/inference.py tests/test_inference.py
git commit -m "feat(service): add parse_detections to extract boxes from result"
```

---

## Task 4: 实现 service/inference.py —— load_model 和 predict_one

**Files:**
- Modify: `service/inference.py`
- Test: `tests/test_inference.py` (append)

- [ ] **Step 1: 写失败测试 (load_model)**

追加到 `tests/test_inference.py`:

```python
from unittest import mock

from service.inference import load_model


def test_load_model_calls_yolov10(monkeypatch, tmp_path):
    """load_model instantiates YOLOv10 with the given path and returns it."""
    model_file = tmp_path / "model.pt"
    model_file.touch()

    fake_instance = object()
    fake_yolov10_cls = mock.MagicMock(return_value=fake_instance)

    with mock.patch("service.inference.YOLOv10", fake_yolov10_cls):
        result = load_model(str(model_file))

    assert result is fake_instance
    fake_yolov10_cls.assert_called_once_with(str(model_file))
```

> 注意:此测试依赖 `service/inference.py` 顶层有 `from doclayout_yolo import YOLOv10`(在 Task 4 Step 4 实现时加上)。

- [ ] **Step 2: 写失败测试 (predict_one)**

追加到 `tests/test_inference.py`(在文件顶部 import 块下,确保 `import io` 和 `from PIL import Image` 已存在):

```python
import asyncio
import io
from unittest import mock

import numpy as np
from PIL import Image

from service.inference import predict_one


class _FakeResults:
    """Mimics doclayout_yolo Results object."""

    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes
        self.orig_shape = (100, 200)

    def plot(self, pil=True, line_width=5, font_size=20):
        # Return a small RGB numpy array
        return np.zeros((100, 200, 3), dtype=np.uint8)


def _fake_image_bytes():
    """Build a 100x200 white PNG in memory."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    pil = Image.fromarray(img[:, :, ::-1])  # BGR->RGB
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def test_predict_one_runs_model_and_returns_dict(monkeypatch):
    """predict_one calls model.predict, parses result, encodes image, returns metadata."""
    image_bytes = _fake_image_bytes()

    fake_box = mock.MagicMock()
    fake_box.xyxy = np.array([[10.0, 20.0, 110.0, 70.0]])
    fake_box.conf = np.array([0.95])
    fake_box.cls = np.array([0])
    fake_results = [_FakeResults(names={0: "title"}, boxes=fake_box)]

    fake_model = mock.MagicMock()
    fake_model.predict = mock.MagicMock(return_value=fake_results)
    fake_semaphore = mock.MagicMock()
    fake_semaphore.__aenter__ = mock.AsyncMock(return_value=None)
    fake_semaphore.__aexit__ = mock.AsyncMock(return_value=None)

    out = asyncio.run(predict_one(
        image_bytes=image_bytes,
        model=fake_model,
        semaphore=fake_semaphore,
        conf=0.2,
        imgsz=1024,
        line_width=5,
        font_size=20,
        device="cpu",
    ))

    # model.predict was called with decoded BGR numpy array
    fake_model.predict.assert_called_once()
    call_args = fake_model.predict.call_args
    # First positional arg should be a numpy array (decoded image)
    img_arg = call_args.args[0]
    assert isinstance(img_arg, np.ndarray)
    assert call_args.kwargs["conf"] == 0.2
    assert call_args.kwargs["imgsz"] == 1024
    assert call_args.kwargs["device"] == "cpu"

    assert out["width"] == 200
    assert out["height"] == 100
    assert out["num_detections"] == 1
    assert out["conf_threshold"] == 0.2
    assert out["model_imgsz"] == 1024
    assert out["device"] == "cpu"
    assert out["detections"][0]["class_name"] == "title"
    assert "annotated_base64" in out
    assert out["annotated_base64"].startswith("data:image/jpeg;base64,")
    assert out["inference_time_ms"] >= 0


def test_predict_one_empty_detections(monkeypatch):
    """No detections returns empty list and num_detections=0."""
    image_bytes = _fake_image_bytes()

    fake_box = mock.MagicMock()
    fake_box.xyxy = np.zeros((0, 4))
    fake_box.conf = np.zeros((0,))
    fake_box.cls = np.zeros((0,), dtype=int)
    fake_results = [_FakeResults(names={0: "title"}, boxes=fake_box)]

    fake_model = mock.MagicMock()
    fake_model.predict = mock.MagicMock(return_value=fake_results)
    fake_semaphore = mock.MagicMock()
    fake_semaphore.__aenter__ = mock.AsyncMock(return_value=None)
    fake_semaphore.__aexit__ = mock.AsyncMock(return_value=None)

    out = asyncio.run(predict_one(
        image_bytes=image_bytes,
        model=fake_model,
        semaphore=fake_semaphore,
        conf=0.2,
        imgsz=1024,
        line_width=5,
        font_size=20,
        device="cpu",
    ))
    assert out["num_detections"] == 0
    assert out["detections"] == []
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/test_inference.py -v`
Expected: 测试 `test_load_model_calls_yolov10` 与 `test_predict_one_*` 失败(`load_model`/`predict_one` 未定义)

- [ ] **Step 4: 实现 load_model 和 predict_one**

修改 `service/inference.py`,在 `parse_detections` 之后追加:

```python
def load_model(model_path: str):
    """Instantiate a YOLOv10 model from path."""
    # Local import to keep the module importable in tests without doclayout_yolo
    from doclayout_yolo import YOLOv10  # noqa: WPS433
    return YOLOv10(model_path)


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


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

    async with asyncio.wait_for(semaphore.acquire(), timeout=timeout):
        t0 = time.perf_counter()
        try:
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
            # plot(pil=True) returns a numpy RGB array when plot returns RGB
            # (depending on doclayout_yolo version, may return PIL Image).
            # Normalize to numpy RGB uint8:
            if isinstance(annotated_rgb, Image.Image):
                annotated_rgb = np.asarray(annotated_rgb)
            inference_ms = int((time.perf_counter() - t0) * 1000)
        finally:
            semaphore.release()

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
```

并在文件顶部 import 区补充:

```python
import asyncio
import time
```

完整 `service/inference.py` 顶部应包含:

```python
"""Model loading and inference execution."""
from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_inference.py -v`
Expected: 6 passed (2 parse + 1 load + 2 predict + 1 empty)

如果 `test_load_model_calls_yolov10` 失败因为 mock 路径复杂,可改为直接 patch `service.inference.YOLOv10`:

修改测试里的 `mock.patch.dict("sys.modules", {"doclayout_yolo": ...})` 改为:

```python
mock_yolov10_mod = mock.MagicMock()
mock_yolov10_mod.YOLOv10 = mock_yolov10_cls
with mock.patch.dict("sys.modules", {"doclayout_yolo": mock_yolov10_mod}):
    result = load_model(str(model_file))
```

或更直接,先在 `service/inference.py` 改为模块级 `YOLOv10` 引用:

```python
from doclayout_yolo import YOLOv10  # noqa: F401
```

然后测试用 `mock.patch("service.inference.YOLOv10", fake_yolov10_cls)`。二选一即可。

- [ ] **Step 6: 提交**

```bash
git add service/inference.py tests/test_inference.py
git commit -m "feat(service): add load_model and predict_one to inference module"
```

---

## Task 5: 实现 service/schemas.py

**Files:**
- Create: `service/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_schemas.py`:

```python
from service.schemas import Detection, HealthResponse, PredictResponse


def test_detection_field_names():
    """Detection model has the documented field names."""
    d = Detection(
        id=0,
        class_id=1,
        class_name="text",
        bbox_xyxy=[10.0, 20.0, 110.0, 70.0],
        bbox_xywh=[60.0, 45.0, 100.0, 50.0],
        score=0.95,
    )
    assert d.id == 0
    assert d.class_name == "text"
    assert d.bbox_xyxy == [10.0, 20.0, 110.0, 70.0]
    assert d.score == 0.95


def test_predict_response_serialization():
    """PredictResponse dumps to camelCase-free JSON (snake_case) by default."""
    resp = PredictResponse(
        width=1240,
        height=1754,
        annotated_image="data:image/jpeg;base64,xxx",
        detections=[],
        num_detections=0,
        inference_time_ms=234,
        model_imgsz=1024,
        conf_threshold=0.2,
        device="cpu",
    )
    j = resp.model_dump()
    assert j["width"] == 1240
    assert j["annotated_image"] == "data:image/jpeg;base64,xxx"
    assert j["num_detections"] == 0
    assert j["inference_time_ms"] == 234
    assert j["device"] == "cpu"


def test_health_response_serialization():
    h = HealthResponse(
        status="ok",
        device="cpu",
        model_loaded=True,
        max_concurrent=1,
        model_path="/tmp/model.pt",
    )
    j = h.model_dump()
    assert j["status"] == "ok"
    assert j["model_loaded"] is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_schemas.py -v`
Expected: `ModuleNotFoundError: No module named 'service.schemas'`

- [ ] **Step 3: 实现 service/schemas.py**

```python
"""Pydantic response schemas for the HTTP service."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    id: int
    class_id: int
    class_name: str
    bbox_xyxy: list[float] = Field(..., description="[x1, y1, x2, y2] in pixel coords")
    bbox_xywh: list[float] = Field(..., description="[cx, cy, w, h]")
    score: float


class PredictResponse(BaseModel):
    width: int
    height: int
    annotated_image: str = Field(..., description="Data-URL: data:image/jpeg;base64,...")
    detections: list[Detection]
    num_detections: int
    inference_time_ms: int
    model_imgsz: int
    conf_threshold: float
    device: str


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    max_concurrent: int
    model_path: str
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_schemas.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add service/schemas.py tests/test_schemas.py
git commit -m "feat(service): add Pydantic response schemas"
```

---

## Task 6: 实现 service/app.py —— 应用骨架 + /health

**Files:**
- Create: `service/app.py`
- Test: `tests/test_app.py`

- [ ] **Step 1: 写失败测试 (健康检查 + 启动期)**

创建 `tests/test_app.py`:

```python
import pytest
from unittest import mock
from fastapi.testclient import TestClient

from service.app import create_app
from service.config import Config


def _fake_config(tmp_path=None):
    m = tmp_path or "/tmp/fake.pt"
    return Config(
        model_path=m,
        device="cpu",
        host="127.0.0.1",
        port=8000,
        max_concurrent=1,
        max_file_size_mb=20,
    )


def test_health_ok_after_model_loaded(tmp_path):
    """/health returns ok once model is loaded."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    fake_model = mock.MagicMock()

    with mock.patch("service.app.load_model", return_value=fake_model), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        client = TestClient(app)
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert body["model_loaded"] is True
    assert body["max_concurrent"] == 1


def test_health_starting_when_no_model(tmp_path):
    """If lifespan didn't run (no model), health returns 503 + status: starting."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    with mock.patch("service.app.load_config", return_value=cfg), \
         mock.patch("service.app.load_model", return_value=mock.MagicMock()):
        app = create_app()
        # Don't enter the lifespan context: model is None.
        client = TestClient(app)  # No `with` — no lifespan
        response = client.get("/health")

    # Without lifespan, our model is None; we expect 503.
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "starting"
    assert body["model_loaded"] is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_app.py -v`
Expected: 测试失败 (`create_app` 不存在或 import 出错)

- [ ] **Step 3: 实现 service/app.py (骨架)**

```python
"""FastAPI application: routes, middleware, entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from service.config import Config, load_config
from service.inference import load_model
from service.schemas import HealthResponse

logger = logging.getLogger("doclayout_service")


def create_app() -> FastAPI:
    """Build the FastAPI app. Loads model on lifespan startup."""
    cfg = load_config()
    state: dict[str, Any] = {"model": None, "config": cfg}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("loading model from %s ...", cfg.model_path)
        state["model"] = load_model(cfg.model_path)
        device_label = cfg.device or "auto"
        logger.info(
            "model loaded, device=%s, max_concurrent=%d",
            device_label,
            cfg.max_concurrent,
        )
        try:
            yield
        finally:
            state["model"] = None

    app = FastAPI(title="DocLayout-YOLO Service", lifespan=lifespan)

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"description": "Model still loading"}},
    )
    def health():
        model = state["model"]
        device_label = cfg.device or "auto"
        if model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "starting",
                    "device": device_label,
                    "model_loaded": False,
                    "max_concurrent": cfg.max_concurrent,
                    "model_path": cfg.model_path,
                },
            )
        return HealthResponse(
            status="ok",
            device=device_label,
            model_loaded=True,
            max_concurrent=cfg.max_concurrent,
            model_path=cfg.model_path,
        )

    app.state.service_state = state
    return app


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    app = create_app()
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_app.py -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add service/app.py tests/test_app.py
git commit -m "feat(service): add FastAPI app skeleton with /health endpoint"
```

---

## Task 7: 添加 /predict/image 路由

**Files:**
- Modify: `service/app.py` (追加路由 + 用到的新 import)
- Test: `tests/test_app.py` (追加测试)

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_app.py`:

```python
import io
import numpy as np
from PIL import Image
from unittest import mock

from fastapi import UploadFile


def _png_bytes(width=200, height=100):
    img = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predict_image_returns_jpeg_bytes(tmp_path):
    """/predict/image returns image/jpeg body."""
    cfg = _fake_config(str(tmp_path / "m.pt"))

    # Mock model predicts one box
    fake_box = mock.MagicMock()
    fake_box.xyxy = np.array([[10.0, 20.0, 110.0, 70.0]])
    fake_box.conf = np.array([0.95])
    fake_box.cls = np.array([0])

    class _PlotResult:
        names = {0: "title"}
        boxes = fake_box
        orig_shape = (100, 200)

        def plot(self, pil=True, line_width=5, font_size=20):
            return np.zeros((100, 200, 3), dtype=np.uint8)

    fake_model = mock.MagicMock()
    fake_model.predict = mock.MagicMock(return_value=[_PlotResult()])

    with mock.patch("service.app.load_model", return_value=fake_model), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        # Force state to skip lifespan (avoids actual load attempt)
        app.state.service_state["model"] = fake_model
        client = TestClient(app)
        response = client.post(
            "/predict/image",
            files={"file": ("test.png", _png_bytes(), "image/png")},
            data={"conf": "0.2", "imgsz": "1024"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert len(response.content) > 0


def test_predict_image_rejects_non_image_mime(tmp_path):
    """Uploading a text file returns 415."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    with mock.patch("service.app.load_model", return_value=mock.MagicMock()), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = mock.MagicMock()
        client = TestClient(app)
        response = client.post(
            "/predict/image",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
    assert response.status_code == 415


def test_predict_image_rejects_corrupt_image(tmp_path):
    """Corrupt image bytes return 400."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    with mock.patch("service.app.load_model", return_value=mock.MagicMock()), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = mock.MagicMock()
        client = TestClient(app)
        response = client.post(
            "/predict/image",
            files={"file": ("bad.png", b"not-an-image", "image/png")},
        )
    # Either 400 from decode failure, or 415/422 if MIME is invalid.
    assert response.status_code in (400, 415, 422)


def test_predict_image_rejects_imgsz_out_of_range(tmp_path):
    """imgsz outside [320, 2048] returns 422."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    with mock.patch("service.app.load_model", return_value=mock.MagicMock()), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = mock.MagicMock()
        client = TestClient(app)
        response = client.post(
            "/predict/image",
            files={"file": ("test.png", _png_bytes(), "image/png")},
            data={"imgsz": "50"},  # below 320
        )
    assert response.status_code == 422


def test_predict_image_requires_file(tmp_path):
    """Missing file field returns 422."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    with mock.patch("service.app.load_model", return_value=mock.MagicMock()), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = mock.MagicMock()
        client = TestClient(app)
        response = client.post("/predict/image")
    assert response.status_code == 422
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_app.py -v -k predict_image`
Expected: 5 failed (路由未注册)

- [ ] **Step 3: 实现 /predict/image 路由**

修改 `service/app.py`。在文件顶部 import 改为:

```python
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
```

并在 `create_app` 内、`/health` 路由后、`return app` 之前插入:

```python
    @app.post("/predict/image", responses={
        400: {"description": "Bad image"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Validation error"},
        503: {"description": "Service not ready"},
    })
    async def predict_image(
        file: UploadFile = File(...),
        conf: float = Form(0.2),
        imgsz: int = Form(1024),
        line_width: int = Form(5),
        font_size: int = Form(20),
    ):
        """Return the annotated image directly (image/jpeg)."""
        # Validate params
        if not (0.0 <= conf <= 1.0):
            return JSONResponse(status_code=422, content={"detail": "conf must be in [0, 1]", "error_code": "INVALID_PARAM"})
        if not (320 <= imgsz <= 2048):
            return JSONResponse(status_code=422, content={"detail": "imgsz must be in [320, 2048]", "error_code": "INVALID_PARAM"})
        if not (1 <= line_width <= 20):
            return JSONResponse(status_code=422, content={"detail": "line_width must be in [1, 20]", "error_code": "INVALID_PARAM"})
        if not (8 <= font_size <= 50):
            return JSONResponse(status_code=422, content={"detail": "font_size must be in [8, 50]", "error_code": "INVALID_PARAM"})

        # MIME check
        if not (file.content_type or "").startswith("image/"):
            return JSONResponse(status_code=415, content={"detail": "unsupported media type", "error_code": "BAD_MIME"})

        # Size check (Content-Length first, then actual)
        if file.size is not None and file.size > cfg.max_file_size_mb * 1024 * 1024:
            return JSONResponse(status_code=413, content={"detail": f"file too large, max {cfg.max_file_size_mb}MB", "error_code": "FILE_TOO_LARGE"})

        raw = await file.read()
        if len(raw) > cfg.max_file_size_mb * 1024 * 1024:
            return JSONResponse(status_code=413, content={"detail": f"file too large, max {cfg.max_file_size_mb}MB", "error_code": "FILE_TOO_LARGE"})

        model = state["model"]
        if model is None:
            return JSONResponse(status_code=503, content={"detail": "model not ready", "error_code": "NOT_READY"})

        # Run inference (reuse logic from predict route)
        from service.inference import predict_one
        sem = asyncio.Semaphore(cfg.max_concurrent)
        try:
            result = await predict_one(
                image_bytes=raw,
                model=model,
                semaphore=sem,
                conf=conf,
                imgsz=imgsz,
                line_width=line_width,
                font_size=font_size,
                device=cfg.device or "cpu",
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc), "error_code": "DECODE_ERROR"})
        except Exception as exc:
            logger.exception("inference failed")
            return JSONResponse(status_code=500, content={"detail": f"inference failed: {exc}", "error_code": "INTERNAL"})

        # Decode data URL -> bytes
        b64 = result["annotated_base64"].split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        return Response(content=img_bytes, media_type="image/jpeg")
```

并在 `service/app.py` 顶部 import 添加:

```python
import asyncio
import base64
```

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_app.py -v -k predict_image`
Expected: 5 passed

如果 `test_predict_image_rejects_imgsz_out_of_range` 因为 `Form` 校验返 422,则 PASS;如果返回 200(因为 Form 自动类型转换),改测试为断言 `status_code == 422`,因为我们的应用层校验应当先于 FastAPI 自动校验或补充后跟它一致。我们已经在 endpoint 内部做了显式校验,应当先触发。

- [ ] **Step 5: 提交**

```bash
git add service/app.py tests/test_app.py
git commit -m "feat(service): add POST /predict/image endpoint"
```

---

## Task 8: 添加 /predict 路由 (返回 JSON)

**Files:**
- Modify: `service/app.py`
- Test: `tests/test_app.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_app.py`:

```python
from service.schemas import PredictResponse


def test_predict_returns_json_with_annotated_image(tmp_path):
    """/predict returns PredictResponse with annotated_image base64 + detections."""
    cfg = _fake_config(str(tmp_path / "m.pt"))

    fake_box = mock.MagicMock()
    fake_box.xyxy = np.array([[10.0, 20.0, 110.0, 70.0]])
    fake_box.conf = np.array([0.95])
    fake_box.cls = np.array([0])

    class _PlotResult:
        names = {0: "title"}
        boxes = fake_box
        orig_shape = (100, 200)

        def plot(self, pil=True, line_width=5, font_size=20):
            return np.zeros((100, 200, 3), dtype=np.uint8)

    fake_model = mock.MagicMock()
    fake_model.predict = mock.MagicMock(return_value=[_PlotResult()])

    with mock.patch("service.app.load_model", return_value=fake_model), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = fake_model
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("test.png", _png_bytes(), "image/png")},
            data={"conf": "0.2"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["width"] == 200
    assert body["height"] == 100
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")
    assert body["num_detections"] == 1
    assert body["detections"][0]["class_name"] == "title"
    assert body["detections"][0]["score"] == 0.95
    assert body["conf_threshold"] == 0.2
    assert body["device"] == "cpu"


def test_predict_serves_concurrent_requests_serially(tmp_path):
    """Multiple concurrent requests all succeed; semaphore doesn't deadlock."""
    cfg = _fake_config(str(tmp_path / "m.pt"))

    fake_box = mock.MagicMock()
    fake_box.xyxy = np.array([[0.0, 0.0, 50.0, 50.0]])
    fake_box.conf = np.array([0.9])
    fake_box.cls = np.array([0])

    class _PlotResult:
        names = {0: "x"}
        boxes = fake_box
        orig_shape = (50, 50)

        def plot(self, pil=True, line_width=5, font_size=20):
            return np.zeros((50, 50, 3), dtype=np.uint8)

    fake_model = mock.MagicMock()
    fake_model.predict = mock.MagicMock(return_value=[_PlotResult()])

    with mock.patch("service.app.load_model", return_value=fake_model), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        app.state.service_state["model"] = fake_model
        client = TestClient(app)
        responses = []
        for _ in range(5):
            responses.append(client.post(
                "/predict",
                files={"file": ("t.png", _png_bytes(50, 50), "image/png")},
            ))
    assert all(r.status_code == 200 for r in responses)
    assert fake_model.predict.call_count == 5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_app.py -v -k predict_returns_json or k predict_serves`
Expected: `test_predict_returns_json_with_annotated_image` 失败 (`/predict` 路由不存在)

- [ ] **Step 3: 实现 /predict 路由 (JSON)**

在 `service/app.py` 的 `create_app` 内、`return app` 之前追加:

```python
    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            400: {"description": "Bad image"},
            413: {"description": "File too large"},
            415: {"description": "Unsupported media type"},
            422: {"description": "Validation error"},
            503: {"description": "Service not ready"},
        },
    )
    async def predict(
        file: UploadFile = File(...),
        conf: float = Form(0.2),
        imgsz: int = Form(1024),
        line_width: int = Form(5),
        font_size: int = Form(20),
    ):
        """Run inference and return JSON with annotated image (base64) + detections."""
        # Validation block (same as /predict/image)
        if not (0.0 <= conf <= 1.0):
            return JSONResponse(status_code=422, content={"detail": "conf must be in [0, 1]", "error_code": "INVALID_PARAM"})
        if not (320 <= imgsz <= 2048):
            return JSONResponse(status_code=422, content={"detail": "imgsz must be in [320, 2048]", "error_code": "INVALID_PARAM"})
        if not (1 <= line_width <= 20):
            return JSONResponse(status_code=422, content={"detail": "line_width must be in [1, 20]", "error_code": "INVALID_PARAM"})
        if not (8 <= font_size <= 50):
            return JSONResponse(status_code=422, content={"detail": "font_size must be in [8, 50]", "error_code": "INVALID_PARAM"})

        if not (file.content_type or "").startswith("image/"):
            return JSONResponse(status_code=415, content={"detail": "unsupported media type", "error_code": "BAD_MIME"})

        raw = await file.read()
        if len(raw) > cfg.max_file_size_mb * 1024 * 1024:
            return JSONResponse(status_code=413, content={"detail": f"file too large, max {cfg.max_file_size_mb}MB", "error_code": "FILE_TOO_LARGE"})

        model = state["model"]
        if model is None:
            return JSONResponse(status_code=503, content={"detail": "model not ready", "error_code": "NOT_READY"})

        from service.inference import predict_one
        sem = state["semaphore"]
        try:
            result = await predict_one(
                image_bytes=raw,
                model=model,
                semaphore=sem,
                conf=conf,
                imgsz=imgsz,
                line_width=line_width,
                font_size=font_size,
                device=cfg.device or "cpu",
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc), "error_code": "DECODE_ERROR"})
        except Exception as exc:
            logger.exception("inference failed")
            return JSONResponse(status_code=500, content={"detail": f"inference failed: {exc}", "error_code": "INTERNAL"})

        # Convert internal result -> response model
        return PredictResponse(
            width=result["width"],
            height=result["height"],
            annotated_image=result["annotated_base64"],
            detections=result["detections"],
            num_detections=result["num_detections"],
            inference_time_ms=result["inference_time_ms"],
            model_imgsz=result["model_imgsz"],
            conf_threshold=result["conf_threshold"],
            device=result["device"],
        )
```

并在 `create_app` 里 `state` 字典初始化处改为:

```python
    state: dict[str, Any] = {
        "model": None,
        "config": cfg,
        "semaphore": asyncio.Semaphore(cfg.max_concurrent),
    }
```

并去掉 `predict_image` 中的 `asyncio.Semaphore(...)` 局部构造,改为 `state["semaphore"]`。修改 `predict_image` 路由内:

```python
        from service.inference import predict_one
        sem = state["semaphore"]  # use module-level semaphore
        try:
            result = await predict_one(...)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_app.py -v`
Expected: 全部 passed

- [ ] **Step 5: 提交**

```bash
git add service/app.py tests/test_app.py
git commit -m "feat(service): add POST /predict JSON endpoint with shared semaphore"
```

---

## Task 9: 添加 GET / 路由 (静态首页)

**Files:**
- Modify: `service/app.py`
- Create: `service/static/index.html`
- Test: `tests/test_app.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_app.py`:

```python
def test_get_root_serves_html(tmp_path):
    """GET / returns index.html."""
    cfg = _fake_config(str(tmp_path / "m.pt"))
    with mock.patch("service.app.load_model", return_value=mock.MagicMock()), \
         mock.patch("service.app.load_config", return_value=cfg):
        app = create_app()
        client = TestClient(app)
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>" in response.text
    assert "no-store" in response.headers.get("cache-control", "")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_app.py -v -k get_root`
Expected: `test_get_root_serves_html` 失败 (`/` 路由不存在或返回 404)

- [ ] **Step 3: 创建 service/static/index.html**

创建文件 `service/static/` 目录,然后创建 `service/static/index.html`:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DocLayout-YOLO 检测服务</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif; max-width: 900px; margin: 20px auto; padding: 0 16px; color: #222; }
    h1 { font-size: 22px; }
    .drop-zone { border: 2px dashed #999; padding: 30px; text-align: center; border-radius: 8px; cursor: pointer; background: #fafafa; }
    .drop-zone.dragging { background: #e8f0fe; border-color: #4285f4; }
    .params { display: grid; grid-template-columns: 120px 1fr 60px; gap: 8px 12px; margin: 20px 0; align-items: center; }
    .params label { text-align: right; }
    button { background: #4285f4; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
    button:disabled { background: #ccc; cursor: not-allowed; }
    .status { margin: 12px 0; padding: 10px; border-radius: 4px; display: none; }
    .status.show { display: block; }
    .status.loading { background: #fff3cd; color: #856404; }
    .status.error { background: #f8d7da; color: #721c24; }
    .status.success { background: #d4edda; color: #155724; }
    .result { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }
    .result img { max-width: 100%; border: 1px solid #ddd; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; }
    th, td { padding: 6px 8px; border-bottom: 1px solid #eee; text-align: left; font-size: 13px; }
    th { background: #f5f5f5; }
  </style>
</head>
<body>
  <h1>DocLayout-YOLO 检测服务</h1>

  <div id="dropZone" class="drop-zone">拖拽图片到这里,或点击选择</div>
  <input type="file" id="fileInput" accept="image/*" style="display:none">

  <div class="params">
    <label for="confRange">conf 阈值</label>
    <input type="range" id="confRange" min="0" max="1" step="0.05" value="0.2">
    <output id="confOut">0.2</output>

    <label for="imgszRange">imgsz</label>
    <input type="range" id="imgszRange" min="320" max="2048" step="32" value="1024">
    <output id="imgszOut">1024</output>

    <label for="lineRange">框线粗细</label>
    <input type="range" id="lineRange" min="1" max="20" step="1" value="5">
    <output id="lineOut">5</output>

    <label for="fontRange">标签字号</label>
    <input type="range" id="fontRange" min="8" max="50" step="1" value="20">
    <output id="fontOut">20</output>
  </div>

  <button id="submitBtn" disabled>开始检测</button>

  <div id="status" class="status"></div>

  <h2 id="summary" style="display:none">检测结果</h2>
  <div class="result" id="result" style="display:none">
    <div>
      <p>标注图:</p>
      <img id="annotatedImg">
      <p><a id="downloadLink" download="annotated.jpg">下载标注图</a></p>
    </div>
    <div>
      <p>原图:</p>
      <img id="originalImg">
    </div>
  </div>

  <table id="detectionTable" style="display:none">
    <thead><tr><th>#</th><th>类别</th><th>置信度</th><th>坐标 (x1, y1, x2, y2)</th></tr></thead>
    <tbody></tbody>
  </table>

  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const submitBtn = document.getElementById('submitBtn');
    const statusEl = document.getElementById('status');
    const summary = document.getElementById('summary');
    const result = document.getElementById('result');
    const detectionTable = document.getElementById('detectionTable');

    let selectedFile = null;

    function showStatus(text, kind) {
      statusEl.textContent = text;
      statusEl.className = 'status show ' + kind;
    }

    function hideStatus() {
      statusEl.className = 'status';
    }

    // Update slider outputs
    ['confRange', 'imgszRange', 'lineRange', 'fontRange'].forEach(id => {
      const out = document.getElementById(id.replace('Range', 'Out'));
      const slider = document.getElementById(id);
      slider.addEventListener('input', () => out.value = slider.value);
    });

    // Drag and drop
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragging'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('dragging');
      const f = e.dataTransfer.files[0];
      if (f) setFile(f);
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) setFile(fileInput.files[0]);
    });

    function setFile(f) {
      if (!f.type.startsWith('image/')) {
        showStatus('请上传图片文件', 'error');
        return;
      }
      selectedFile = f;
      dropZone.textContent = f.name + ' (' + Math.round(f.size / 1024) + ' KB)';
      submitBtn.disabled = false;
      hideStatus();
    }

    submitBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      submitBtn.disabled = true;
      showStatus('推理中...(CPU 模式下预计 5-15 秒)', 'loading');
      summary.style.display = 'none';
      result.style.display = 'none';
      detectionTable.style.display = 'none';

      const form = new FormData();
      form.append('file', selectedFile);
      form.append('conf', document.getElementById('confRange').value);
      form.append('imgsz', document.getElementById('imgszRange').value);
      form.append('line_width', document.getElementById('lineRange').value);
      form.append('font_size', document.getElementById('fontRange').value);

      try {
        const r = await fetch('/predict', { method: 'POST', body: form });
        if (!r.ok) {
          const err = await r.json().catch(() => ({detail: r.statusText}));
          showStatus('错误: ' + (err.detail || r.status), 'error');
          submitBtn.disabled = false;
          return;
        }
        const data = await r.json();
        document.getElementById('annotatedImg').src = data.annotated_image;
        document.getElementById('originalImg').src = URL.createObjectURL(selectedFile);
        document.getElementById('downloadLink').href = data.annotated_image;

        summary.style.display = 'block';
        summary.textContent = `检测结果: ${data.num_detections} 个区域,耗时 ${data.inference_time_ms}ms`;
        result.style.display = 'grid';

        const tbody = detectionTable.querySelector('tbody');
        tbody.innerHTML = '';
        data.detections.forEach(d => {
          const row = document.createElement('tr');
          row.innerHTML = `<td>${d.id + 1}</td><td>${d.class_name}</td><td>${d.score.toFixed(3)}</td><td>[${d.bbox_xyxy.map(v => v.toFixed(1)).join(', ')}]</td>`;
          tbody.appendChild(row);
        });
        detectionTable.style.display = data.detections.length > 0 ? 'table' : 'none';

        showStatus('完成', 'success');
      } catch (e) {
        showStatus('请求失败: ' + e.message, 'error');
      } finally {
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
```

- [ ] **Step 4: 实现 GET / 路由**

修改 `service/app.py`。在 import 区添加:

```python
from pathlib import Path
from fastapi.responses import FileResponse
```

并在 `create_app` 内,`/predict` 路由后追加:

```python
    @app.get("/", include_in_schema=False)
    def root():
        index_path = Path(__file__).parent / "static" / "index.html"
        return FileResponse(
            index_path,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_app.py -v`
Expected: 全部 passed

- [ ] **Step 6: 提交**

```bash
git add service/app.py service/static/index.html tests/test_app.py
git commit -m "feat(service): add GET / serving static upload UI"
```

---

## Task 10: 编写 service/README.md + 集成冒烟检查

**Files:**
- Create: `service/README.md`
- Manual verification (no automated test changes)

- [ ] **Step 1: 创建 service/README.md**

```markdown
# DocLayout-YOLO HTTP Service

将 DocLayout-YOLO 部署为本地 HTTP 服务,接受图片上传,返回标注图片 + JSON 布局数据。

## 安装

```bash
pip install -e ".[service]"
```

## 启动

```bash
# 1. 下载模型(若还没有)
# 见 README.md 中的模型下载链接

# 2. 设置环境变量并启动
export MODEL_PATH=/path/to/doclayout_yolo_docstructbench_imgsz1024.pt
python service/app.py
```

服务监听 `http://localhost:8000`,浏览器打开即可使用上传界面。

## API 简述

- `GET /` — 上传页面
- `POST /predict` — multipart 上传,返回 JSON (含 base64 标注图 + 检测数据)
- `POST /predict/image` — multipart 上传,直接返回标注图 (`image/jpeg`)
- `GET /health` — 健康检查 (启动期 503,运行期 200)

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL_PATH` | (必填) | 模型 `.pt` 文件路径 |
| `DEVICE` | 自动 | `cpu` / `cuda` / `mps`,留空自动检测 |
| `HOST` | `0.0.0.0` | 绑定地址 |
| `PORT` | `8000` | 端口 |
| `MAX_CONCURRENT` | `1` | 推理并发上限 (GPU 建议保持 1) |
| `MAX_FILE_SIZE_MB` | `20` | 上传文件最大体积 |
```

- [ ] **Step 2: 手动验证启动**

Run:
```bash
MODEL_PATH=path/to/some.pt python -c "
from service.app import create_app
app = create_app()
print('OK:', [r.path for r in app.routes])
"
```
Expected: 打印路由列表,包含 `/`, `/health`, `/predict`, `/predict/image`

- [ ] **Step 3: 手动验证服务可启动且 /health 返 503→200**

Run:
```bash
export MODEL_PATH=/path/to/some.pt
python service/app.py &  # 后台启动
sleep 8
curl -s http://localhost:8000/health
# 期望: {"status":"ok","device":"cpu","model_loaded":true,...}
kill %1
```
Expected: `/health` 返回 `model_loaded: true`

- [ ] **Step 4: 手动验证 curl 上传图片**

Run:
```bash
export MODEL_PATH=/path/to/some.pt
python service/app.py &
SERVER_PID=$!
sleep 8
# 假设有 assets/example/0.jpg
curl -s -o /tmp/result.jpg -F "file=@assets/example/0.jpg" \
  -F "conf=0.2" -F "imgsz=1024" \
  http://localhost:8000/predict/image
ls -lh /tmp/result.jpg
file /tmp/result.jpg  # 应识别为 JPEG
kill $SERVER_PID
```
Expected: `/tmp/result.jpg` 存在且为 JPEG

- [ ] **Step 5: 手动验证 curl 返回 JSON**

Run:
```bash
curl -s -F "file=@assets/example/0.jpg" -F "conf=0.2" \
  http://localhost:8000/predict | python -m json.tool | head -40
```
Expected: 输出包含 `detections`, `num_detections`, `annotated_image`, `inference_time_ms`

- [ ] **Step 6: 提交**

```bash
git add service/README.md
git commit -m "docs(service): add README with install and usage"
```

---

## 自审 (Self-Review) 检查

实施者 (你) 在开始之前应检查:

1. **规范覆盖:** spec 中提到每个 API、性能目标、错误码、UI 元素是否都有对应任务?见下方对照表。
2. **占位符扫描:** 计划中不允许 "TODO"、"TBD"、"实现细节见代码" 等模糊占位符。
3. **类型一致性:** `parse_detections`、`predict_one`、`HealthResponse`、`PredictResponse` 等所有任务中使用的函数/类签名要一致。

### 规范覆盖对照表

| Spec 要求 | 对应任务 |
|---|---|
| 安装 `[service]` extras | Task 1 |
| `MODEL_PATH` 等环境变量配置 | Task 2 |
| 模型单例 + 启动加载 | Task 4 + Task 6 (lifespan) |
| `parse_detections` JSON 化 | Task 3 |
| `predict_one` + 异步 + semaphore | Task 4 |
| `/health` 启动期 503 / 运行期 200 | Task 6 |
| `/predict/image` 直接返图 | Task 7 |
| `/predict` 返回 JSON 含 base64 | Task 8 |
| `GET /` 静态首页 + `Cache-Control: no-store` | Task 9 |
| 简单的 HTML UI 含拖拽/参数滑块/双图对比/检测表 | Task 9 |
| 错误码 (404, 413, 415, 422, 503, 500) | Task 7 + Task 8 |
| 单测 mock 模型,验证业务逻辑 | Tasks 2-9 |
| README + 启动说明 | Task 10 |

---

## 执行交接

完成计划保存后,用户选择执行方式:

**Subagent-Driven (推荐):** 每个任务调度一个全新的 subagent,任务之间审查。
**Inline Execution:** 在当前会话中按任务执行,带检查点。
