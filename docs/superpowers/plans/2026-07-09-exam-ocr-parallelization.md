# 试卷 OCR 并行化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `service/exam/pipeline.py` OCR 阶段从 for 循环串行改为 `ThreadPoolExecutor` 并行（CPU 保守版：默认 2 workers），并把 `OcrEngine` 改为进程内单例，避免每请求重复加载 PaddleOCR；`MAX_CONCURRENT` 默认 1→2。

**Architecture:** 改动分两层：引擎层（`service/exam/ocr.py`）新增 `get_engine()` 双检锁单例；流水线层（`service/exam/pipeline.py`）把 Stage 3 OCR 的 for 循环改为 `_ocr_one_box` worker + `ThreadPoolExecutor.map`，worker 返回 `(OcrBlock, error_str | None)` 元组，主线程解包并聚合错误。`OcrEngine` 类保留向后兼容（构造零成本，内部调 `get_engine`）。所有改动通过 `MAX_OCR_WORKERS` / `MAX_CONCURRENT` 环境变量可回退到现状。

**Tech Stack:** Python 3.10+、`concurrent.futures.ThreadPoolExecutor`（stdlib）、PaddleOCR（已有懒加载）、pytest（已有）。

**Spec:** `docs/superpowers/specs/2026-07-09-exam-ocr-parallelization-design.md`

---

## 文件结构

```
service/exam/
├── ocr.py             # +get_engine/reset_engine_singleton; OcrEngine 内部调 get_engine
├── pipeline.py        # Stage 3 OCR 改用 ThreadPoolExecutor.map(_ocr_one_box, ...)
├── tasks.py           # reset_for_tests 调 reset_engine_singleton
└── (其他文件不动)

service/
└── config.py          # MAX_CONCURRENT 默认 1 → 2

tests/exam/
├── test_ocr.py        # +test_get_engine_returns_singleton
│                      # +test_get_engine_concurrent_init_only_once
│                      # +autouse fixture reset_engine_singleton
└── test_pipeline.py   # +test_pipeline_ocr_uses_thread_pool
                       # +test_pipeline_ocr_block_order_preserved
```

---

## Task 1: 添加 OcrEngine 进程内单例 `get_engine()`

**Files:**
- Modify: `service/exam/ocr.py:1-145`（在 `_load_paddleocr` 上方加 import + 新增函数）
- Modify: `tests/exam/test_ocr.py:1-12`（import 新增的 `get_engine`、`reset_engine_singleton`）

- [ ] **Step 1: 编辑 `tests/exam/test_ocr.py`，在文件顶部 import 区追加两行**

将第 5-11 行：
```python
from service.exam.ocr import (
    OcrEngine,
    TextBlock,
    _leftmost_chunk,
    _split_into_chunks,
    ocr_leftmost,
)
```

替换为：
```python
from service.exam.ocr import (
    OcrEngine,
    TextBlock,
    _leftmost_chunk,
    _split_into_chunks,
    get_engine,
    ocr_leftmost,
    reset_engine_singleton,
)
```

- [ ] **Step 2: 写第一个失败测试 — `test_get_engine_returns_singleton`**

在 `tests/exam/test_ocr.py` 末尾追加：

```python
def test_get_engine_returns_singleton(monkeypatch):
    """get_engine() returns the same instance across calls (process-wide cache)."""
    reset_engine_singleton()
    monkeypatch.setattr("service.exam.ocr._load_paddleocr", lambda lang="ch": "fake-engine")
    e1 = get_engine()
    e2 = get_engine()
    assert e1 is e2
    assert e1 == "fake-engine"
```

- [ ] **Step 3: 运行测试，验证失败**

Run: `python -m pytest tests/exam/test_ocr.py::test_get_engine_returns_singleton -v`
Expected: `FAILED` — `ImportError: cannot import name 'get_engine'` 或 `reset_engine_singleton`

- [ ] **Step 4: 在 `service/exam/ocr.py` 中实现 `get_engine` + `reset_engine_singleton`**

将 `service/exam/ocr.py:1-12` 区域（`from __future__ import annotations` 之后到 `def _load_paddleocr` 之前）替换为：

```python
"""PaddleOCR engine wrapper with lazy loading."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

# Above this aspect ratio, PaddleOCR's DB text detector fails to find most
# text lines (returns ~1 block instead of the full text). Empirically
# determined from real exam crops produced by the YOLO detector.
MAX_ASPECT_RATIO = 4.0

# Process-wide singleton state. Double-checked locking in get_engine() so
# first-call concurrent access still triggers exactly one PaddleOCR load.
_engine_lock = threading.Lock()
_engine: "object | None" = None


def get_engine(lang: str = "ch"):
    """Return the process-wide singleton PaddleOCR engine (lazy + thread-safe)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = _load_paddleocr(lang=lang)
    return _engine


def reset_engine_singleton() -> None:
    """Test helper: drop the cached engine so the next get_engine() reloads."""
    global _engine
    with _engine_lock:
        _engine = None
```

- [ ] **Step 5: 运行测试，验证通过**

Run: `python -m pytest tests/exam/test_ocr.py::test_get_engine_returns_singleton -v`
Expected: `PASSED`

- [ ] **Step 6: 写第二个失败测试 — 并发初始化只触发一次加载**

在 `tests/exam/test_ocr.py` 末尾追加：

```python
def test_get_engine_concurrent_init_only_once(monkeypatch):
    """10 threads racing on first get_engine() must trigger exactly one load."""
    import threading
    import time

    reset_engine_singleton()
    call_count = {"n": 0}
    load_started = threading.Event()

    def fake_load(lang="ch"):
        call_count["n"] += 1
        load_started.set()
        time.sleep(0.05)  # widen race window
        return "fake-engine"

    monkeypatch.setattr("service.exam.ocr._load_paddleocr", fake_load)

    threads = [threading.Thread(target=get_engine) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert call_count["n"] == 1
    assert get_engine() == "fake-engine"
```

- [ ] **Step 7: 运行测试，验证通过（双检锁已实现）**

Run: `python -m pytest tests/exam/test_ocr.py::test_get_engine_concurrent_init_only_once -v`
Expected: `PASSED`

- [ ] **Step 8: 跑全部 OCR 测试，确认基线通过**

Run: `python -m pytest tests/exam/test_ocr.py -v`
Expected: All existing tests still pass (the `OcrEngine` class hasn't been changed yet).

- [ ] **Step 9: 提交**

```bash
git add service/exam/ocr.py tests/exam/test_ocr.py
git commit -m "feat(exam): add process-wide OcrEngine singleton with double-checked locking"
```

---

## Task 2: `OcrEngine.ocr_image` 改用单例

**Files:**
- Modify: `service/exam/ocr.py:130-145`（`OcrEngine` 类）
- Modify: `tests/exam/test_ocr.py:1-25`（加 autouse fixture 防止单例状态泄漏）

- [ ] **Step 1: 加 autouse fixture 复位单例**

将 `tests/exam/test_ocr.py` 第 1-12 行替换为：

```python
"""OCR engine tests with mocked PaddleOCR (no model download)."""
import threading

import numpy as np
import pytest

from service.exam.ocr import (
    OcrEngine,
    TextBlock,
    _leftmost_chunk,
    _split_into_chunks,
    get_engine,
    ocr_leftmost,
    reset_engine_singleton,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test starts/ends with no cached engine (avoid state leakage)."""
    reset_engine_singleton()
    yield
    reset_engine_singleton()
```

注意：`reset_for_tests` 等价物在 `service/exam/tasks.py`（Task 5 引入），但 OCR 测试不依赖任务存储，所以这里用本地 fixture。

- [ ] **Step 2: 跑现有 OCR 测试，确认仍然通过**

Run: `python -m pytest tests/exam/test_ocr.py -v`
Expected: All tests PASS（autouse fixture 在每个用例前后复位，行为等价）

- [ ] **Step 3: 重构 `OcrEngine.ocr_image` 使用单例**

将 `service/exam/ocr.py:130-145`（`OcrEngine` 类）替换为：

```python
class OcrEngine:
    """Thin wrapper that delegates to the process-wide PaddleOCR singleton.

    Construction is zero-cost: no engine state is held on the instance.
    Backward-compatible with the original per-instance caching API.
    """

    def __init__(self, lang: str = "ch") -> None:
        self._lang = lang

    def ocr_image(self, img: np.ndarray) -> list[TextBlock]:
        return ocr_leftmost(get_engine(self._lang), img)
```

- [ ] **Step 4: 跑全部 OCR 测试**

Run: `python -m pytest tests/exam/test_ocr.py -v`
Expected: All tests PASS（所有现有测试 monkeypatch `_load_paddleocr` 并 reset 单例，行为保持）

- [ ] **Step 5: 跑全套测试，确认基线没破**

Run: `python -m pytest tests/ -v`
Expected: All previously-passing tests still pass. `test_ocr.py` 现有 14 个 + 新增 2 个 = 16 个全通过。

- [ ] **Step 6: 提交**

```bash
git add service/exam/ocr.py tests/exam/test_ocr.py
git commit -m "refactor(exam): route OcrEngine.ocr_image through process-wide singleton"
```

---

## Task 3: 流水线 OCR 阶段改用 ThreadPoolExecutor

**Files:**
- Modify: `service/exam/pipeline.py:1-25`（import 区加 `concurrent.futures`、`os`、`OcrEngine` 仍从 ocr 导入）
- Modify: `service/exam/pipeline.py:135-185`（Stage 3 OCR 整段重写）
- Modify: `tests/exam/test_pipeline.py:1-10`（import 加 `concurrent.futures`、必要类型）

- [ ] **Step 1: 写失败测试 — 流水线使用 ThreadPoolExecutor**

在 `tests/exam/test_pipeline.py` 末尾追加：

```python
def test_pipeline_ocr_uses_thread_pool(monkeypatch):
    """Stage 3 OCR must dispatch via concurrent.futures.ThreadPoolExecutor."""
    import concurrent.futures as cf

    async def fake_predict_one(**kwargs):
        return {
            "width": 800, "height": 1000,
            "annotated_base64": "data:img",
            "detections": [
                {"id": 0, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [50, 100, 750, 200], "bbox_xywh": [400, 150, 700, 100],
                 "score": 0.9},
            ],
            "num_detections": 1, "inference_time_ms": 100,
            "model_imgsz": 1024, "conf_threshold": 0.3, "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((1000, 800, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline, "OcrEngine", lambda lang="ch": mock_ocr_empty())
    monkeypatch.setattr(pipeline, "redraw_with_questions",
                        lambda *a, **k: "data:image/jpeg;base64,redrawn")

    seen = {"max_workers": None}

    class SpyExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
            seen["max_workers"] = max_workers
            self._real = cf.ThreadPoolExecutor(max_workers)
        def __enter__(self):
            return self._real.__enter__()
        def __exit__(self, *args):
            return self._real.__exit__(*args)
        def map(self, fn, iterable, **kwargs):
            return self._real.map(fn, iterable, **kwargs)

    monkeypatch.setattr(pipeline.concurrent.futures, "ThreadPoolExecutor", SpyExecutor)

    tid = tasks.create_task()
    pipeline.run_pipeline(tid, b"fake", {
        "expand_mode": "pixel", "expand_top": 0, "expand_bottom": 0,
        "expand_left": 0, "expand_right": 0,
    }, model=None, semaphore=asyncio.Semaphore(1))

    state = tasks.get_task(tid)
    assert state["status"] == "done", state.get("error")
    assert seen["max_workers"] is not None and seen["max_workers"] >= 1
```

- [ ] **Step 2: 运行测试，验证失败**

Run: `python -m pytest tests/exam/test_pipeline.py::test_pipeline_ocr_uses_thread_pool -v`
Expected: `FAILED` — `AttributeError: module 'service.exam.pipeline' has no attribute 'concurrent'` 或 `assert seen["max_workers"] is None`

- [ ] **Step 3: 写失败测试 — OCR 块顺序保持**

在 `tests/exam/test_pipeline.py` 末尾追加：

```python
def test_pipeline_ocr_block_order_preserved(monkeypatch):
    """pool.map semantics: ocr_blocks must come out in plain_expanded order
    even if individual OCR calls complete out of order."""
    import time

    async def fake_predict_one(**kwargs):
        return {
            "width": 1000, "height": 1000,
            "annotated_base64": "data:img",
            "detections": [
                {"id": i, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [10, 100 + i * 80, 990, 100 + i * 80 + 60],
                 "bbox_xywh": [500, 130 + i * 80, 980, 60],
                 "score": 0.9}
                for i in range(5)
            ],
            "num_detections": 5, "inference_time_ms": 100,
            "model_imgsz": 1024, "conf_threshold": 0.3, "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((1000, 1000, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline, "redraw_with_questions",
                        lambda *a, **k: "data:image/jpeg;base64,redrawn")

    class OcrTagsByCallOrder:
        def __init__(self):
            self.n = 0
        def ocr_image(self, img):
            self.n += 1
            order = self.n
            # Odd calls finish first (would scramble order without pool.map).
            time.sleep(0.04 if order % 2 == 1 else 0.10)
            return [TextBlock(text=f"box{order}", bbox=[0, 0, 10, 10], score=0.9)]

    monkeypatch.setattr(pipeline, "OcrEngine",
                        lambda lang="ch": OcrTagsByCallOrder())

    tid = tasks.create_task()
    pipeline.run_pipeline(tid, b"fake", {
        "expand_mode": "pixel", "expand_top": 0, "expand_bottom": 0,
        "expand_left": 0, "expand_right": 0,
    }, model=None, semaphore=asyncio.Semaphore(1))

    state = tasks.get_task(tid)
    assert state["status"] == "done", state.get("error")
    block_texts = [b.block_texts[0] for b in state["final"].ocr_blocks]
    assert block_texts == ["box1", "box2", "box3", "box4", "box5"], block_texts
```

- [ ] **Step 4: 运行测试，验证失败（pipeline 还没改）**

Run: `python -m pytest tests/exam/test_pipeline.py::test_pipeline_ocr_block_order_preserved -v`
Expected: `FAILED` — `AttributeError: module 'service.exam.pipeline' has no attribute 'concurrent'` 或断言失败（当前串行执行下 block_texts 顺序是 ["box1","box2","box3","box4","box5"] 也会过；这条是改完之后的回归保护，所以必须先改 pipeline）

- [ ] **Step 5: 改 `service/exam/pipeline.py` 的 import 区**

将 `service/exam/pipeline.py:1-23`：

```python
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
from service.exam.question_filter import is_question, normalize_question_number
from service.exam.schemas import FinalResult, OcrBlock, Question
```

替换为：

```python
"""5-stage exam pipeline orchestrator.

Stages: decode → yolo → expand → ocr → filter → geometry → redraw.
Intermediate errors are recorded per-stage but don't fail the task.
Fatal errors (decode failure, YOLO crash) mark the task failed.
"""
from __future__ import annotations

import base64
import concurrent.futures
import io
import os
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from service.exam import tasks
from service.exam.expand import expand_boxes
from service.exam.geometry import extend_questions
from service.exam.ocr import OcrEngine, TextBlock
from service.exam.question_filter import is_question, normalize_question_number
from service.exam.schemas import FinalResult, OcrBlock, Question

# Default OCR worker count. Tunable via env for fallback to fully-serial
# behavior on weak hardware (set MAX_OCR_WORKERS=1).
MAX_OCR_WORKERS = int(os.environ.get("MAX_OCR_WORKERS", "2"))
```

- [ ] **Step 6: 替换 Stage 3 OCR 整段**

将 `service/exam/pipeline.py:135-185`（即 `# Stage 3: OCR` 整段）替换为：

```python
    # Stage 3: OCR
    t0 = time.perf_counter()
    ocr_blocks: list[OcrBlock] = []
    ocr_failed_count = 0
    ocr_first_error: str | None = None
    try:
        if not plain_expanded:
            tasks.update_stage(
                task_id, "ocr",
                duration_ms=int((time.perf_counter() - t0) * 1000),
                payload={"num_blocks": 0},
            )
        else:
            args_list = [(box, img_bgr, w, h) for box in plain_expanded]
            n_workers = max(1, min(MAX_OCR_WORKERS, len(plain_expanded)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                results: list[tuple[OcrBlock, str | None]] = list(
                    pool.map(_ocr_one_box, args_list)
                )
            ocr_blocks = [r[0] for r in results]
            errors = [r[1] for r in results if r[1] is not None]
            ocr_failed_count = len(errors)
            ocr_first_error = errors[0] if errors else None
            ocr_payload: dict = {"num_blocks": len(ocr_blocks)}
            if ocr_failed_count:
                ocr_payload["num_failed"] = ocr_failed_count
            tasks.update_stage(
                task_id, "ocr",
                duration_ms=int((time.perf_counter() - t0) * 1000),
                payload=ocr_payload,
                error=ocr_first_error,
            )
    except Exception as e:
        tasks.update_stage(task_id, "ocr", duration_ms=0, error=str(e))
        ocr_blocks = []
```

- [ ] **Step 7: 在 `pipeline.py` 文件顶部（imports 之后）新增 `_ocr_one_box` worker**

在 `service/exam/pipeline.py` 中，紧跟 `def _to_rgb` 函数（约第 50 行）之后、`def redraw_with_questions` 之前，插入：

```python
def _ocr_one_box(args: tuple) -> tuple[OcrBlock, str | None]:
    """OCR a single cropped plain-text box. Returns (block, error_str).

    Runs in a worker thread; per-box exceptions are caught and surfaced as
    the second tuple element so the main thread can aggregate them after
    pool.map returns.
    """
    box, img_bgr, w, h = args
    x1, y1, x2, y2 = [int(round(v)) for v in box["bbox_xyxy"]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return (
            OcrBlock(
                source_detection_id=box["id"],
                bbox_xyxy=box["bbox_xyxy"],
                text="", score=0.0, is_question=False,
            ),
            None,
        )
    crop = img_bgr[y1:y2, x1:x2]
    try:
        blocks: list[TextBlock] = OcrEngine(lang="ch").ocr_image(crop)
    except Exception as e:
        return (
            OcrBlock(
                source_detection_id=box["id"],
                bbox_xyxy=box["bbox_xyxy"],
                text="", score=0.0, is_question=False,
            ),
            f"per-box OCR failed: {e}",
        )
    block_texts = [normalize_question_number(b.text) for b in blocks]
    full_text = " ".join(block_texts) if block_texts else ""
    avg_score = sum(b.score for b in blocks) / len(blocks) if blocks else 0.0
    crop_b64 = _encode_jpeg_base64(
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return (
        OcrBlock(
            source_detection_id=box["id"],
            bbox_xyxy=box["bbox_xyxy"],
            text=full_text, score=avg_score,
            is_question=False,  # set in filter stage
            crop_image=crop_b64,
            block_texts=block_texts,
        ),
        None,
    )
```

- [ ] **Step 8: 跑两个新 pipeline 测试，验证通过**

Run: `python -m pytest tests/exam/test_pipeline.py::test_pipeline_ocr_uses_thread_pool tests/exam/test_pipeline.py::test_pipeline_ocr_block_order_preserved -v`
Expected: `2 passed`

- [ ] **Step 9: 跑全部 pipeline 测试，确认回归通过**

Run: `python -m pytest tests/exam/test_pipeline.py -v`
Expected: All pre-existing tests (10+ 个) + 新增 2 个全通过。特别注意：
- `test_pipeline_runs_through_all_stages`（用 `monkeypatch.setattr(pipeline, "OcrEngine", ...)`）→ `_ocr_one_box` 走 mock OcrEngine，仍然工作
- `test_pipeline_preserves_per_box_ocr_error` → `FailingOcr.ocr_image` 抛 RuntimeError，worker 捕获后写入 errors 列表，`stages.ocr.error` 有值
- `test_pipeline_question_number_not_first_ocr_block`、`test_pipeline_multiline_question_with_offset_lines`、`test_pipeline_no_question_number_anywhere_is_not_a_question` → 行为不变

- [ ] **Step 10: 跑全套测试**

Run: `python -m pytest tests/ -v`
Expected: 全套通过

- [ ] **Step 11: 提交**

```bash
git add service/exam/pipeline.py tests/exam/test_pipeline.py
git commit -m "perf(exam): parallelize OCR stage with ThreadPoolExecutor (env-tunable workers)"
```

---

## Task 4: `MAX_CONCURRENT` 默认 1→2

**Files:**
- Modify: `service/config.py:35`（单行）

- [ ] **Step 1: 修改默认并发数**

将 `service/config.py:35`：

```python
        max_concurrent=int(os.environ.get("MAX_CONCURRENT", "1")),
```

替换为：

```python
        max_concurrent=int(os.environ.get("MAX_CONCURRENT", "2")),
```

- [ ] **Step 2: 跑 config 测试**

Run: `python -m pytest tests/test_config.py -v`
Expected: All PASS（默认改动不影响单测，单测通过 env 显式设值）

- [ ] **Step 3: 跑全套测试**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: 提交**

```bash
git add service/config.py
git commit -m "perf(service): bump MAX_CONCURRENT default from 1 to 2"
```

---

## Task 5: `tasks.reset_for_tests` 也复位 OCR 单例

**Files:**
- Modify: `service/exam/tasks.py:119-122`

- [ ] **Step 1: 修改 `reset_for_tests`**

将 `service/exam/tasks.py:119-122`：

```python
def reset_for_tests() -> None:
    """Clear all tasks (test helper)."""
    _store.clear()
```

替换为：

```python
def reset_for_tests() -> None:
    """Clear all tasks and drop the OCR engine singleton (test helper)."""
    _store.clear()
    from service.exam.ocr import reset_engine_singleton
    reset_engine_singleton()
```

延迟 import 是为了避免 `tasks.py` 启动时加载 `service.exam.ocr`（会拖慢 `/predict` 启动）——`tasks.py` 在普通请求路径里不依赖 OCR。

- [ ] **Step 2: 跑 task 测试**

Run: `python -m pytest tests/exam/test_tasks.py -v`
Expected: All PASS

- [ ] **Step 3: 跑全套测试**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: 提交**

```bash
git add service/exam/tasks.py
git commit -m "test(exam): reset OCR engine singleton in tasks.reset_for_tests"
```

---

## Task 6: 端到端验证 + 回退路径确认

**Files:** 无（只跑测试 + 手动 smoke）

- [ ] **Step 1: 跑全套测试**

Run: `python -m pytest tests/ -v`
Expected: All PASS（之前任务都通过，累计新增 4 个测试：test_ocr.py +2、test_pipeline.py +2）

- [ ] **Step 2: 验证回退路径 — 串行配置下行为不变**

Run:
```bash
MAX_OCR_WORKERS=1 MAX_CONCURRENT=1 python -m pytest tests/ -v
```
Expected: All PASS

（说明：workers=1 串行执行，max_concurrent=1 不并发，行为等价改动前。所有测试仍通过。）

- [ ] **Step 3: 验证 `MAX_OCR_WORKERS=4`（更激进）也不挂**

Run:
```bash
MAX_OCR_WORKERS=4 python -m pytest tests/exam/test_pipeline.py -v
```
Expected: All PASS（pipeline 测试用 mock，不依赖真实 PaddleOCR，workers=4 也能跑通；只测逻辑正确性）

- [ ] **Step 4: 手动 smoke（仅在能跑 PaddleOCR 的环境）**

如有 PaddleOCR 模型：

```bash
MODEL_PATH=models/doclayout_yolo_docstructbench_imgsz1024.pt \
  python -m service.app &
sleep 3
curl -sS -X POST http://localhost:8000/predict/exam \
  -F "file=@第五题.jpg" \
  -F "expand_mode=pixel" \
  -F "expand_top=20" -F "expand_bottom=20" \
  -F "expand_left=20" -F "expand_right=20" \
  | tee /tmp/submit.json
TASK_ID=$(python -c "import json; print(json.load(open('/tmp/submit.json'))['task_id'])")
# 等 done
for i in 1 2 3 4 5 6 7 8 9 10; do
  STATUS=$(curl -sS http://localhost:8000/predict/exam/$TASK_ID/status | python -c "import json,sys; print(json.load(sys.stdin)['status'])")
  echo "poll $i: $STATUS"
  [ "$STATUS" = "done" ] && break
  sleep 1
done
curl -sS http://localhost:8000/predict/exam/$TASK_ID/status \
  | python -c "import json,sys; d=json.load(sys.stdin); print('stages:', d['stages']); print('num_questions:', len(d['final']['questions']))"
```

Expected: `status: done`；`stages.ocr.duration_ms` 比改动前的等价物小 ≥ 30%；`num_questions` 至少 1。

若无 PaddleOCR：跳过此步。

- [ ] **Step 5: 最终 commit（如有 README/CHANGELOG 更新）**

如果改了文档（README、CHANGELOG），commit：

```bash
git add <changed-docs>
git commit -m "docs: document MAX_OCR_WORKERS and updated MAX_CONCURRENT default"
```

否则跳过。

---

## Self-Review

按 spec 各节检查 plan 覆盖度：

| Spec 章节 | Plan 任务 |
|---|---|
| 1.1/1.2 背景与目标（线程池 + 单例 + MAX_CONCURRENT 提升） | Task 1-4 |
| 1.3 非目标（API 不变、不动 base64 等） | 不在 plan 中改动，符合 |
| 2 关键决策 | Task 1（双检锁）、Task 3（MAX_OCR_WORKERS=2）、Task 4（MAX_CONCURRENT=2） |
| 3.2 架构改动 | Task 1-3 实现 |
| 3.3 引擎单例 | Task 1 + Task 2 |
| 4.1 `ocr.py` get_engine/reset_engine_singleton | Task 1 |
| 4.1 OcrEngine 改用单例 | Task 2 |
| 4.2 `pipeline.py` 线程池 + `_ocr_one_box` | Task 3 |
| 4.3 `config.py` MAX_CONCURRENT 默认 1→2 | Task 4 |
| 4.4 `tasks.reset_for_tests` 调 `reset_engine_singleton` | Task 5 |
| 5.1 数据流 | Task 3 实现 |
| 5.2 线程安全矩阵 | Task 1（双检锁）+ Task 3（`img_bgr` 只读切片） |
| 5.3 配置与回退 | Task 6 Step 2 验证 |
| 6 错误处理 | Task 3 异常→errors 列表聚合（与改动前一致） |
| 7.1 test_ocr.py 新增 | Task 1（2 测试）+ Task 2（autouse fixture） |
| 7.2 test_pipeline.py 新增 | Task 3（2 测试） |
| 7.3 现有测试兼容 | Task 2 Step 2 + Task 3 Step 9 验证 |
| 8.1/8.2/8.3 文件改动清单 | 全部覆盖 |
| 9 风险 | Task 1 解决线程安全；Task 6 验证回退 |
| 10 成功标准 | Task 6 验证 |

无遗漏。

**类型一致性**：
- `get_engine(lang: str = "ch")` 在 Task 1 定义，Task 2/3/5 引用一致。
- `reset_engine_singleton()` 在 Task 1 定义，Task 2 fixture、Task 5 `reset_for_tests` 引用一致。
- `OcrEngine(lang="ch").ocr_image(img)` 在 Task 2 确认行为，Task 3 worker 使用一致。
- `_ocr_one_box(args) -> tuple[OcrBlock, str | None]` 在 Task 3 定义、Task 3 主循环解包一致。
- `MAX_OCR_WORKERS` 在 Task 3 引入，Task 6 回退测试引用。

**无 placeholders**：所有代码块都是完整可粘贴的代码。命令都带预期输出。
