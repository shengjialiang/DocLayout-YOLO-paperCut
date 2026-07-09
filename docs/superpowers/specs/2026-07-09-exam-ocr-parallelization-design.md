# DocLayout-YOLO 试卷流水线 OCR 并行化 — 设计文档

- **日期**：2026-07-09
- **作者**：Claude
- **状态**：Approved（待用户最终确认）
- **适用版本**：main（含 `3fadea6` 及之前的 `fix(exam): ...` 系列提交）
- **前置依赖**：[`2026-07-08-doclayout-yolo-exam-pipeline-design.md`](./2026-07-08-doclayout-yolo-exam-pipeline-design.md)

---

## 1. 背景与目标

### 1.1 背景

当前 `service/exam/pipeline.py:142` 的 OCR 阶段以 for 循环**串行**处理每个 `plain text` 框：

```python
for box in plain_expanded:
    ...
    blocks = ocr_engine.ocr_image(crop)
    ...
```

`OcrEngine` 的 `ocr_image` 内部调用 PaddleOCR C++ 推理，单次约 100-500ms（CPU）。一张含 5-8 个 plain text 框的试卷，光 OCR 就要 1-4 秒。配合 `service/config.py:35` 的 `MAX_CONCURRENT=1`（默认串行化所有 HTTP 请求），端到端延迟居高不下。

### 1.2 目标

在不改变 API 契约、不破坏现有测试的前提下：

1. OCR 阶段改为**线程池并行**（CPU 保守版：2 workers）。
2. PaddleOCR 引擎改为**进程内单例**，避免每请求重复懒加载。
3. `MAX_CONCURRENT` 默认值从 1 提升到 2，配合 OCR 内部线程池叠加并发。
4. 所有改动可通过环境变量回退（`MAX_OCR_WORKERS=1` 即恢复纯串行）。

### 1.3 非目标

- 不改 YOLO 推理路径。
- 不改 OCR 模型（仍为 PaddleOCR `ch`）。
- 不改 API 契约（`/predict/exam` 请求/响应字段不变）。
- 不消除 base64 来回转换（属于另一个改进方向 B）。
- 不做进程级 OCR worker pool（避免模型内存翻倍）。

---

## 2. 关键决策（已通过）

| 决策点 | 选择 | 理由 |
|---|---|---|
| 并行模型 | **线程池**（`ThreadPoolExecutor`） | PaddleOCR C++ 推理释放 GIL；线程共享模型内存；启动开销 < 进程池 |
| OCR workers | **默认 2**，env `MAX_OCR_WORKERS` 可调 | CPU-only + 一般机器的稳态值；不会打满 4 核 |
| 引擎实例化 | **模块级单例**（懒加载 + double-check） | 避免每请求 ~1s 的 PaddleOCR 加载；首次并发安全 |
| 请求级并发 | `MAX_CONCURRENT` 默认 **2** | 单线程串行 + 内部 2 workers 浪费多核；env 仍可调回 1 |
| 线程安全 | 信任 PaddleOCR C++ 推理的并发安全 | 社区实践 OK；出问题退化为 `Lock` 串行（仍有单例收益） |
| 顺序保证 | `pool.map` 按入参顺序产出 | 上游 `plain_expanded` 顺序与 `ocr_blocks[i].source_detection_id` 一致 |
| 测试 | 单例/顺序/线程安全各一个单测；不新增 E2E | 单测足够覆盖行为；E2E 慢测试已有 |

---

## 3. 架构

### 3.1 改动前

```
pipeline.run_pipeline
└─ Stage 3: ocr
   ├─ ocr_engine = OcrEngine(lang="ch")     ← 每次新建对象
   └─ for box in plain_expanded:            ← 串行
        ocr_engine.ocr_image(crop)
```

### 3.2 改动后

```
pipeline.run_pipeline
└─ Stage 3: ocr
   ├─ engine = get_engine()                  ← 进程内单例（首次懒加载）
   └─ with ThreadPoolExecutor(max_workers=N) as pool:   ← N=MAX_OCR_WORKERS (env, 默认 2)
        ocr_blocks = list(pool.map(
            _ocr_one_box,
            [(box, img_bgr, w, h, engine) for box in plain_expanded],
        ))
```

`_ocr_one_box` 承载原来 for 循环体里的单 box 逻辑（裁剪、OCR 调用、错误处理、`OcrBlock` 组装）。

### 3.3 引擎单例

```
service/exam/ocr.py
├─ _engine_lock = threading.Lock()
├─ _engine: PaddleOCR | None = None
├─ get_engine(lang="ch") -> PaddleOCR   ← 新增；双检锁懒加载
└─ OcrEngine
   ├─ __init__(lang)                    ← 改为只记 lang，不再持有 _engine
   └─ ocr_image(img)                    ← 内部调 get_engine() + ocr_leftmost
```

`__init__` 保持兼容性（`OcrEngine(lang="ch")` 仍可构造），但实例属性 `_engine` 移除，构造零成本。

---

## 4. 组件改动

### 4.1 `service/exam/ocr.py`

**新增**：

```python
import threading

_engine_lock = threading.Lock()
_engine: "PaddleOCR | None" = None


def get_engine(lang: str = "ch"):
    """Return the process-wide singleton PaddleOCR engine (lazy, thread-safe)."""
    global _engine
    if _engine is None:                    # fast path, no lock
        with _engine_lock:
            if _engine is None:            # double-check under lock
                _engine = _load_paddleocr(lang=lang)
    return _engine


def reset_engine_singleton() -> None:
    """Test helper: drop the cached engine so the next call reloads."""
    global _engine
    with _engine_lock:
        _engine = None
```

**修改** `OcrEngine`：

```python
class OcrEngine:
    def __init__(self, lang: str = "ch") -> None:
        self._lang = lang                  # 只记 lang，不再 self._engine = None

    def ocr_image(self, img: np.ndarray) -> list[TextBlock]:
        engine = get_engine(self._lang)    # 单例
        return ocr_leftmost(engine, img)
```

`_run_ocr` / `_split_into_chunks` / `_leftmost_chunk` / `ocr_leftmost` 不动。

### 4.2 `service/exam/pipeline.py`

**Stage 3 OCR**（`pipeline.py:135-185` 区域）改写：

```python
import concurrent.futures
import os
import threading

MAX_OCR_WORKERS = int(os.environ.get("MAX_OCR_WORKERS", "2"))


def _ocr_one_box(args: tuple) -> OcrBlock:
    """OCR a single cropped box. Runs in a worker thread."""
    box, img_bgr, w, h, engine, regex_compiled = args
    x1, y1, x2, y2 = [int(round(v)) for v in box["bbox_xyxy"]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return OcrBlock(
            source_detection_id=box["id"], bbox_xyxy=box["bbox_xyxy"],
            text="", score=0.0, is_question=False,
        )
    crop = img_bgr[y1:y2, x1:x2]
    try:
        blocks: list[TextBlock] = ocr_leftmost(engine, crop)
    except Exception as e:
        # 错误通过返回值带出，避免 worker 异常打断 pool.map
        return OcrBlock(
            source_detection_id=box["id"], bbox_xyxy=box["bbox_xyxy"],
            text="", score=0.0, is_question=False, _error=f"per-box OCR failed: {e}",
        )
    block_texts = [normalize_question_number(b.text) for b in blocks]
    full_text = " ".join(block_texts) if block_texts else ""
    avg_score = sum(b.score for b in blocks) / len(blocks) if blocks else 0.0
    crop_b64 = _encode_jpeg_base64(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return OcrBlock(
        source_detection_id=box["id"], bbox_xyxy=box["bbox_xyxy"],
        text=full_text, score=avg_score, is_question=False,
        crop_image=crop_b64, block_texts=block_texts,
    )
```

主循环（替换 `for box in plain_expanded` 段）：

```python
try:
    engine = get_engine(lang="ch")
    args_list = [
        (b, img_bgr, w, h, engine, None)
        for b in plain_expanded
    ]
    n_workers = max(1, min(MAX_OCR_WORKERS, len(plain_expanded)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        ocr_blocks = list(pool.map(_ocr_one_box, args_list))
    # 聚合错误
    ocr_failed_count = sum(1 for b in ocr_blocks if getattr(b, "_error", None))
    ocr_first_error = next(
        (b._error for b in ocr_blocks if getattr(b, "_error", None)), None
    )
    # 清理 OcrBlock 上的临时 _error 字段（不入 JSON）
    for b in ocr_blocks:
        if hasattr(b, "_error"):
            delattr(b, "_error")
    ...
```

`OcrBlock` 是 Pydantic `BaseModel` 且默认不允许 extra 字段（见 `service/exam/schemas.py`）。**确定方案**：`_ocr_one_box` 内部 try/except 后返回 `(OcrBlock, error_str | None)` 元组；主线程 `pool.map` 后用 `zip` 拆开并把 `OcrBlock` 填入 `ocr_blocks`、把 `error_str` 通过 `threading.Lock` 保护的 module-level `errors: list[str]` 收集。最终聚合逻辑同改动前：`ocr_failed_count = len(errors)`，`ocr_first_error = errors[0] if errors else None`。

### 4.3 `service/config.py`

```python
max_concurrent=int(os.environ.get("MAX_CONCURRENT", "2")),   # 原 "1"
```

### 4.4 `service/exam/tasks.py`

新增 `reset_for_tests` 内调 `reset_engine_singleton()`：

```python
def reset_for_tests() -> None:
    _store.clear()
    from service.exam.ocr import reset_engine_singleton
    reset_engine_singleton()
```

---

## 5. 数据流

### 5.1 OCR 阶段数据流（改动后）

```
plain_expanded: list[dict]  (按 YOLO 检测顺序)
        │
        ▼
args_list = [(box, img_bgr, w, h, engine), ...] × N
        │
        ▼
ThreadPoolExecutor(max_workers=2)
  ├─ worker 0: _ocr_one_box(args[0]) ─┐
  └─ worker 1: _ocr_one_box(args[1]) ─┤  并行执行 C++ OCR
                                      │
            pool.map 按入参顺序收集 ◀┘
                │
                ▼
ocr_blocks: list[OcrBlock]  ← 顺序与 plain_expanded 一一对应
```

`source_detection_id` 不变；`block_texts` 顺序由 PaddleOCR 决定（不依赖并发调度）；`_ocr_one_box` 内部无共享可变状态。

### 5.2 线程安全矩阵

| 共享资源 | 访问方 | 是否线程安全 | 备注 |
|---|---|---|---|
| `_engine: PaddleOCR` | 多线程 `get_engine()` | ✅（双检锁） | 懒加载完成后只读 |
| `PaddleOCR.ocr()` C++ 内部 | 多线程并发调用 | ⚠️ 信任社区实践 | 失败时退化为 `Lock` 串行 |
| `img_bgr: np.ndarray` | 多线程只读切片 | ✅ | `arr[y1:y2, x1:x2]` 返回 view，不修改原数组 |
| `errors: list[str]` | 多线程 append | ❌ | 改用 `threading.Lock` 保护 |

### 5.3 配置与回退

| 环境变量 | 默认 | 设为 1 的效果 |
|---|---|---|
| `MAX_OCR_WORKERS=1` | 2 | OCR 完全串行（行为等同改动前） |
| `MAX_CONCURRENT=1` | 2 | HTTP 请求级串行（OCR 内部仍可并行） |

两者都设为 1 = 完全等价于改动前的行为。运维可按机器配置渐进调高。

---

## 6. 错误处理

| 阶段 | 改动前 | 改动后 |
|---|---|---|
| 引擎加载失败 | `OcrEngine.__init__` 抛异常 → 整个 OCR 阶段 error | `get_engine()` 抛异常 → 同左；多线程下只触发一次 |
| 单 box OCR 抛异常 | 计入 `ocr_failed_count` + `ocr_first_error` | 通过 worker 返回值带出；聚合逻辑不变 |
| pool 内部异常 | N/A | `pool.map` 不会吞异常；`_ocr_one_box` 内部 try/except 保证不外抛 |
| 线程池关闭 | N/A | `with` 语句自动 `shutdown(wait=True)`，确保 worker 完成 |

不变：
- 单阶段异常 → `stages.ocr.error`，任务不标 failed（与改动前一致）。
- 致命错误（YOLO 崩溃、解码失败）路径不变。

---

## 7. 测试策略

### 7.1 新增单测（`tests/exam/test_ocr.py`）

```python
def test_get_engine_returns_singleton():
    """get_engine() called twice returns the same instance (caching works)."""
    reset_engine_singleton()
    monkeypatch.setattr(ocr, "_load_paddleocr", lambda lang="ch": object())
    e1 = get_engine()
    e2 = get_engine()
    assert e1 is e2


def test_get_engine_concurrent_init_only_once():
    """Under 10 threads racing on first get_engine(), _load_paddleocr called once."""
    reset_engine_singleton()
    call_count = [0]
    def fake_load(lang="ch"):
        call_count[0] += 1
        time.sleep(0.05)  # widen race window
        return object()
    monkeypatch.setattr(ocr, "_load_paddleocr", fake_load)

    threads = [threading.Thread(target=get_engine) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert call_count[0] == 1
```

### 7.2 新增单测（`tests/exam/test_pipeline.py`）

```python
def test_pipeline_uses_threadpool_for_ocr(monkeypatch):
    """Pipeline.run_pipeline invokes ThreadPoolExecutor in OCR stage."""
    executor_called = [False]
    real_executor = concurrent.futures.ThreadPoolExecutor
    def spy_executor(max_workers):
        executor_called[0] = True
        return real_executor(max_workers)
    monkeypatch.setattr(pipeline.concurrent.futures, "ThreadPoolExecutor", spy_executor)
    # ... run pipeline with mocked OCR ...
    assert executor_called[0]


def test_pipeline_ocr_block_order_preserved():
    """Plaintext boxes processed in parallel must come out in input order."""
    # feed 5 plain text boxes, OCR returns delay-scaled results,
    # assert ocr_blocks[i].source_detection_id == input_boxes[i]["id"]
```

### 7.3 不变的测试

`tests/exam/test_ocr.py` 现有用例（mock `_load_paddleocr`）应继续通过；`reset_engine_singleton()` 在 fixture 里调用避免跨用例污染。

### 7.4 性能基准（手动，不进 CI）

`scripts/bench_ocr_parallel.py`（**不进 git**，仅本地对比）：

```python
# 同一张 5-plain-text 试卷，串行 vs 2 workers vs 4 workers
# 输出每阶段 duration_ms 表格
```

预期（CPU i5 4 核 估算）：

| workers | OCR 阶段耗时 | 端到端 |
|---|---|---|
| 1（基线） | ~1800ms | ~2500ms |
| 2 | ~1000ms | ~1700ms |
| 4 | ~900ms | ~1600ms（边际收益递减） |

---

## 8. 文件改动清单

### 8.1 修改

```
service/exam/ocr.py          # +get_engine/reset_engine_singleton; OcrEngine 改用单例
service/exam/pipeline.py     # Stage 3 OCR 改用 ThreadPoolExecutor + _ocr_one_box
service/config.py            # MAX_CONCURRENT 默认 1→2
service/exam/tasks.py        # reset_for_tests 调 reset_engine_singleton
```

### 8.2 新增测试

```
tests/exam/test_ocr.py       # +test_get_engine_returns_singleton
                             # +test_get_engine_concurrent_init_only_once
tests/exam/test_pipeline.py  # +test_pipeline_uses_threadpool_for_ocr
                             # +test_pipeline_ocr_block_order_preserved
```

### 8.3 不变

- `service/exam/expand.py`、`geometry.py`、`question_filter.py`、`schemas.py`
- `service/app.py`、`inference.py`、`schemas.py`
- `service/exam/tasks.py` 主体（仅 reset_for_tests 加一行）
- API 契约、UI、`pyproject.toml`

---

## 9. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| PaddleOCR C++ 多线程不安全（崩溃/段错误） | 中 | OCR 阶段失败 | 设 `MAX_OCR_WORKERS=1` 立即回退；后续改 per-thread engine 缓存 |
| 多线程 base64 编码抢占 CPU 导致波动 | 低 | 单次延迟抖动 | workers=2 已限定；监控 |
| 单例 + 测试隔离：`get_engine` 缓存导致单测之间污染 | 中 | 假阳性/假阴性 | `reset_for_tests` 调 `reset_engine_singleton`；新增 fixture |
| 现有 E2E（`@pytest.mark.slow`）变慢/挂掉 | 低 | CI 卡住 | 该测试目前已 skip（在 `MODEL_PATH` 未设置时）；本地可手动跑 |
| `MAX_CONCURRENT=2` 双请求同时加载 PaddleOCR 引发双倍内存 | 低（懒加载只在首次） | 启动期短暂内存翻倍 | 接受；首次加载 ~200MB，10 秒内结束 |

---

## 10. 成功标准

- [ ] `MAX_OCR_WORKERS=1, MAX_CONCURRENT=1` 行为与改动前一致（基线）
- [ ] 默认配置下，5-plain-text 试卷端到端延迟下降 ≥ 30%
- [ ] 全部 pytest 通过（含新增 4 个单测）
- [ ] `service/exam/ocr.py` 新增单例/线程安全单测；`service/exam/pipeline.py` 新增线程池 + 顺序单测
- [ ] 现有 E2E（`tests/exam/test_e2e.py`）手动跑通
- [ ] 文档：本 spec + 后续 plan 进入 `docs/superpowers/specs/` & `docs/superpowers/plans/`
- [ ] 无 API 契约变更；UI 不动
