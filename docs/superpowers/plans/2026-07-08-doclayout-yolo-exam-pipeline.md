# DocLayout-YOLO 试卷版面流水线 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增异步 `/predict/exam` 端点，对试卷图片做"模型检测 → 框放大 → OCR → 题号判定 → 坐标延伸 → 重绘"，始终返回多阶段调试数据。

**Architecture:** 在 `service/exam/` 子包下实现 5 阶段流水线（decode→yolo→expand→ocr→filter→geometry），由 `service/app.py` 增加 POST 提交 + GET 状态轮询端点；UI 在 `index.html` 增加"试卷模式"页签。OCR 用 PaddleOCR 懒加载，任务用内存 dict + LRU。

**Tech Stack:** FastAPI（已有）、PaddleOCR（新增）、pytest（已有）、Pydantic v2（已有）。

**Spec:** `docs/superpowers/specs/2026-07-08-doclayout-yolo-exam-pipeline-design.md`

---

## 文件结构

```
service/exam/
├── __init__.py
├── schemas.py         # Pydantic 模型
├── expand.py          # 框放大
├── question_filter.py # 题号正则
├── geometry.py        # 坐标延伸
├── tasks.py           # 内存任务存储
├── ocr.py             # PaddleOCR 封装
└── pipeline.py        # 5 阶段编排

tests/exam/
├── __init__.py
├── test_expand.py
├── test_question_filter.py
├── test_geometry.py
├── test_tasks.py
├── test_ocr.py
├── test_pipeline.py
└── test_app_exam.py

tests/fixtures/exam/
├── single_question.png
├── three_questions.png
└── no_plain_text.png

# 修改
service/app.py
service/static/index.html
pyproject.toml
```

---

## Task 1: 添加 [exam] 可选依赖到 pyproject.toml

**Files:**
- Modify: `pyproject.toml:84-91`（`[project.optional-dependencies]` 区域）

- [ ] **Step 1: 编辑 pyproject.toml**

在 `service = [...]` 块后追加新块：

```toml
exam = [
    "paddleocr>=2.7",
    "paddlepaddle>=2.6",
]
```

完整 `[project.optional-dependencies]` 区域应包含：`service`、`exam`、`dev`、`export`、`explorer`、`logging`、`extra`。

- [ ] **Step 2: 验证文件语法**

```bash
python -c "import tomllib; tomllib.loads(open('pyproject.toml').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add pyproject.toml
git commit -m "build: add [exam] extras for paddleocr"
```

---

## Task 2: 创建 Pydantic 模型 exam/schemas.py

**Files:**
- Create: `service/exam/__init__.py`
- Create: `service/exam/schemas.py`
- Create: `tests/exam/__init__.py`
- Create: `tests/exam/test_schemas.py`

- [ ] **Step 1: 写测试**

`tests/exam/test_schemas.py`：

```python
"""Pydantic schema validation tests."""
import pytest
from pydantic import ValidationError

from service.exam.schemas import (
    Question,
    OcrBlock,
    FinalResult,
    TaskSubmitResponse,
    TaskStatusResponse,
)


def test_question_serialization():
    q = Question(
        index=1,
        bbox_xyxy=[127.0, 871.0, 1523.0, 1501.0],
        matched_text="1. (计算题) ...",
        source_detection_ids=[1, 6],
        ocr_score=0.93,
    )
    assert q.index == 1
    assert q.ocr_score == 0.93
    d = q.model_dump()
    assert d["matched_text"] == "1. (计算题) ..."


def test_ocr_block_with_is_question_flag():
    block = OcrBlock(
        source_detection_id=1,
        bbox_xyxy=[127.0, 871.0, 1523.0, 1501.0],
        text="1. 题",
        score=0.93,
        is_question=True,
    )
    assert block.is_question is True


def test_final_result_with_questions_and_blocks():
    final = FinalResult(
        annotated_image="data:image/jpeg;base64,abc",
        original_image="data:image/jpeg;base64,xyz",
        questions=[
            Question(
                index=1, bbox_xyxy=[0, 0, 100, 100],
                matched_text="1.", source_detection_ids=[0], ocr_score=0.9,
            )
        ],
        ocr_blocks=[
            OcrBlock(
                source_detection_id=0, bbox_xyxy=[0, 0, 100, 100],
                text="1.", score=0.9, is_question=True,
            )
        ],
    )
    assert len(final.questions) == 1
    assert len(final.ocr_blocks) == 1


def test_task_submit_response_required_fields():
    r = TaskSubmitResponse(task_id="abc123", status="queued")
    d = r.model_dump()
    assert d["task_id"] == "abc123"
    assert d["status"] == "queued"
    assert d["submit_url"] == "/predict/exam/abc123/status"


def test_task_status_response_running():
    r = TaskStatusResponse(
        task_id="abc", status="running", progress="ocr",
        stages={"yolo": {"duration_ms": 6500, "payload": {}, "error": None}},
        final=None, error=None,
    )
    d = r.model_dump()
    assert d["status"] == "running"
    assert d["stages"]["yolo"]["duration_ms"] == 6500


def test_invalid_bbox_xyxy_wrong_length():
    with pytest.raises(ValidationError):
        Question(
            index=1, bbox_xyxy=[0, 0, 100],
            matched_text="x", source_detection_ids=[], ocr_score=0.0,
        )
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/exam/test_schemas.py -v
```

Expected: ImportError `No module named 'service.exam'`

- [ ] **Step 3: 创建空包文件**

`service/exam/__init__.py`：
```python
"""Exam paper processing pipeline."""
```

`tests/exam/__init__.py`：
```python
"""Exam pipeline tests."""
```

- [ ] **Step 4: 实现 schemas.py**

`service/exam/schemas.py`：

```python
"""Pydantic models for exam pipeline."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Question(BaseModel):
    """Final extended question box."""
    index: int
    bbox_xyxy: list[float] = Field(..., min_length=4, max_length=4)
    matched_text: str
    source_detection_ids: list[int]
    ocr_score: float


class OcrBlock(BaseModel):
    """OCR result for a single cropped plain text box."""
    source_detection_id: int
    bbox_xyxy: list[float] = Field(..., min_length=4, max_length=4)
    text: str
    score: float
    is_question: bool


class FinalResult(BaseModel):
    """Pipeline final output."""
    annotated_image: str = Field(..., description="Data-URL: data:image/jpeg;base64,...")
    original_image: str
    questions: list[Question]
    ocr_blocks: list[OcrBlock]


class StageResult(BaseModel):
    """Per-stage execution record."""
    model_config = ConfigDict(protected_namespaces=())
    started_at: float
    finished_at: float | None = None
    duration_ms: int | None = None
    payload: dict | None = None
    error: str | None = None


class TaskSubmitResponse(BaseModel):
    """Returned immediately after POST /predict/exam."""
    task_id: str
    status: Literal["queued"]
    submit_url: str


class TaskStatusResponse(BaseModel):
    """Returned by GET /predict/exam/{task_id}/status."""
    model_config = ConfigDict(protected_namespaces=())
    task_id: str
    status: Literal["queued", "running", "done", "failed", "expired"]
    progress: str | None = None
    stages: dict[str, StageResult]
    final: FinalResult | None = None
    error: str | None = None

    @classmethod
    def make_submit_url(cls, task_id: str) -> str:
        return f"/predict/exam/{task_id}/status"
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
pytest tests/exam/test_schemas.py -v
```

Expected: 6 passed

- [ ] **Step 6: 提交**

```bash
git add service/exam/__init__.py service/exam/schemas.py tests/exam/__init__.py tests/exam/test_schemas.py
git commit -m "feat(exam): add Pydantic schemas for task lifecycle and results"
```

---

## Task 3: 实现框放大 — 像素模式

**Files:**
- Create: `service/exam/expand.py`
- Create: `tests/exam/test_expand.py`

- [ ] **Step 1: 写测试 — 像素模式**

`tests/exam/test_expand.py`：

```python
"""Box expansion tests (pixel / ratio, 4-direction, clamping)."""
import pytest

from service.exam.expand import expand_boxes


def make_box(x1, y1, x2, y2, cls="plain text"):
    return {
        "id": 0,
        "class_id": 0,
        "class_name": cls,
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_xywh": [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
        "score": 0.9,
    }


def test_pixel_mode_expands_all_four_directions():
    boxes = [make_box(100, 100, 200, 200)]
    out = expand_boxes(
        boxes, mode="pixel",
        top=10, bottom=20, left=5, right=15,
        img_width=1000, img_height=1000,
    )
    assert len(out) == 1
    x1, y1, x2, y2 = out[0]["bbox_xyxy"]
    assert (x1, y1, x2, y2) == (95, 90, 215, 220)


def test_pixel_mode_skips_non_plain_text():
    boxes = [
        make_box(100, 100, 200, 200, cls="plain text"),
        make_box(50, 50, 80, 80, cls="figure"),
    ]
    out = expand_boxes(
        boxes, mode="pixel",
        top=10, bottom=10, left=10, right=10,
        img_width=1000, img_height=1000,
    )
    # figure unchanged
    assert out[1]["bbox_xyxy"] == [50, 50, 80, 80]
    # plain text expanded
    assert out[0]["bbox_xyxy"] == [90, 90, 210, 210]


def test_pixel_mode_clamps_to_image_bounds():
    boxes = [make_box(5, 5, 10, 10)]
    out = expand_boxes(
        boxes, mode="pixel",
        top=100, bottom=100, left=100, right=100,
        img_width=50, img_height=50,
    )
    assert out[0]["bbox_xyxy"] == [0, 0, 50, 50]
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/exam/test_expand.py -v
```

Expected: ImportError `No module named 'service.exam.expand'`

- [ ] **Step 3: 实现 expand.py（仅像素模式 + 占位 ratio）**

`service/exam/expand.py`：

```python
"""Box expansion: pixel or ratio mode, 4-direction independent values."""
from __future__ import annotations

from typing import Literal


def expand_boxes(
    boxes: list[dict],
    *,
    mode: Literal["pixel", "ratio"],
    top: float,
    bottom: float,
    left: float,
    right: float,
    img_width: int,
    img_height: int,
) -> list[dict]:
    """Expand plain text boxes outward; other classes untouched.

    Args:
        boxes: Detection dicts with `class_name` and `bbox_xyxy`.
        mode: "pixel" — values are absolute pixels. "ratio" — values are fractions of box size.
        top/bottom/left/right: Expansion per direction (in chosen mode).
        img_width, img_height: Image dimensions for clamping.

    Returns:
        New list of boxes with expanded `bbox_xyxy`. Only `plain text` class is modified.
    """
    out = []
    for box in boxes:
        if box.get("class_name") != "plain text":
            out.append(box)
            continue
        x1, y1, x2, y2 = box["bbox_xyxy"]
        w, h = x2 - x1, y2 - y1
        if mode == "pixel":
            dx_left, dx_right = left, right
            dy_top, dy_bottom = top, bottom
        elif mode == "ratio":
            dx_left, dx_right = left * w, right * w
            dy_top, dy_bottom = top * h, bottom * h
        else:
            raise ValueError(f"unsupported mode: {mode}")
        nx1 = max(0.0, x1 - dx_left)
        ny1 = max(0.0, y1 - dy_top)
        nx2 = min(float(img_width), x2 + dx_right)
        ny2 = min(float(img_height), y2 + dy_bottom)
        new_box = dict(box)
        new_box["bbox_xyxy"] = [nx1, ny1, nx2, ny2]
        new_box["bbox_xywh"] = [
            (nx1 + nx2) / 2, (ny1 + ny2) / 2,
            nx2 - nx1, ny2 - ny1,
        ]
        out.append(new_box)
    return out
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/exam/test_expand.py -v
```

Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add service/exam/expand.py tests/exam/test_expand.py
git commit -m "feat(exam): add box expansion (pixel + ratio + clamping)"
```

---

## Task 4: 添加比例模式测试

**Files:**
- Modify: `tests/exam/test_expand.py`（追加测试）

- [ ] **Step 1: 追加测试**

```python
def test_ratio_mode_expands_proportional_to_box():
    boxes = [make_box(100, 100, 200, 300)]   # w=100, h=200
    out = expand_boxes(
        boxes, mode="ratio",
        top=0.1, bottom=0.1, left=0.2, right=0.2,
        img_width=1000, img_height=1000,
    )
    # top/bottom = 0.1 * 200 = 20 each
    # left/right = 0.2 * 100 = 20 each
    assert out[0]["bbox_xyxy"] == [80, 80, 220, 320]


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        expand_boxes(
            [make_box(0, 0, 10, 10)],
            mode="bad", top=0, bottom=0, left=0, right=0,
            img_width=10, img_height=10,
        )


def test_empty_input_returns_empty():
    out = expand_boxes(
        [], mode="pixel", top=10, bottom=10, left=10, right=10,
        img_width=100, img_height=100,
    )
    assert out == []
```

- [ ] **Step 2: 运行测试**

```bash
pytest tests/exam/test_expand.py -v
```

Expected: 6 passed（之前 3 个 + 新增 3 个）

- [ ] **Step 3: 提交**

```bash
git add tests/exam/test_expand.py
git commit -m "test(exam): add ratio mode and edge case tests for expand"
```

---

## Task 5: 实现题号正则判定

**Files:**
- Create: `service/exam/question_filter.py`
- Create: `tests/exam/test_question_filter.py`

- [ ] **Step 1: 写测试**

`tests/exam/test_question_filter.py`：

```python
"""Question number regex matching tests."""
import pytest

from service.exam.question_filter import (
    DEFAULT_QUESTION_REGEX,
    is_question,
    validate_regex,
)


def test_default_regex_matches_dotted():
    assert is_question("1. (计算题) 1+1=?", DEFAULT_QUESTION_REGEX)


def test_default_regex_matches_paren():
    assert is_question("(2) 已知 x = 1", DEFAULT_QUESTION_REGEX)


def test_default_regex_matches_two_digit():
    assert is_question("10. 阅读理解", DEFAULT_QUESTION_REGEX)


def test_default_regex_matches_with_leading_whitespace():
    assert is_question("  3. 选择题", DEFAULT_QUESTION_REGEX)


def test_default_regex_rejects_subnumber():
    assert not is_question("1.1 子项", DEFAULT_QUESTION_REGEX)


def test_default_regex_rejects_plain_text():
    assert not is_question("本题共 10 分", DEFAULT_QUESTION_REGEX)


def test_default_regex_rejects_empty():
    assert not is_question("", DEFAULT_QUESTION_REGEX)


def test_custom_regex():
    assert is_question("Question 3: ...", r"^Question\s+\d+")


def test_validate_regex_accepts_valid():
    assert validate_regex(r"^\d+\.") == r"^\d+\."


def test_validate_regex_rejects_invalid():
    with pytest.raises(ValueError):
        validate_regex(r"[")


def test_validate_regex_rejects_matches_empty():
    with pytest.raises(ValueError):
        validate_regex(r".*")
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/exam/test_question_filter.py -v
```

Expected: ImportError

- [ ] **Step 3: 实现 question_filter.py**

`service/exam/question_filter.py`：

```python
"""Question number detection via regex."""
from __future__ import annotations

import re

DEFAULT_QUESTION_REGEX = r"^\s*\(?\d+[\.\)]"


def validate_regex(pattern: str) -> str:
    """Validate a Python regex. Raises ValueError if invalid or matches empty."""
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"invalid regex: {e}") from e
    if compiled.match(""):
        raise ValueError("regex matches empty string; would trivially match all boxes")
    return pattern


def is_question(text: str, regex: str = DEFAULT_QUESTION_REGEX) -> bool:
    """Return True if text starts with a question number per the regex."""
    if not text:
        return False
    return re.match(regex, text) is not None
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/exam/test_question_filter.py -v
```

Expected: 11 passed

- [ ] **Step 5: 提交**

```bash
git add service/exam/question_filter.py tests/exam/test_question_filter.py
git commit -m "feat(exam): add question number regex filter with validation"
```

---

## Task 6: 实现坐标延伸几何逻辑

**Files:**
- Create: `service/exam/geometry.py`
- Create: `tests/exam/test_geometry.py`

- [ ] **Step 1: 写测试**

`tests/exam/test_geometry.py`：

```python
"""Question box coordinate extension tests."""
from service.exam.geometry import extend_questions


def box(x1, y1, x2, y2, cls="plain text"):
    return {"class_name": cls, "bbox_xyxy": [x1, y1, x2, y2]}


def question(text, x1, y1, x2, y2, source_ids=(0,)):
    return {"text": text, "bbox_xyxy": [x1, y1, x2, y2], "source_ids": list(source_ids)}


def test_three_questions_extend_to_next_top():
    """Middle question's bottom = next question's top."""
    all_boxes = [
        box(0, 0, 100, 50, cls="figure"),       # 0: top
        box(0, 100, 100, 200),                  # 1: question 1 (plain text)
        box(0, 250, 100, 320),                  # 2: question 2
        box(0, 400, 100, 500),                  # 3: question 3
        box(0, 600, 100, 800, cls="table"),     # 4: tail plain text max-y2
    ]
    qs = [
        question("1.", 0, 100, 100, 180, source_ids=[1]),
        question("2.", 0, 250, 100, 290, source_ids=[2]),
        question("3.", 0, 400, 100, 480, source_ids=[3]),
    ]
    out = extend_questions(qs, all_boxes)
    # Q1 bottom = Q2 top = 250
    assert out[0]["bbox_xyxy"] == [0, 100, 100, 250]
    # Q2 bottom = Q3 top = 400
    assert out[1]["bbox_xyxy"] == [0, 250, 100, 400]
    # Q3 bottom = max(y2) plain text = 800 (table class)
    # Wait — only plain text class is considered for "last box" per spec
    # Re-run with table also plain text
    all_boxes[4] = box(0, 600, 100, 800, cls="plain text")
    out = extend_questions(qs, all_boxes)
    assert out[2]["bbox_xyxy"] == [0, 400, 100, 800]


def test_last_question_uses_max_y2_plain_text_not_other_classes():
    all_boxes = [
        box(0, 100, 100, 200),                                  # q1
        box(0, 250, 100, 320),                                  # q2
        box(0, 400, 100, 500),                                  # q3
        box(0, 600, 100, 1500, cls="figure"),                   # 极高 figure (max y2)
        box(0, 550, 100, 700),                                  # plain text (NOT max y2)
    ]
    qs = [
        question("1.", 0, 100, 100, 180, source_ids=[1]),
        question("2.", 0, 250, 100, 290, source_ids=[2]),
        question("3.", 0, 400, 100, 480, source_ids=[3]),
    ]
    out = extend_questions(qs, all_boxes)
    # Last question bottom should be max y2 plain text = 700, not 1500
    assert out[2]["bbox_xyxy"] == [0, 400, 100, 700]


def test_single_question_uses_max_y2_plain_text():
    all_boxes = [
        box(0, 100, 100, 200),
        box(0, 500, 100, 900, cls="plain text"),
    ]
    qs = [question("1.", 0, 100, 100, 180, source_ids=[0])]
    out = extend_questions(qs, all_boxes)
    assert out[0]["bbox_xyxy"] == [0, 100, 100, 900]


def test_zero_questions_returns_empty():
    assert extend_questions([], []) == []


def test_input_order_unordered_questions_sorted_by_y1():
    all_boxes = [
        box(0, 100, 100, 200),
        box(0, 250, 100, 320),
    ]
    qs = [
        question("2.", 0, 250, 100, 290, source_ids=[1]),
        question("1.", 0, 100, 100, 180, source_ids=[0]),
    ]
    out = extend_questions(qs, all_boxes)
    # After sort by y1: Q1 (y=100), Q2 (y=250). Q1 bottom = 250.
    assert out[0]["bbox_xyxy"][1] == 100
    assert out[0]["bbox_xyxy"][3] == 250
    assert out[1]["bbox_xyxy"][1] == 250
    # last question has no "next", uses max plain text y2 = 320
    assert out[1]["bbox_xyxy"][3] == 320


def test_no_plain_text_returns_questions_unchanged_for_last():
    """When no plain text exists beyond last question, keep its original bottom."""
    all_boxes = [
        box(0, 100, 100, 200, cls="figure"),
        box(0, 250, 100, 320, cls="table"),
    ]
    qs = [question("1.", 0, 100, 100, 180, source_ids=[0])]
    out = extend_questions(qs, all_boxes)
    # No plain text at all; last question's bottom = original 180
    assert out[0]["bbox_xyxy"] == [0, 100, 100, 180]
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/exam/test_geometry.py -v
```

Expected: ImportError

- [ ] **Step 3: 实现 geometry.py**

`service/exam/geometry.py`：

```python
"""Question box coordinate extension logic."""
from __future__ import annotations


def extend_questions(
    questions: list[dict],
    all_boxes: list[dict],
) -> list[dict]:
    """Extend each question's bottom to the next question's top.

    The LAST question's bottom = max(y2) of all plain-text-class boxes,
    per spec section 1.3 ("最后框 = 所有框中 y2 最大且类别为 plain text").

    If no plain-text box exists, the last question keeps its original bottom.

    Args:
        questions: Each dict has keys `bbox_xyxy: [x1, y1, x2, y2]`, `text`, `source_ids`.
        all_boxes: All original detections, each with `class_name` and `bbox_xyxy`.

    Returns:
        New list of questions with extended bbox_xyxy. Original `text` and `source_ids` preserved.
    """
    if not questions:
        return []

    sorted_qs = sorted(questions, key=lambda q: q["bbox_xyxy"][1])

    # Compute max y2 of plain text class for last question's bottom.
    plain_text_y2s = [
        b["bbox_xyxy"][3] for b in all_boxes
        if b.get("class_name") == "plain text"
    ]
    last_bottom = max(plain_text_y2s) if plain_text_y2s else None

    out = []
    for i, q in enumerate(sorted_qs):
        x1, y1, x2, y2 = q["bbox_xyxy"]
        new_box = dict(q)
        if i + 1 < len(sorted_qs):
            next_top = sorted_qs[i + 1]["bbox_xyxy"][1]
            new_box["bbox_xyxy"] = [x1, y1, x2, next_top]
        elif last_bottom is not None:
            new_box["bbox_xyxy"] = [x1, y1, x2, last_bottom]
        else:
            # Keep original
            new_box["bbox_xyxy"] = [x1, y1, x2, y2]
        out.append(new_box)
    return out
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/exam/test_geometry.py -v
```

Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add service/exam/geometry.py tests/exam/test_geometry.py
git commit -m "feat(exam): add question box coordinate extension logic"
```

---

## Task 7: 实现内存任务存储 — 基础 CRUD

**Files:**
- Create: `service/exam/tasks.py`
- Create: `tests/exam/test_tasks.py`

- [ ] **Step 1: 写测试（基础部分）**

`tests/exam/test_tasks.py`：

```python
"""Task store tests: CRUD, stage updates, LRU eviction, TTL expiration."""
import time

import pytest

from service.exam import tasks
from service.exam.schemas import FinalResult, Question, OcrBlock


@pytest.fixture(autouse=True)
def reset_store():
    """Clear the in-memory store between tests."""
    tasks._store.clear()
    yield
    tasks._store.clear()


def test_create_task_returns_id_and_queued_state():
    tid = tasks.create_task()
    assert isinstance(tid, str) and len(tid) > 0
    state = tasks.get_task(tid)
    assert state is not None
    assert state["status"] == "queued"
    assert state["stages"] == {}
    assert state["final"] is None


def test_get_missing_task_returns_none():
    assert tasks.get_task("nonexistent") is None


def test_update_stage_records_duration():
    tid = tasks.create_task()
    tasks.update_stage(tid, "yolo", duration_ms=1234, payload={"num_detections": 5})
    state = tasks.get_task(tid)
    assert state["status"] == "running"
    assert "yolo" in state["stages"]
    assert state["stages"]["yolo"]["duration_ms"] == 1234
    assert state["stages"]["yolo"]["payload"] == {"num_detections": 5}


def test_update_stage_with_error_does_not_fail_task():
    tid = tasks.create_task()
    tasks.update_stage(tid, "ocr", duration_ms=500, error="PaddleOCR failed")
    state = tasks.get_task(tid)
    # Spec: intermediate errors don't fail the task
    assert state["status"] == "running"
    assert state["stages"]["ocr"]["error"] == "PaddleOCR failed"


def test_mark_done_writes_final_and_status_done():
    tid = tasks.create_task()
    final = FinalResult(
        annotated_image="data:img",
        original_image="data:orig",
        questions=[],
        ocr_blocks=[],
    )
    tasks.mark_done(tid, final)
    state = tasks.get_task(tid)
    assert state["status"] == "done"
    assert state["final"] == final


def test_mark_failed_writes_error_and_status_failed():
    tid = tasks.create_task()
    tasks.mark_failed(tid, "boom")
    state = tasks.get_task(tid)
    assert state["status"] == "failed"
    assert state["error"] == "boom"


def test_get_task_serializable_dict():
    tid = tasks.create_task()
    tasks.update_stage(tid, "yolo", duration_ms=100, payload={"k": "v"})
    state = tasks.get_task(tid)
    # Ensure no internal references / Pydantic models in serializable form
    assert isinstance(state, dict)
    assert isinstance(state["stages"], dict)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/exam/test_tasks.py -v
```

Expected: ImportError

- [ ] **Step 3: 实现 tasks.py（基础部分 + LRU/TTL 占位）**

`service/exam/tasks.py`：

```python
"""In-memory task store with LRU eviction and TTL expiration."""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from typing import Any

from service.exam.schemas import FinalResult

# Tunables
MAX_TASKS = 64
TASK_TTL_SECONDS = 600  # 10 minutes

# OrderedDict preserves insertion order; we move-to-end on access for LRU semantics.
_store: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _now() -> float:
    return time.time()


def _purge_expired() -> None:
    """Remove tasks older than TASK_TTL_SECONDS."""
    cutoff = _now() - TASK_TTL_SECONDS
    expired = [tid for tid, s in _store.items() if s["updated_at"] < cutoff]
    for tid in expired:
        _store.pop(tid, None)


def create_task() -> str:
    """Create a new task and return its ID."""
    _purge_expired()
    tid = uuid.uuid4().hex[:12]
    now = _now()
    _store[tid] = {
        "task_id": tid,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "stages": {},
        "final": None,
        "error": None,
        "progress": None,
    }
    _store.move_to_end(tid)
    _evict_if_needed()
    return tid


def get_task(task_id: str) -> dict[str, Any] | None:
    """Return task state as dict, or None if missing/expired."""
    _purge_expired()
    state = _store.get(task_id)
    if state is None:
        return None
    if _now() - state["updated_at"] > TASK_TTL_SECONDS:
        _store.pop(task_id, None)
        return None
    _store.move_to_end(task_id)
    return dict(state)


def update_stage(
    task_id: str,
    stage: str,
    *,
    duration_ms: int,
    payload: dict | None = None,
    error: str | None = None,
) -> None:
    """Record a stage's completion. Intermediate errors don't fail the task."""
    state = _store.get(task_id)
    if state is None:
        return
    now = _now()
    state["stages"][stage] = {
        "duration_ms": duration_ms,
        "payload": payload,
        "error": error,
    }
    state["status"] = "running"
    state["progress"] = stage
    state["updated_at"] = now
    _store.move_to_end(task_id)
    _evict_if_needed()


def mark_done(task_id: str, final: FinalResult) -> None:
    """Mark task as done with final result."""
    state = _store.get(task_id)
    if state is None:
        return
    state["final"] = final
    state["status"] = "done"
    state["progress"] = None
    state["updated_at"] = _now()
    _store.move_to_end(task_id)


def mark_failed(task_id: str, error: str) -> None:
    """Mark task as failed with error message."""
    state = _store.get(task_id)
    if state is None:
        return
    state["error"] = error
    state["status"] = "failed"
    state["progress"] = None
    state["updated_at"] = _now()
    _store.move_to_end(task_id)


def _evict_if_needed() -> None:
    """LRU eviction: drop oldest entries if over MAX_TASKS."""
    while len(_store) > MAX_TASKS:
        _store.popitem(last=False)


def reset_for_tests() -> None:
    """Clear all tasks (test helper)."""
    _store.clear()
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/exam/test_tasks.py -v
```

Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add service/exam/tasks.py tests/exam/test_tasks.py
git commit -m "feat(exam): add in-memory task store with LRU + TTL"
```

---

## Task 8: 添加 LRU + TTL 测试

**Files:**
- Modify: `tests/exam/test_tasks.py`（追加）

- [ ] **Step 1: 追加测试**

```python
def test_lru_eviction_when_over_max(monkeypatch):
    monkeypatch.setattr(tasks, "MAX_TASKS", 3)
    tids = [tasks.create_task() for _ in range(4)]
    # The oldest (tids[0]) should be evicted
    assert tasks.get_task(tids[0]) is None
    for tid in tids[1:]:
        assert tasks.get_task(tid) is not None


def test_lru_access_promotes_to_most_recent(monkeypatch):
    monkeypatch.setattr(tasks, "MAX_TASKS", 3)
    t1 = tasks.create_task()
    t2 = tasks.create_task()
    t3 = tasks.create_task()
    # Access t1 to promote it
    tasks.get_task(t1)
    # Adding t4 should evict t2 (now the oldest)
    t4 = tasks.create_task()
    assert tasks.get_task(t1) is not None
    assert tasks.get_task(t2) is None
    assert tasks.get_task(t3) is not None
    assert tasks.get_task(t4) is not None


def test_ttl_expiration(monkeypatch):
    monkeypatch.setattr(tasks, "TASK_TTL_SECONDS", 0.1)
    tid = tasks.create_task()
    assert tasks.get_task(tid) is not None
    time.sleep(0.2)
    assert tasks.get_task(tid) is None
```

- [ ] **Step 2: 运行测试**

```bash
pytest tests/exam/test_tasks.py -v
```

Expected: 10 passed

- [ ] **Step 3: 提交**

```bash
git add tests/exam/test_tasks.py
git commit -m "test(exam): add LRU and TTL expiration tests"
```

---

## Task 9: 实现 OCR 引擎封装

**Files:**
- Create: `service/exam/ocr.py`
- Create: `tests/exam/test_ocr.py`

- [ ] **Step 1: 写测试（mock PaddleOCR）**

`tests/exam/test_ocr.py`：

```python
"""OCR engine tests with mocked PaddleOCR (no model download)."""
import numpy as np
import pytest

from service.exam.ocr import OcrEngine, TextBlock


def test_text_block_dataclass():
    block = TextBlock(text="hello", bbox=[0, 0, 100, 50], score=0.95)
    assert block.text == "hello"
    assert block.score == 0.95


def test_ocr_engine_lazy_load(monkeypatch):
    """OcrEngine should not call PaddleOCR at construction."""
    loaded = {"called": False}

    def fake_load():
        loaded["called"] = True
        return "fake-engine"

    monkeypatch.setattr(
        "service.exam.ocr._load_paddleocr", fake_load
    )
    engine = OcrEngine(lang="ch")
    assert loaded["called"] is False


def test_ocr_engine_loads_on_first_use(monkeypatch):
    loaded = {"called": False}

    def fake_load():
        loaded["called"] = True
        return "fake-engine"

    monkeypatch.setattr(
        "service.exam.ocr._load_paddleocr", fake_load
    )
    monkeypatch.setattr(
        "service.exam.ocr._run_ocr",
        lambda engine, img: [TextBlock(text="1.", bbox=[0, 0, 10, 10], score=0.9)],
    )
    engine = OcrEngine(lang="ch")
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    result = engine.ocr_image(img)
    assert loaded["called"] is True
    assert len(result) == 1
    assert result[0].text == "1."


def test_ocr_engine_caches_engine(monkeypatch):
    """PaddleOCR should only load once per OcrEngine instance."""
    call_count = {"n": 0}

    def fake_load():
        call_count["n"] += 1
        return "engine"

    monkeypatch.setattr("service.exam.ocr._load_paddleocr", fake_load)
    monkeypatch.setattr(
        "service.exam.ocr._run_ocr",
        lambda e, img: [],
    )
    engine = OcrEngine(lang="ch")
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    engine.ocr_image(img)
    engine.ocr_image(img)
    assert call_count["n"] == 1


def test_ocr_engine_handles_load_failure(monkeypatch):
    def fake_load():
        raise RuntimeError("model not found")
    monkeypatch.setattr("service.exam.ocr._load_paddleocr", fake_load)
    engine = OcrEngine(lang="ch")
    with pytest.raises(RuntimeError):
        engine.ocr_image(np.zeros((10, 10, 3), dtype=np.uint8))
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/exam/test_ocr.py -v
```

Expected: ImportError

- [ ] **Step 3: 实现 ocr.py**

`service/exam/ocr.py`：

```python
"""PaddleOCR engine wrapper with lazy loading."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TextBlock:
    """Single OCR result block."""
    text: str
    bbox: list[float]   # [x1, y1, x2, y2]
    score: float


def _load_paddleocr(lang: str = "ch"):
    """Load PaddleOCR (lazy import to avoid heavy startup cost)."""
    from paddleocr import PaddleOCR  # type: ignore
    return PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)


def _run_ocr(engine, img: np.ndarray) -> list[TextBlock]:
    """Run PaddleOCR and convert results to TextBlock list."""
    raw = engine.ocr(img, cls=False)
    blocks: list[TextBlock] = []
    if not raw or not raw[0]:
        return blocks
    for line in raw[0]:
        # line = [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, score)]
        box_pts = line[0]
        text_score = line[1]
        text = text_score[0]
        score = float(text_score[1])
        xs = [p[0] for p in box_pts]
        ys = [p[1] for p in box_pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        blocks.append(TextBlock(text=text, bbox=[x1, y1, x2, y2], score=score))
    return blocks


class OcrEngine:
    """Lazy-loaded PaddleOCR engine.

    PaddleOCR models are not loaded at construction time. The first call to
    `ocr_image` triggers loading. Subsequent calls reuse the loaded engine.
    """

    def __init__(self, lang: str = "ch") -> None:
        self._lang = lang
        self._engine = None

    def ocr_image(self, img: np.ndarray) -> list[TextBlock]:
        if self._engine is None:
            self._engine = _load_paddleocr(self._lang)
        return _run_ocr(self._engine, img)
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/exam/test_ocr.py -v
```

Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add service/exam/ocr.py tests/exam/test_ocr.py
git commit -m "feat(exam): add lazy-loaded PaddleOCR engine wrapper"
```

---

## Task 10: 实现流水线 — 主流程编排

**Files:**
- Create: `service/exam/pipeline.py`
- Create: `tests/exam/test_pipeline.py`

- [ ] **Step 1: 写测试（mock 全部外部依赖）**

`tests/exam/test_pipeline.py`：

```python
"""Pipeline orchestration tests with mocked YOLO + OCR."""
import numpy as np
import pytest

from service.exam import pipeline, tasks
from service.exam.ocr import TextBlock


@pytest.fixture(autouse=True)
def reset_store():
    tasks.reset_for_tests()
    yield
    tasks.reset_for_tests()


def fake_yolo_result(boxes):
    """Build a fake YOLO Results-like object with .boxes and .names."""
    class FakeBoxes:
        def __init__(self, boxes):
            self.xyxy = np.array([b["xyxy"] for b in boxes], dtype=float)
            self.conf = np.array([b["conf"] for b in boxes], dtype=float)
            self.cls = np.array([b["cls"] for b in boxes], dtype=int)
    class FakeResult:
        def __init__(self, boxes):
            self.boxes = FakeBoxes(boxes) if boxes else None
            self.names = {0: "plain text", 1: "figure"}
        def plot(self, **kwargs):
            # Return a small image with shape (H, W, 3) RGB
            return np.zeros((100, 100, 3), dtype=np.uint8)
    return FakeResult(boxes)


def test_pipeline_runs_through_all_stages(monkeypatch):
    # Mock YOLO predict_one
    async def fake_predict_one(**kwargs):
        return {
            "width": 800, "height": 1000,
            "annotated_base64": "data:image/jpeg;base64,fake",
            "detections": [
                {"id": 0, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [50, 100, 750, 200], "bbox_xywh": [400, 150, 700, 100],
                 "score": 0.95},
                {"id": 1, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [50, 300, 750, 400], "bbox_xywh": [400, 350, 700, 100],
                 "score": 0.90},
                {"id": 2, "class_id": 0, "class_name": "plain text",
                 "bbox_xyxy": [50, 500, 750, 950], "bbox_xywh": [400, 725, 700, 450],
                 "score": 0.85},
            ],
            "num_detections": 3,
            "inference_time_ms": 5000,
            "model_imgsz": 1024,
            "conf_threshold": 0.3,
            "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)

    # Mock OCR engine
    class FakeOcr:
        def __init__(self):
            self.calls = 0
        def ocr_image(self, img):
            self.calls += 1
            # First call → "1.", Second → "2.", Third → "3."
            texts = ["1. (计算题) 1+1=?", "2. 阅读理解...", "3. 作文"]
            idx = self.calls - 1
            return [TextBlock(text=texts[idx], bbox=[0, 0, 100, 50], score=0.9)]
    fake_ocr = FakeOcr()
    monkeypatch.setattr(pipeline, "OcrEngine", lambda lang="ch": fake_ocr)

    # Mock image redraw (skip actual rendering)
    def fake_redraw(img_bgr, questions, original_detections):
        return "data:image/jpeg;base64,redrawn"
    monkeypatch.setattr(pipeline, "redraw_with_questions", fake_redraw)

    # Mock decode
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((1000, 800, 3), dtype=np.uint8))

    tid = tasks.create_task()
    params = {
        "expand_mode": "pixel",
        "expand_top": 10, "expand_bottom": 10,
        "expand_left": 10, "expand_right": 10,
        "question_regex": r"^\s*\(?\d+[\.\)]",
        "conf": 0.3, "imgsz": 1024,
    }
    pipeline.run_pipeline(tid, b"fake-image-bytes", params)

    state = tasks.get_task(tid)
    assert state["status"] == "done"
    assert state["error"] is None
    # All 5 stages recorded
    for stage in ("yolo", "expand", "ocr", "filter", "geometry"):
        assert stage in state["stages"], f"stage {stage} missing"
        assert state["stages"][stage]["error"] is None
    # 3 questions found
    assert len(state["final"].questions) == 3
    # Q1 bottom = Q2 top after extension
    q1 = state["final"].questions[0]
    q2 = state["final"].questions[1]
    # After expand: y1 ~ 90, y2 ~ 210 (per pixel expand of [100, 200])
    # Q1 extended bottom = Q2 top ~ 290
    assert q1.bbox_xyxy[3] == pytest.approx(q2.bbox_xyxy[1], abs=2)


def test_pipeline_marks_failed_on_fatal_yolo_error(monkeypatch):
    async def fake_predict_one(**kwargs):
        raise RuntimeError("YOLO crashed")
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((100, 100, 3), dtype=np.uint8))

    tid = tasks.create_task()
    pipeline.run_pipeline(tid, b"x", {})
    state = tasks.get_task(tid)
    assert state["status"] == "failed"
    assert "YOLO" in state["error"]


def test_pipeline_zero_questions_still_completes(monkeypatch):
    async def fake_predict_one(**kwargs):
        return {
            "width": 800, "height": 1000,
            "annotated_base64": "data:img",
            "detections": [],
            "num_detections": 0,
            "inference_time_ms": 100,
            "model_imgsz": 1024, "conf_threshold": 0.3, "device": "cpu",
        }
    monkeypatch.setattr("service.inference.predict_one", fake_predict_one)
    monkeypatch.setattr(pipeline, "_decode_image_bytes",
                        lambda b: np.zeros((100, 100, 3), dtype=np.uint8))

    tid = tasks.create_task()
    pipeline.run_pipeline(tid, b"x", {})
    state = tasks.get_task(tid)
    assert state["status"] == "done"
    assert state["final"].questions == []
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/exam/test_pipeline.py -v
```

Expected: ImportError on `service.exam.pipeline`

- [ ] **Step 3: 实现 pipeline.py**

`service/exam/pipeline.py`：

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
from service.exam.question_filter import is_question
from service.exam.schemas import FinalResult, OcrBlock, Question


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image bytes")
    return img


def _encode_jpeg_base64(rgb_or_bgr: np.ndarray) -> str:
    """Encode numpy image as data-URL base64 JPEG. Accepts RGB or BGR."""
    if rgb_or_bgr.ndim == 3 and rgb_or_bgr.shape[2] == 3:
        # If colors look BGR (heuristic: high blue variance), swap.
        # For simplicity, assume BGR input and convert.
        rgb = cv2.cvtColor(rgb_or_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = rgb_or_bgr
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else bgr


def redraw_with_questions(
    img_bgr: np.ndarray,
    questions: list[dict],
    original_detections: list[dict],
) -> str:
    """Draw extended question boxes on original image, return base64 jpeg data-URL."""
    rgb = _to_rgb(img_bgr.copy())
    pil = Image.fromarray(rgb)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil)
    for q in questions:
        x1, y1, x2, y2 = q["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=5)
    return _encode_jpeg_base64(np.asarray(pil))


def run_pipeline(task_id: str, image_bytes: bytes, params: dict[str, Any]) -> None:
    """Run the 5-stage pipeline for a task. Updates task state in-place."""
    import service.inference as inference

    # Stage 0: decode
    try:
        img_bgr = _decode_image_bytes(image_bytes)
        h, w = img_bgr.shape[:2]
    except Exception as e:
        tasks.mark_failed(task_id, f"decode failed: {e}")
        return

    # Stage 1: YOLO
    t0 = time.perf_counter()
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(inference.predict_one(
                image_bytes=image_bytes,
                model=None,           # placeholder; predict_one in tests uses fake
                semaphore=__import__("asyncio").Semaphore(1),
                conf=params.get("conf", 0.3),
                imgsz=params.get("imgsz", 1024),
                line_width=5,
                font_size=20,
                device=params.get("device", "cpu"),
            ))
        finally:
            loop.close()
        detections = result["detections"]
        tasks.update_stage(task_id, "yolo",
                           duration_ms=int((time.perf_counter() - t0) * 1000),
                           payload={"num_detections": len(detections)})
    except Exception as e:
        tasks.update_stage(task_id, "yolo", duration_ms=0, error=str(e))
        tasks.mark_failed(task_id, f"yolo failed: {e}")
        return

    # Stage 2: expand
    t0 = time.perf_counter()
    try:
        expanded = expand_boxes(
            detections,
            mode=params.get("expand_mode", "pixel"),
            top=params.get("expand_top", 20),
            bottom=params.get("expand_bottom", 20),
            left=params.get("expand_left", 20),
            right=params.get("expand_right", 20),
            img_width=w, img_height=h,
        )
        plain_expanded = [b for b in expanded if b["class_name"] == "plain text"]
        tasks.update_stage(task_id, "expand",
                           duration_ms=int((time.perf_counter() - t0) * 1000),
                           payload={"num_expanded": len(plain_expanded)})
    except Exception as e:
        tasks.update_stage(task_id, "expand", duration_ms=0, error=str(e))
        plain_expanded = []

    # Stage 3: OCR
    t0 = time.perf_counter()
    ocr_blocks: list[OcrBlock] = []
    try:
        ocr_engine = OcrEngine(lang="ch")
        for box in plain_expanded:
            x1, y1, x2, y2 = [int(round(v)) for v in box["bbox_xyxy"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                ocr_blocks.append(OcrBlock(
                    source_detection_id=box["id"],
                    bbox_xyxy=box["bbox_xyxy"], text="", score=0.0,
                    is_question=False,
                ))
                continue
            crop = img_bgr[y1:y2, x1:x2]
            try:
                blocks: list[TextBlock] = ocr_engine.ocr_image(crop)
            except Exception as e:
                tasks.update_stage(task_id, "ocr", duration_ms=0,
                                   error=f"per-box OCR failed: {e}")
                blocks = []
            full_text = " ".join(b.text for b in blocks) if blocks else ""
            avg_score = sum(b.score for b in blocks) / len(blocks) if blocks else 0.0
            ocr_blocks.append(OcrBlock(
                source_detection_id=box["id"],
                bbox_xyxy=box["bbox_xyxy"],
                text=full_text, score=avg_score,
                is_question=False,  # set in filter stage
            ))
        tasks.update_stage(task_id, "ocr",
                           duration_ms=int((time.perf_counter() - t0) * 1000),
                           payload={"num_blocks": len(ocr_blocks)})
    except Exception as e:
        tasks.update_stage(task_id, "ocr", duration_ms=0, error=str(e))
        ocr_blocks = []

    # Stage 4: filter
    t0 = time.perf_counter()
    regex = params.get("question_regex", r"^\s*\(?\d+[\.\)]")
    try:
        questions: list[dict] = []
        for block in ocr_blocks:
            matched = is_question(block.text, regex)
            block.is_question = matched
            if matched:
                src_box = next(
                    (b for b in detections if b["id"] == block.source_detection_id),
                    None,
                )
                questions.append({
                    "text": block.text,
                    "bbox_xyxy": block.bbox_xyxy,
                    "source_ids": [block.source_detection_id],
                    "ocr_score": block.score,
                })
        tasks.update_stage(task_id, "filter",
                           duration_ms=int((time.perf_counter() - t0) * 1000),
                           payload={"num_questions": len(questions)})
    except Exception as e:
        tasks.update_stage(task_id, "filter", duration_ms=0, error=str(e))
        questions = []

    # Stage 5: geometry
    t0 = time.perf_counter()
    try:
        extended = extend_questions(questions, detections)
        tasks.update_stage(task_id, "geometry",
                           duration_ms=int((time.perf_counter() - t0) * 1000),
                           payload={"num_extended": len(extended)})
    except Exception as e:
        tasks.update_stage(task_id, "geometry", duration_ms=0, error=str(e))
        extended = questions

    # Stage 6: redraw + final
    try:
        annotated_b64 = redraw_with_questions(img_bgr, extended, detections)
        original_b64 = _encode_jpeg_base64(img_bgr)
        question_models = [
            Question(
                index=i + 1,
                bbox_xyxy=q["bbox_xyxy"],
                matched_text=q["text"],
                source_detection_ids=q["source_ids"],
                ocr_score=q["ocr_score"],
            )
            for i, q in enumerate(extended)
        ]
        final = FinalResult(
            annotated_image=annotated_b64,
            original_image=original_b64,
            questions=question_models,
            ocr_blocks=ocr_blocks,
        )
        tasks.mark_done(task_id, final)
    except Exception as e:
        tasks.update_stage(task_id, "geometry", duration_ms=0,
                           error=f"redraw failed: {e}")
        tasks.mark_failed(task_id, f"final assembly failed: {e}")
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/exam/test_pipeline.py -v
```

Expected: 3 passed

> 注意：第一次测试中 YOLO 调用使用 fake，`predict_one` 内部使用 `model=None` 会被 monkeypatch 拦截；如有问题可调整 `predict_one` 入口以接受预注入的 model 引用，或用 monkeypatch 直接拦截 `_run_model_predict`。

- [ ] **Step 5: 提交**

```bash
git add service/exam/pipeline.py tests/exam/test_pipeline.py
git commit -m "feat(exam): add 5-stage pipeline orchestrator with error capture"
```

---

## Task 11: 重构 pipeline 以正确集成 YOLO 模型

**Files:**
- Modify: `service/exam/pipeline.py`
- Modify: `tests/exam/test_pipeline.py`

观察：上一个 Task 中 pipeline 直接调用 `predict_one` 需要真实 model。在生产路径中，模型从 `service.app.state` 获取。需重构为可注入。

- [ ] **Step 1: 修改 pipeline 签名**

`service/exam/pipeline.py` 中 `run_pipeline` 改为：

```python
def run_pipeline(
    task_id: str,
    image_bytes: bytes,
    params: dict[str, Any],
    *,
    model,                       # injected model (YOLOv10 instance)
    semaphore,                   # asyncio.Semaphore
) -> None:
    ...
    # Stage 1 内部：
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(inference.predict_one(
            image_bytes=image_bytes,
            model=model,
            semaphore=semaphore,
            conf=params.get("conf", 0.3),
            imgsz=params.get("imgsz", 1024),
            line_width=5,
            font_size=20,
            device=params.get("device", "cpu"),
        ))
    finally:
        loop.close()
```

- [ ] **Step 2: 更新测试**

`tests/exam/test_pipeline.py` 中每个调用 `pipeline.run_pipeline` 的位置追加 `model=None, semaphore=__import__("asyncio").Semaphore(1)`：

```python
pipeline.run_pipeline(tid, b"x", params, model=None, semaphore=asyncio.Semaphore(1))
```

- [ ] **Step 3: 重新运行测试**

```bash
pytest tests/exam/test_pipeline.py -v
```

Expected: 3 passed

- [ ] **Step 4: 提交**

```bash
git add service/exam/pipeline.py tests/exam/test_pipeline.py
git commit -m "refactor(exam): inject model and semaphore into pipeline"
```

---

## Task 12: 路由 — POST /predict/exam

**Files:**
- Modify: `service/app.py`（追加路由）
- Create: `tests/exam/test_app_exam.py`

- [ ] **Step 1: 写测试**

`tests/exam/test_app_exam.py`：

```python
"""HTTP tests for /predict/exam endpoints."""
import io

import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from service.exam import tasks


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_tasks():
    tasks.reset_for_tests()
    yield
    tasks.reset_for_tests()


def _make_image(filename="test.png"):
    # Minimal valid PNG bytes (1x1 red pixel)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x5c\xcd\xff\x69\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return (filename, png_bytes, "image/png")


def test_submit_returns_task_id_and_queued_status(client):
    r = client.post("/predict/exam", files={"file": _make_image()})
    assert r.status_code == 200
    data = r.json()
    assert "task_id" in data
    assert data["status"] == "queued"
    assert data["submit_url"].startswith("/predict/exam/")


def test_status_returns_404_for_unknown_task(client):
    r = client.get("/predict/exam/nonexistent/status")
    assert r.status_code == 404


def test_status_returns_200_for_known_task(client):
    r = client.post("/predict/exam", files={"file": _make_image()})
    tid = r.json()["task_id"]
    r2 = client.get(f"/predict/exam/{tid}/status")
    assert r2.status_code == 200
    body = r2.json()
    assert body["task_id"] == tid
    assert body["status"] in ("queued", "running", "done")


def test_invalid_expand_mode_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"expand_mode": "bad"})
    assert r.status_code == 422


def test_non_numeric_expand_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"expand_top": "abc"})
    assert r.status_code == 422


def test_negative_pixel_expand_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"expand_top": "-5"})
    assert r.status_code == 422


def test_out_of_range_ratio_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"expand_mode": "ratio", "expand_top": "0.9"})
    assert r.status_code == 422


def test_invalid_regex_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"question_regex": "["})
    assert r.status_code == 422


def test_regex_matching_empty_returns_422(client):
    r = client.post("/predict/exam",
                    files={"file": _make_image()},
                    data={"question_regex": ".*"})
    assert r.status_code == 422


def test_non_image_mime_returns_415(client):
    r = client.post("/predict/exam",
                    files={"file": ("notes.txt", b"plain text", "text/plain")})
    assert r.status_code == 415
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/exam/test_app_exam.py -v
```

Expected: 4 failed (no /predict/exam endpoints), 6 failed (file format issues)

- [ ] **Step 3: 修改 service/app.py**

在 `service/app.py` 中追加路由（在 `root()` 路由之前）：

```python
from service.exam import tasks as exam_tasks
from service.exam.pipeline import run_pipeline
from service.exam.question_filter import validate_regex


# Reject helper for invalid regex
def _validate_exam_params(
    expand_mode: str,
    expand_top: float, expand_bottom: float,
    expand_left: float, expand_right: float,
    question_regex: str | None,
) -> str | None:
    """Return error_code if invalid, else None."""
    if expand_mode not in ("pixel", "ratio"):
        return "INVALID_EXPAND_MODE"
    if expand_mode == "pixel":
        for name, v in [("expand_top", expand_top), ("expand_bottom", expand_bottom),
                        ("expand_left", expand_left), ("expand_right", expand_right)]:
            if v < 0:
                return f"{name.upper()}_NEGATIVE"
    else:  # ratio
        for name, v in [("expand_top", expand_top), ("expand_bottom", expand_bottom),
                        ("expand_left", expand_left), ("expand_right", expand_right)]:
            if v < -0.5 or v > 0.5:
                return f"{name.upper()}_OUT_OF_RANGE"
    if question_regex is not None:
        try:
            validate_regex(question_regex)
        except ValueError as e:
            return f"INVALID_REGEX: {e}"
    return None
```

然后在 `create_app()` 函数内（`@app.post("/predict")` 之前）追加：

```python
    @app.post("/predict/exam")
    async def predict_exam(
        file: UploadFile = File(...),
        expand_mode: str = Form("pixel"),
        expand_top: float = Form(20.0),
        expand_bottom: float = Form(20.0),
        expand_left: float = Form(20.0),
        expand_right: float = Form(20.0),
        question_regex: str | None = Form(None),
        conf: float = Form(0.3),
        imgsz: int = Form(1024),
    ):
        # Validate params first
        try:
            expand_top_f = float(expand_top)
            expand_bottom_f = float(expand_bottom)
            expand_left_f = float(expand_left)
            expand_right_f = float(expand_right)
        except (TypeError, ValueError):
            return JSONResponse(status_code=422,
                                content={"detail": "expand_* must be numeric",
                                         "error_code": "INVALID_PARAM"})

        err = _validate_exam_params(expand_mode, expand_top_f, expand_bottom_f,
                                    expand_left_f, expand_right_f, question_regex)
        if err:
            return JSONResponse(status_code=422, content={"detail": err, "error_code": err})

        # MIME check
        if not (file.content_type or "").startswith("image/"):
            return JSONResponse(status_code=415,
                                content={"detail": "unsupported media type",
                                         "error_code": "BAD_MIME"})

        raw = await file.read()
        if len(raw) > cfg.max_file_size_mb * 1024 * 1024:
            return JSONResponse(status_code=413,
                                content={"detail": f"file too large, max {cfg.max_file_size_mb}MB",
                                         "error_code": "FILE_TOO_LARGE"})

        # Submit task
        task_id = exam_tasks.create_task()
        params = {
            "expand_mode": expand_mode,
            "expand_top": expand_top_f,
            "expand_bottom": expand_bottom_f,
            "expand_left": expand_left_f,
            "expand_right": expand_right_f,
            "question_regex": question_regex,
            "conf": conf,
            "imgsz": imgsz,
            "device": cfg.device or "cpu",
        }
        model = state["model"]
        sem = state["semaphore"]
        if model is None:
            return JSONResponse(status_code=503,
                                content={"detail": "model not ready",
                                         "error_code": "NOT_READY"})

        # Fire-and-forget background task
        import asyncio
        asyncio.create_task(_run_exam_pipeline_async(task_id, raw, params, model, sem))

        from service.exam.schemas import TaskSubmitResponse
        return TaskSubmitResponse(
            task_id=task_id,
            status="queued",
            submit_url=f"/predict/exam/{task_id}/status",
        )

    @app.get("/predict/exam/{task_id}/status")
    async def exam_status(task_id: str):
        state = exam_tasks.get_task(task_id)
        if state is None:
            return JSONResponse(status_code=404,
                                content={"detail": "task not found or expired",
                                         "error_code": "TASK_NOT_FOUND"})
        from service.exam.schemas import TaskStatusResponse, StageResult
        stages = {
            k: StageResult(
                started_at=0.0,
                finished_at=0.0,
                duration_ms=v.get("duration_ms"),
                payload=v.get("payload"),
                error=v.get("error"),
            )
            for k, v in state["stages"].items()
        }
        return TaskStatusResponse(
            task_id=state["task_id"],
            status=state["status"],
            progress=state.get("progress"),
            stages=stages,
            final=state["final"],
            error=state.get("error"),
        )


async def _run_exam_pipeline_async(task_id, image_bytes, params, model, sem):
    """Run pipeline in background thread (OCR is CPU-bound, blocking)."""
    import asyncio
    loop = asyncio.get_event_loop()
    # run_pipeline signature requires keyword-only model= and semaphore=;
    # functools.partial binds them so run_in_executor can call positionally.
    from functools import partial
    await loop.run_in_executor(
        None,
        partial(run_pipeline, task_id, image_bytes, params, model=model, semaphore=sem),
    )
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/exam/test_app_exam.py -v
```

Expected: 10 passed

- [ ] **Step 5: 提交**

```bash
git add service/app.py tests/exam/test_app_exam.py
git commit -m "feat(service): add /predict/exam endpoints with task polling"
```

---

## Task 13: 端到端测试（slow 标记）

**Files:**
- Create: `tests/exam/test_e2e.py`

- [ ] **Step 1: 写 E2E 测试**

```python
"""End-to-end test with real PaddleOCR. Marked slow."""
import time

import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from service.exam import tasks


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_tasks():
    tasks.reset_for_tests()
    yield
    tasks.reset_for_tests()


@pytest.mark.slow
def test_real_pipeline_with_real_paddleocr(client):
    """Submit, poll until done, verify final has at least one question."""
    with open("assets/example/exam_paper.jpg", "rb") as f:
        r = client.post(
            "/predict/exam",
            files={"file": ("exam_paper.jpg", f, "image/jpeg")},
            data={
                "expand_mode": "pixel",
                "expand_top": 20, "expand_bottom": 20,
                "expand_left": 20, "expand_right": 20,
            },
        )
    assert r.status_code == 200
    tid = r.json()["task_id"]

    # Poll up to 120 seconds
    deadline = time.time() + 120
    while time.time() < deadline:
        r = client.get(f"/predict/exam/{tid}/status")
        assert r.status_code == 200
        status = r.json()["status"]
        if status == "done":
            body = r.json()
            assert body["final"] is not None
            assert len(body["final"]["ocr_blocks"]) > 0
            return
        if status == "failed":
            pytest.fail(f"task failed: {body['error']}")
        time.sleep(1)
    pytest.fail("task did not complete within 120s")
```

- [ ] **Step 2: 不实际运行（默认 pytest 跳过 @slow）**

```bash
pytest tests/exam/test_e2e.py -v -m "not slow"
```

Expected: 1 deselected (skipped)

- [ ] **Step 3: 提交**

```bash
git add tests/exam/test_e2e.py
git commit -m "test(exam): add slow E2E test with real PaddleOCR"
```

---

## Task 14: UI — 试卷模式页签

**Files:**
- Modify: `service/static/index.html`

- [ ] **Step 1: 在 index.html 中添加 tab UI**

在 `<h1>DocLayout-YOLO 检测服务</h1>` 之后插入 tab 切换：

```html
<div class="tabs">
  <button class="tab active" data-tab="general">通用版面</button>
  <button class="tab" data-tab="exam">试卷模式</button>
</div>
```

把现有 `<div id="dropZone">` 和后续表单/按钮区域包到 `<div id="tab-general" class="tab-content">` 中。

在 `<div id="tab-general">` 之后追加：

```html
<div id="tab-exam" class="tab-content" style="display:none">
  <h2>试卷版面识别</h2>
  <div id="dropZoneExam" class="drop-zone">拖拽试卷图片到这里,或点击选择</div>
  <input type="file" id="fileInputExam" accept="image/*" style="display:none">

  <div class="params">
    <label>放大模式</label>
    <div>
      <label><input type="radio" name="expandMode" value="pixel" checked> 像素</label>
      <label><input type="radio" name="expandMode" value="ratio"> 比例</label>
    </div>
    <span></span>

    <label for="examExpandTop">顶部</label>
    <input type="number" id="examExpandTop" value="20" step="0.05">
    <span id="examExpandTopUnit">px</span>

    <label for="examExpandBottom">底部</label>
    <input type="number" id="examExpandBottom" value="20" step="0.05">
    <span id="examExpandBottomUnit">px</span>

    <label for="examExpandLeft">左侧</label>
    <input type="number" id="examExpandLeft" value="20" step="0.05">
    <span id="examExpandLeftUnit">px</span>

    <label for="examExpandRight">右侧</label>
    <input type="number" id="examExpandRight" value="20" step="0.05">
    <span id="examExpandRightUnit">px</span>

    <label for="examRegex">题号正则</label>
    <input type="text" id="examRegex" value="^\s*\(?\d+[\.\)]" size="30">
    <span></span>
  </div>

  <button id="submitExamBtn" disabled>开始识别</button>

  <div id="examStatus" class="status"></div>

  <h2 id="examSummary" style="display:none">识别结果</h2>
  <div class="result" id="examResult" style="display:none">
    <div>
      <p>题目标注图:</p>
      <img id="examAnnotatedImg">
      <p><a id="examDownloadLink" download="exam_annotated.jpg">下载标注图</a></p>
    </div>
    <div>
      <p>原图:</p>
      <img id="examOriginalImg">
    </div>
  </div>

  <h3 id="examStagesTitle" style="display:none">阶段耗时</h3>
  <table id="examStagesTable" style="display:none">
    <thead><tr><th>阶段</th><th>耗时 (ms)</th><th>结果</th><th>错误</th></tr></thead>
    <tbody></tbody>
  </table>

  <h3 id="examQuestionsTitle" style="display:none">题目列表</h3>
  <table id="examQuestionsTable" style="display:none">
    <thead><tr><th>#</th><th>题号文本</th><th>坐标 (x1,y1,x2,y2)</th><th>置信度</th></tr></thead>
    <tbody></tbody>
  </table>
</div>
```

- [ ] **Step 2: 添加 CSS**

在 `<style>` 块内追加：

```css
.tabs { display: flex; gap: 4px; margin: 16px 0; border-bottom: 1px solid #ddd; }
.tab { padding: 8px 16px; background: #f0f0f0; border: 1px solid #ddd; border-bottom: none; cursor: pointer; }
.tab.active { background: #fff; border-bottom: 1px solid #fff; margin-bottom: -1px; font-weight: bold; }
```

- [ ] **Step 3: 添加 JS — tab 切换、模式切换、提交与轮询**

在 `<script>` 块最末尾追加：

```javascript
// ===== 试卷模式 =====
const examDropZone = document.getElementById('dropZoneExam');
const examFileInput = document.getElementById('fileInputExam');
const examSubmitBtn = document.getElementById('submitExamBtn');
const examStatusEl = document.getElementById('examStatus');
const examSummary = document.getElementById('examSummary');
const examResult = document.getElementById('examResult');
const examStagesTitle = document.getElementById('examStagesTitle');
const examStagesTable = document.getElementById('examStagesTable');
const examQuestionsTitle = document.getElementById('examQuestionsTitle');
const examQuestionsTable = document.getElementById('examQuestionsTable');

let examFile = null;

function showExamStatus(text, kind) {
  examStatusEl.textContent = text;
  examStatusEl.className = 'status show ' + kind;
}
function hideExamStatus() { examStatusEl.className = 'status'; }

// Tab switching
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    document.getElementById('tab-general').style.display = tab === 'general' ? '' : 'none';
    document.getElementById('tab-exam').style.display = tab === 'exam' ? '' : 'none';
  });
});

// Mode toggle → update unit labels
function updateUnitLabels() {
  const mode = document.querySelector('input[name="expandMode"]:checked').value;
  const unit = mode === 'pixel' ? 'px' : '比例';
  ['examExpandTop', 'examExpandBottom', 'examExpandLeft', 'examExpandRight'].forEach(id => {
    document.getElementById(id + 'Unit').textContent = unit;
  });
}
document.querySelectorAll('input[name="expandMode"]').forEach(r => {
  r.addEventListener('change', updateUnitLabels);
});
updateUnitLabels();

// Drag & drop for exam
examDropZone.addEventListener('click', () => examFileInput.click());
examDropZone.addEventListener('dragover', e => { e.preventDefault(); examDropZone.classList.add('dragging'); });
examDropZone.addEventListener('dragleave', () => examDropZone.classList.remove('dragging'));
examDropZone.addEventListener('drop', e => {
  e.preventDefault();
  examDropZone.classList.remove('dragging');
  const f = e.dataTransfer.files[0];
  if (f) setExamFile(f);
});
examFileInput.addEventListener('change', () => {
  if (examFileInput.files[0]) setExamFile(examFileInput.files[0]);
});

function setExamFile(f) {
  if (!f.type.startsWith('image/')) {
    showExamStatus('请上传图片文件', 'error');
    return;
  }
  examFile = f;
  examDropZone.textContent = f.name + ' (' + Math.round(f.size / 1024) + ' KB)';
  examSubmitBtn.disabled = false;
  hideExamStatus();
}

examSubmitBtn.addEventListener('click', async () => {
  if (!examFile) return;
  examSubmitBtn.disabled = true;
  showExamStatus('提交任务...', 'loading');
  examSummary.style.display = 'none';
  examResult.style.display = 'none';
  examStagesTitle.style.display = 'none';
  examStagesTable.style.display = 'none';
  examQuestionsTitle.style.display = 'none';
  examQuestionsTable.style.display = 'none';

  const form = new FormData();
  form.append('file', examFile);
  form.append('expand_mode', document.querySelector('input[name="expandMode"]:checked').value);
  form.append('expand_top', document.getElementById('examExpandTop').value);
  form.append('expand_bottom', document.getElementById('examExpandBottom').value);
  form.append('expand_left', document.getElementById('examExpandLeft').value);
  form.append('expand_right', document.getElementById('examExpandRight').value);
  form.append('question_regex', document.getElementById('examRegex').value);

  try {
    const r = await fetch('/predict/exam', { method: 'POST', body: form });
    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: r.statusText}));
      showExamStatus('提交失败: ' + (err.detail || r.status), 'error');
      examSubmitBtn.disabled = false;
      return;
    }
    const submit = await r.json();
    showExamStatus('任务已提交，ID=' + submit.task_id + '，等待处理...', 'loading');

    // Poll
    const deadline = Date.now() + 5 * 60 * 1000;
    while (Date.now() < deadline) {
      await new Promise(res => setTimeout(res, 1000));
      const sr = await fetch(submit.submit_url);
      if (!sr.ok) {
        showExamStatus('查询失败: ' + sr.status, 'error');
        examSubmitBtn.disabled = false;
        return;
      }
      const status = await sr.json();
      if (status.status === 'done') {
        renderExamResult(status);
        return;
      }
      if (status.status === 'failed') {
        showExamStatus('任务失败: ' + (status.error || 'unknown'), 'error');
        examSubmitBtn.disabled = false;
        return;
      }
      showExamStatus('处理中...当前阶段=' + (status.progress || 'queued'), 'loading');
    }
    showExamStatus('任务超时', 'error');
    examSubmitBtn.disabled = false;
  } catch (e) {
    showExamStatus('请求失败: ' + e.message, 'error');
    examSubmitBtn.disabled = false;
  }
});

function renderExamResult(status) {
  const final = status.final;
  document.getElementById('examAnnotatedImg').src = final.annotated_image;
  document.getElementById('examOriginalImg').src = final.original_image;
  document.getElementById('examDownloadLink').href = final.annotated_image;
  examSummary.style.display = 'block';
  examSummary.textContent = `识别结果: ${final.questions.length} 个题目`;
  examResult.style.display = 'grid';

  // Stages table
  const stagesTbody = examStagesTable.querySelector('tbody');
  stagesTbody.innerHTML = '';
  for (const [name, stage] of Object.entries(status.stages)) {
    const row = document.createElement('tr');
    const result = stage.payload ? JSON.stringify(stage.payload) : '-';
    row.innerHTML = `<td>${name}</td><td>${stage.duration_ms ?? '-'}</td><td>${result}</td><td>${stage.error || ''}</td>`;
    stagesTbody.appendChild(row);
  }
  examStagesTitle.style.display = 'block';
  examStagesTable.style.display = 'table';

  // Questions table
  const qTbody = examQuestionsTable.querySelector('tbody');
  qTbody.innerHTML = '';
  final.questions.forEach(q => {
    const row = document.createElement('tr');
    row.innerHTML = `<td>${q.index}</td><td>${(q.matched_text || '').slice(0, 50)}</td>` +
      `<td>[${q.bbox_xyxy.map(v => v.toFixed(1)).join(', ')}]</td><td>${q.ocr_score.toFixed(3)}</td>`;
    qTbody.appendChild(row);
  });
  examQuestionsTitle.style.display = final.questions.length > 0 ? 'block' : 'none';
  examQuestionsTable.style.display = final.questions.length > 0 ? 'table' : 'none';

  showExamStatus('完成', 'success');
  examSubmitBtn.disabled = false;
}
```

- [ ] **Step 4: 验证 HTML 语法（Python 简单解析）**

```bash
python -c "
from html.parser import HTMLParser
class P(HTMLParser):
    def error(self, msg): raise ValueError(msg)
p = P()
p.feed(open('service/static/index.html').read())
print('OK')
"
```

Expected: `OK`

- [ ] **Step 5: 提交**

```bash
git add service/static/index.html
git commit -m "feat(service): add exam mode tab to upload UI"
```

---

## Task 15: README 更新

**Files:**
- Modify: `service/README.md`

- [ ] **Step 1: 在 API 简述之后追加考试端点**

```bash
cat >> service/README.md << 'EOF'

## 试卷模式（Exam Mode）

新增 `/predict/exam` 异步端点，专门用于试卷版面识别。

### 用法

```bash
# 1. 提交任务
curl -X POST http://localhost:8000/predict/exam \
  -F "file=@path/to/exam_paper.jpg" \
  -F "expand_mode=pixel" \
  -F "expand_top=20" -F "expand_bottom=20" \
  -F "expand_left=20" -F "expand_right=20" \
  -F "question_regex=^\s*\(?\d+[\.\)]" \
  -o submit.json

# 2. 轮询状态
TASK_ID=$(jq -r .task_id submit.json)
curl http://localhost:8000/predict/exam/$TASK_ID/status > status.json
```

返回 JSON 包含：
- `status` — `queued` / `running` / `done` / `failed`
- `stages` — 5 阶段（yolo/expand/ocr/filter/geometry）耗时与错误
- `final.questions` — 识别出的题目框（含坐标、文本、置信度）
- `final.ocr_blocks` — 每个 plain text 框的 OCR 结果

### 安装

```bash
pip install -e ".[exam]"
```

### 环境

PaddleOCR 模型在首次 OCR 调用时懒加载（约 150MB 下载）。

EOF
```

- [ ] **Step 2: 验证**

```bash
tail -50 service/README.md
```

- [ ] **Step 3: 提交**

```bash
git add service/README.md
git commit -m "docs(service): document /predict/exam endpoints"
```

---

## Task 16: 完整测试运行 + 覆盖率核查

- [ ] **Step 1: 运行全部测试**

```bash
pytest tests/ -v --tb=short
```

Expected: 全部通过（含原有 22 个 + exam 子包 ~50 个）

- [ ] **Step 2: 检查 exam 子包覆盖率**

```bash
pytest tests/exam/ --cov=service.exam --cov-report=term-missing
```

目标：`exam/` 子包 ≥ 90% 覆盖率

- [ ] **Step 3: 验证现有 /predict 未受影响**

```bash
pytest tests/test_app.py tests/test_inference.py tests/test_config.py tests/test_schemas.py -v
```

Expected: 22 passed (no regressions)

---

## 风险与回滚

| 风险 | 回滚 |
|---|---|
| PaddleOCR 模型下载失败 | lazy load, 阶段 error 不阻断 |
| 异步 worker 内存泄漏 | LRU + TTL 自动清理 |
| UI 改动破坏现有页面 | tab 切换隔离，原 tab 默认显示 |
| 现有 /predict 行为变化 | task 12 路由独立，无共享代码 |
