# DocLayout-YOLO 试卷版面流水线 — 设计文档

- **日期**：2026-07-08
- **作者**：Claude
- **状态**：Approved（待用户最终确认）
- **适用版本**：main（含已合并的 `feature/add-http-service`）

---

## 1. 背景与目标

### 1.1 背景
当前 `service/` 提供通用的版面检测（`/predict`、`/predict/image`），无 OCR、无版面后处理。用户希望针对**中文试卷**场景：
- 对模型输出的 `plain text` 框先放大并裁切
- 通过 OCR 识别裁切内容
- 用正则判定哪些是"题目框"
- 将相邻题目框的版面向下延伸，形成完整题目区块
- 输出重新标注后的图像 + 多阶段调试数据

### 1.2 目标
在不破坏现有 `/predict` 行为的前提下，新增**异步**试卷流水线端点，**始终返回多阶段调试数据**以便排查 OCR 与正则匹配问题。

### 1.3 非目标
- 不修改现有 `/predict` / `/predict/image` 行为
- 不做分布式任务队列（内存 LRU 即可）
- 不支持批量多图（一次一张）
- 不做 OCR 训练或模型微调

---

## 2. 关键决策（已通过）

| 决策点 | 选择 |
|---|---|
| OCR 引擎 | **PaddleOCR**（中文最优） |
| 集成方式 | **新增专用端点** `/predict/exam` |
| 题号正则 | 默认阿拉伯数字（`1.`/`(2)`/`10.`），API 支持自定义 |
| 框放大 | **混合**：像素 + 比例 + 4 方向独立值，UI 提供切换 |
| "最后框" 定义 | 所有框中 **y2 最大且类别为 plain text** |
| 响应输出 | **多阶段调试**（始终返回完整中间数据） |
| 执行模式 | **异步任务 + 状态轮询** |
| 错误处理 | 中间过程错误**不阻断任务**，仅记录在响应 |
| UI 范围 | **同步更新** `index.html`（新增"试卷模式"页签） |
| 架构 | **方案 A**：轻量独立模块化（`exam/` 子包） |

---

## 3. 架构

### 3.1 总体流程

```
                           HTTP 客户端
                                │
                                ▼
   POST /predict/exam  ──▶  FastAPI 路由  ──▶ 立即返回 {task_id, status: "queued"}
                                │
                                ▼
                         内存任务队列  (dict[task_id, TaskState])
                                │
                                ▼
                         后台 worker (asyncio.create_task)
                          ├─ Stage 1: YOLO 推理（复用现有 inference.py）
                          ├─ Stage 2: 框放大（exam/expand.py）
                          ├─ Stage 3: 裁切 + OCR（exam/ocr.py）
                          ├─ Stage 4: 题号判定（exam/question_filter.py）
                          └─ Stage 5: 坐标延伸 + 重绘（exam/geometry.py）
                                │
                                ▼
                         更新 TaskState → 标记 status: "done"
                                │
                                ▼
   GET /predict/exam/{id}/status  ◀──  客户端轮询
```

### 3.2 端点表面

| 端点 | 方法 | 作用 |
|---|---|---|
| `/predict/exam` | POST | 提交任务（form-data） |
| `/predict/exam/{task_id}/status` | GET | 轮询任务状态 |
| `/` | GET | UI 增加"试卷模式"页签 |

### 3.3 核心原则
- **单一事实源**：`TaskState` 包含原始框、放大框、OCR、题目框、最终图，所有阶段数据
- **后台不阻塞 HTTP**：路由处理只接收 + 入队；后台 worker 异步跑流水线
- **可观测**：每阶段时间戳 + 状态字段写入 `TaskState`

---

## 4. 组件分解

| 模块 | 文件 | 职责 | 公共接口 |
|---|---|---|---|
| **裁切放大** | `exam/expand.py` | 把 plain text 框按 4 方向外扩；像素或比例模式 | `expand_boxes(boxes, mode, top, bottom, left, right) -> boxes` |
| **OCR 引擎封装** | `exam/ocr.py` | PaddleOCR 懒加载 + 单图识别 | `OcrEngine` 类；`ocr_image(np.ndarray) -> list[TextBlock]`；`TextBlock = {text, bbox, score}` |
| **题号判定** | `exam/question_filter.py` | 用正则匹配识别文本 | `is_question(text: str, regex: str) -> bool` |
| **几何延伸** | `exam/geometry.py` | 题目框下边=下个题目框上边；最后框下边=max(y2) plain text 框 y2 | `extend_questions(questions, all_boxes) -> questions` |
| **任务存储** | `exam/tasks.py` | 内存 dict[task_id, TaskState]；LRU 淘汰 + 状态字段 | `create_task()`, `get_task(id)`, `update_stage(id, stage, payload)` |
| **流水线编排** | `exam/pipeline.py` | 5 阶段串行调度；catch 异常写入 error 字段而非 raise | `run_pipeline(task_id, image_bytes, params) -> None` |
| **Pydantic 模型** | `exam/schemas.py` | `ExamRequest` / `TaskSubmitResponse` / `TaskStatusResponse` 等 | — |
| **路由 + 后台 worker** | `service/app.py`（追加） | `POST /predict/exam`、`GET /predict/exam/{task_id}/status`；启动期创建 worker | — |
| **UI** | `service/static/index.html`（追加） | "试卷模式"页签；参数面板（像素/比例 + 上下/左右）；提交 + 轮询 + 结果展示 | — |

### 4.1 TaskState

```python
@dataclass
class TaskState:
    task_id: str
    status: Literal["queued", "running", "done", "failed", "expired"]
    created_at: float
    updated_at: float
    stages: dict[str, StageResult]  # yolo / expand / ocr / filter / geometry
    final: FinalResult | None
    error: str | None
```

### 4.2 StageResult

```python
class StageResult(BaseModel):
    started_at: float
    finished_at: float | None
    duration_ms: int | None
    payload: dict | None      # 各阶段专属数据
    error: str | None         # 阶段内错误，不阻断任务
```

---

## 5. 数据流

### 5.1 POST /predict/exam 请求体（multipart/form-data）

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `file` | ✓ | UploadFile | 试卷图片 |
| `expand_mode` | | `"pixel"` \| `"ratio"`（默认 `"pixel"`） | 框放大模式 |
| `expand_top` | | float（默认 20） | 顶部外扩（像素或比例） |
| `expand_bottom` | | float（默认 20） | 底部外扩 |
| `expand_left` | | float（默认 20） | 左侧外扩 |
| `expand_right` | | float（默认 20） | 右侧外扩 |
| `question_regex` | | string（默认内置阿拉伯数字正则） | 题号正则 |
| `conf` / `imgsz` | | float / int | 透传 YOLO 参数 |

### 5.2 POST 立即响应

```json
{
  "task_id": "abc123",
  "status": "queued",
  "submit_url": "/predict/exam/abc123/status"
}
```

### 5.3 GET /predict/exam/{task_id}/status 响应

运行中：

```json
{
  "task_id": "abc123",
  "status": "running",
  "progress": "ocr",
  "stages": {
    "yolo":     { "duration_ms": 6500, "payload": {"num_detections": 18}, "error": null },
    "expand":   { "duration_ms": 12,   "payload": {"num_expanded": 7},     "error": null },
    "ocr":      { "duration_ms": null, "payload": null,                    "error": null },
    "filter":   { "duration_ms": null, "payload": null,                    "error": null },
    "geometry": { "duration_ms": null, "payload": null,                    "error": null }
  },
  "final": null,
  "error": null
}
```

`status = "done"` 时 `final` 包含：

```json
{
  "final": {
    "annotated_image": "data:image/jpeg;base64,...",
    "original_image": "data:image/jpeg;base64,...",
    "questions": [
      {
        "index": 1,
        "bbox_xyxy": [127.0, 871.0, 1523.0, 1501.0],
        "matched_text": "1. (计算题) ...",
        "source_detection_ids": [1, 6],
        "ocr_score": 0.93
      }
    ],
    "ocr_blocks": [
      {
        "source_detection_id": 1,
        "bbox_xyxy": [...],
        "text": "1. (计算题) ...",
        "score": 0.93,
        "is_question": true
      }
    ]
  }
}
```

### 5.4 流水线阶段细节

```
[0] decode image_bytes → np.ndarray (BGR)
        ↓
[1] YOLO 推理（复用 service.inference.predict_one）
        → det_res (Results 对象)
        ↓
[2] parse_detections → original_detections
        ↓
[3] expand_boxes（仅 plain text 类）→ expanded_boxes
        ↓
[4] crop each expanded box → ocr_blocks
        ↓
[5] is_question(block.text, regex) → is_question flag
        ↓
[6] extend_questions：题目框下边=下一题上边；最后题= max(y2, plain text)
        ↓
[7] 重绘：把 extended questions 画到原图
        ↓
[8] 写入 TaskState.final
```

---

## 6. 错误处理

### 6.1 分层错误模型

```
端点层（HTTP）
  ├─ 422：参数校验失败
  ├─ 413：文件过大
  ├─ 415：MIME 错误
  └─ 503：模型未加载

任务层
  └─ 任何 422/413/415 错误在落盘阶段就拒绝，不创建 task

流水线层（每阶段内部）
  ├─ 单阶段异常 → 写入 stages[xx].error
  ├─ status: "running" 但能进入下一阶段
  └─ 最终 status: "done"（带部分错误）或 "failed"（致命）

致命错误 → status: "failed" + error 字段
  ├─ YOLO 推理崩溃
  ├─ 图像解码失败
  └─ 任务在 worker 中启动前崩溃
```

### 6.2 各阶段错误处理

| 阶段 | 错误情况 | 处理 |
|---|---|---|
| decode | cv2.imdecode 返回 None | 阶段 error，TaskState 标 failed（致命） |
| yolo | predict 抛异常 | 阶段 error，failed（致命） |
| expand | 0 个 plain text 框 | 阶段 warning，`"no plain text"` |
| expand | expand 后坐标越界 | clamp 到 `[0, W]` / `[0, H]` |
| ocr | 单张裁切为空图像 | ocr_blocks 该项 = `{text: "", score: 0}` |
| ocr | PaddleOCR 模型崩溃 | 阶段 error，进入"empty OCR"分支 |
| filter | 正则非法 | 阶段 error（致命） |
| filter | 0 个题号命中 | 阶段 payload = `{"num_questions": 0}`，不报错 |
| geometry | 题目框少于 1 个 | 阶段 warning，使用原始题目框 |
| 重绘 | 任意错误 | 阶段 error，final.image = 原图 base64 |

### 6.3 验证层（POST 时立即拒绝）

```
expand_mode 不在 {"pixel","ratio"}            → 422
expand_* 非数字                              → 422
expand_* < 0（像素）/ 不在 [-0.5, 0.5]（比例） → 422
question_regex 非 Python 正则                 → 422
question_regex 匹配空字符串                    → 422
```

### 6.4 PaddleOCR 懒加载

- 启动时**不加载**，避免拖慢现有 `/predict`
- 首次调用时加载；失败 → 阶段 error，status = failed
- 加载耗时会被记录到 `stages.ocr.duration_ms`

### 6.5 任务清理

- 内存存储上限 N=64（LRU 淘汰）
- 单任务超时：10 分钟
- 超时任务 → 标记 `"expired"` + 不再返回 final
- 进程重启清空全部

### 6.6 日志

- 阶段开始/结束 → INFO（含耗时）
- 阶段错误 → ERROR（带堆栈）
- 最终状态 → INFO（任务 ID + 总耗时）

---

## 7. 测试策略

### 7.1 测试文件结构

```
tests/
├── test_app.py                  # 现有：不变
├── test_config.py
├── test_inference.py
├── test_schemas.py
├── exam/                        # 新增子目录
│   ├── __init__.py
│   ├── test_expand.py
│   ├── test_question_filter.py
│   ├── test_geometry.py
│   ├── test_tasks.py
│   ├── test_pipeline.py
│   ├── test_ocr.py              # mock PaddleOCR
│   └── test_app_exam.py
└── fixtures/
    └── exam/
        ├── single_question.png
        ├── three_questions.png
        └── no_plain_text.png
```

### 7.2 测试层次

| 层 | 工具 | 覆盖 |
|---|---|---|
| 单元 | pytest | 各模块纯函数逻辑 |
| Mock OCR | monkeypatch + Fake OCR | 流水线阶段间边界，不依赖真实 PaddleOCR |
| 端到端 | FastAPI TestClient + 真实 PaddleOCR | 1 个 E2E（CI 可标记 `@pytest.mark.slow`） |
| 校验 | Form 字段边界 | 422 参数错误覆盖 |

### 7.3 关键测试用例

**test_expand.py**
- pixel 模式：4 方向独立值
- ratio 模式：0.1 比例；大/小框自适应
- 越界：clamp 到边界
- 0 plain text 框 → 返回空列表
- 仅 plain text 类被放大，其他类别跳过

**test_question_filter.py**
- 默认正则：匹配 `"1."` / `"(2)"` / `"10."` / `"Question 3"`
- 默认正则：不匹配 `"1.1"` / `"abc"` / 空字符串
- 自定义正则：通过
- 非法正则：抛 `ValueError`

**test_geometry.py**
- 3 个题号 → 中间题下边 = 下题上边
- 最后一个题下边 = max(y2) plain text
- 1 个题号 → 仍能找到 max plain text 下边
- 0 个题号 → 返回空列表
- 输入框顺序乱：先按 y1 排序

**test_tasks.py**
- create → get 状态
- update_stage 不变 status
- final 写入后 status 自动变 done
- LRU：第 65 个任务 → 最早 task 被淘汰
- 过期：10 分钟前创建 → get_task 标记 expired

**test_pipeline.py**
- 全 5 阶段跑通：fake OCR → 状态 done + final 有数据
- OCR 错误注入 → stages.ocr.error 有值，status 仍 done
- 致命错误（YOLO 抛异常）→ status failed
- 0 题号 → 状态 done，questions=[]
- 自定义正则 + 实际题目匹配

**test_app_exam.py**
- 200 路径：上传 → task_id → 轮询 → done + final
- 422：expand_mode="bad" / 非数字
- 413：上传 30MB 文件
- 415：上传 .txt 文件
- 503 模型未加载（mock）

### 7.4 E2E 测试（slow 标记）

```python
@pytest.mark.slow
def test_real_pipeline_with_real_paddleocr(tmp_image):
    """Requires PaddleOCR models downloaded (~150MB)."""
```

### 7.5 质量门槛

- 覆盖率目标：`exam/` 子包 ≥ 90%
- 所有 5 个 stage 函数单独可测
- HTTP 测试用 TestClient，不依赖真实 uvicorn

### 7.6 端到端手动验证

```bash
MODEL_PATH=... python -m service.app

curl -X POST http://localhost:8000/predict/exam \
  -F "file=@assets/example/exam_paper.jpg" \
  -F "expand_mode=pixel" \
  -F "expand_top=20" -F "expand_bottom=20" \
  -F "question_regex=^\s*\(?\d+[\.\)]" \
  -o submit.json

TASK_ID=$(jq -r .task_id submit.json)
while [ "$(jq -r .status poll.json)" != "done" ]; do
  curl http://localhost:8000/predict/exam/$TASK_ID/status > poll.json
  sleep 1
done

jq '.stages' poll.json
jq '.final.questions' poll.json
```

---

## 8. 文件改动清单

### 8.1 新增文件

```
service/exam/__init__.py
service/exam/pipeline.py
service/exam/ocr.py
service/exam/expand.py
service/exam/question_filter.py
service/exam/geometry.py
service/exam/tasks.py
service/exam/schemas.py

tests/exam/__init__.py
tests/exam/test_expand.py
tests/exam/test_question_filter.py
tests/exam/test_geometry.py
tests/exam/test_tasks.py
tests/exam/test_pipeline.py
tests/exam/test_ocr.py
tests/exam/test_app_exam.py

tests/fixtures/exam/single_question.png
tests/fixtures/exam/three_questions.png
tests/fixtures/exam/no_plain_text.png
```

### 8.2 修改文件

```
service/app.py               # 追加 /predict/exam 路由 + 后台 worker
service/static/index.html    # 追加"试卷模式"页签
pyproject.toml               # 追加 [exam] extras（paddleocr / paddlepaddle）
README.md                    # 追加 exam 端点说明
service/README.md            # 追加 exam 端点说明
```

### 8.3 不变文件

```
service/config.py            # 复用 max_file_size_mb
service/inference.py         # 复用 predict_one / parse_detections
service/schemas.py           # 不动
```

---

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| PaddleOCR 模型加载慢（首次 5-10s） | 懒加载；UI 显示进度 |
| OCR 模型 150MB 体积 | 可选 `pip install paddleocr`，不强制 |
| 单次请求总耗时长（30s+） | 异步任务 + 轮询，避免 HTTP 超时 |
| 内存任务存储重启丢失 | 接受（个人/小团队场景，文档明示） |
| OCR 错识别导致题号误判 | 多阶段调试输出方便排查；正则可自定义 |

---

## 10. 成功标准

- [ ] `POST /predict/exam` 提交后 100ms 内返回 task_id
- [ ] `GET /predict/exam/{id}/status` 返回完整 5 阶段数据
- [ ] 端到端测试：上传 exam_paper.jpg，正确识别至少 1 个题目框并延伸坐标
- [ ] UI 切换"试卷模式"后可调参并轮询结果
- [ ] 现有 `/predict` / `/predict/image` 行为零变化
- [ ] 全部 pytest 通过（含现有 22 个测试 + 新增 exam/ 子包测试）
