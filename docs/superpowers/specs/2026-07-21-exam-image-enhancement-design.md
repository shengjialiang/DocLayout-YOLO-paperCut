# 试卷图片增强(Exam Image Enhancement)— 设计文档

**日期**: 2026-07-21
**状态**: Approved (待用户最终书面审阅)
**作者**: liangjiasheng (with Claude Code)

---

## 1. 背景与目标

现有 `/predict/exam` 端点直接对上传图片跑 YOLO → OCR 流水线。在「试卷识别」页面,用户上传的往往是**手机拍摄的试卷照片**(背景桌面 + 透视畸变 + 阴影 + 弯曲),原始图直接送识别会显著降低检出率。

**目标**:在「开始识别」之前增加一个**「增强图片」**步骤,让用户先看到增强效果(原图 vs 增强图并排),可选「用增强图识别」或「用原图识别」。

**非目标(YAGNI)**:
- 不支持交互式手动调参(只做一键自动增强)
- 不训练新模型(只用预训练 DocTr + OpenCV 经典 CV)
- 不替代现有识别流水线,仅作为**预处理前置模块**

---

## 2. 选型决策(用户已确认)

| 决策点 | 选择 |
|---|---|
| 执行位置 | **服务端**(Python 后端) |
| 典型输入 | **手机拍摄的试卷照片** |
| 交互流程 | **一键增强 + 原图/增强图预览** |
| 算法组合 | **OpenCV 传统 CV + DocTr ML 混合** |
| 集成方式 | **方案 A**:独立的异步任务端点(镜像 `/predict/exam`) |

---

## 3. 架构

新增独立的「增强模块」,与现有识别流水线**解耦**。两个流水线通过前端数据流衔接,不共享内部状态。

```
┌─────────────────────────────────────────────────────────┐
│                       Browser                            │
│  [选择图片] → [增强图片] → 轮询 → 预览                  │
│                              ↓                          │
│              [用增强图识别] / [用原图识别]                │
│                              ↓                          │
│              POST /predict/exam (现有识别端点)           │
└─────────────────────────────────────────────────────────┘
                              ↕ HTTP / data URLs
┌─────────────────────────────────────────────────────────┐
│              FastAPI service (service/app.py)            │
│                                                          │
│  /predict/exam/enhance (新增)  ───► enhance.py (新增)    │
│  /predict/exam/enhance/{id}/status (新增)                │
│  /predict/exam  (现有,不动)  ───► pipeline.py (不动)      │
└─────────────────────────────────────────────────────────┘
```

**修改的文件**:
- 新增: `service/exam/enhance.py`
- 新增: `tests/test_enhance.py`, `tests/test_enhance_api.py`
- 修改: `service/app.py`(新增 2 个端点)
- 修改: `service/exam/tasks.py`(支持 `enhance:` 前缀的任务 key)
- 修改: `service/config.py`(新增 `docrect_model_path` 字段)
- 修改: `service/static/index.html`(新增按钮 + 预览区 + 增强图识别逻辑)
- 修改: `pyproject.toml`(新增 `[project.optional-dependencies].enhance`)
- 修改: `start.bat`(检测 DocRect 模型文件)

**完全不动**: `service/exam/pipeline.py`、`service/exam/ocr.py`、`service/exam/geometry.py`、`service/exam/expand.py`、`service/inference.py`

---

## 4. 增强流水线

实现位于 `service/exam/enhance.py`。

### 4.1 阶段定义

| # | 阶段 | 算法 | 输入 → 输出 |
|---|---|---|---|
| 0 | `decode` | OpenCV `imdecode` | bytes → BGR ndarray |
| 1 | `edge_crop` | 灰度 → `GaussianBlur` → `Canny` → `findContours` → 按面积 + `approxPolyDP` 取最大四边形 → `getPerspectiveTransform` + `warpPerspective` | BGR → 透视矫正后 BGR |
| 2 | `deskew` | 灰度 → `threshold` → `findContours` → `minAreaRect` 取主文字块角度 → 仿射旋转(±15° 限幅) | BGR → 旋转后 BGR |
| 3 | `dewarping` | DocTr 预训练模型(优先 ONNX Runtime,fallback PyTorch) | BGR → 去弯后 BGR |
| 4 | `clahe_enhance` | `cvtColor → LAB → CLAHE(clipLimit=2.0) → cvtColor`;可选 + `bilateralFilter` + 轻度锐化 | BGR → 增强后 BGR |

### 4.2 模块 API

```python
def run_enhance_pipeline(
    task_id: str,
    image_bytes: bytes,
    params: dict[str, Any],
) -> None:
    """更新任务状态,最终 tasks.mark_done(task_id, final)。
    params 可选字段:
      - enable_deskew: bool = True
      - enable_dewarp: bool = True (若 docrect 模型未配置则自动 False)
      - enable_clahe: bool = True
    """


@dataclass
class EnhancementFinal:
    original_image: str        # data:image/jpeg;base64,...
    enhanced_image: str        # data:image/jpeg;base64,...
    applied_stages: list[str]  # ["edge_crop", "deskew", "clahe", ...]
    enhanced_bytes_b64: str    # 原始 JPEG bytes 的 base64 — 前端可直接 fetch().blob() 还原 Blob
    output_size: tuple[int, int]  # (width, height)
```

### 4.3 DocTr 集成策略

- **首选**:`onnxruntime`(CPU 推理快,体积小)→ 需要 `models/docrect.onnx`(~15 MB)
- **备选**:PyTorch 直接推理(`models/DocTr_cached.pth` ~80 MB)
- 路径由环境变量 `DOCRECT_MODEL_PATH` 指定
- 模型懒加载:第一次进入阶段时初始化,后续复用(全局单例)
- **若文件不存在**:启动打 `[INFO]`,dewarping 阶段跳过,其余阶段正常运行

### 4.4 单阶段失败的语义

每个阶段的失败**不阻断**后续阶段。在 stages 表里记录 `error` 字段,最终结果继续返回。

| 阶段 | 失败情形 | 处理 |
|---|---|---|
| `edge_crop` | 找不到四边形 / 透视变换后图像过小 | 跳过,`payload.applied=false`,`error="NO_QUAD_FOUND"` |
| `deskew` | 角度超过 ±15° / 旋转后图像空 | 跳过,`payload.applied=false` |
| `dewarping` | 模型文件缺失 / ONNX/PyTorch 异常 | 跳过,`payload.applied=false`,`error="DOCRECT_UNAVAILABLE: ..."` |
| `clahe_enhance` | 任意异常 | 跳过,`payload.applied=false` |

**整体失败**(任务 `failed`):
- `decode` 失败:图片不可读
- `final` 装配失败

**完全无增强场景**:`applied_stages == []` 时,`enhanced_image == original_image`,前端显示「未应用任何增强」提示但不报错。

---

## 5. API 设计

### 5.1 提交增强任务

```
POST /predict/exam/enhance
  Content-Type: multipart/form-data
  Body:
    file: UploadFile (image/*, 必填)
    enable_deskew: bool = True
    enable_dewarp: bool = True
    enable_clahe: bool = True

  Response 202:
    {
      "task_id": "<uuid>",
      "status": "queued",
      "submit_url": "/predict/exam/enhance/<task_id>/status"
    }
```

错误码:
- `400 BAD_MIME` — 非图片
- `413 FILE_TOO_LARGE` — 超过 `cfg.max_file_size_mb`
- `503 NOT_READY` — 服务启动中

### 5.2 轮询增强状态

```
GET /predict/exam/enhance/<task_id>/status

  Response 200:
    {
      "task_id": "<uuid>",
      "status": "queued|running|done|failed",
      "progress": "decode|edge_crop|deskew|dewarping|clahe",
      "stages": {
        "decode":     {"duration_ms": int, "payload": dict, "error": str|null},
        "edge_crop":  {...},
        "deskew":     {...},
        "dewarping":  {...},
        "clahe_enhance": {...}
      },
      "final": null | {
        "original_image":   "data:image/jpeg;base64,...",
        "enhanced_image":   "data:image/jpeg;base64,...",
        "applied_stages":   ["edge_crop","deskew","clahe"],
        "enhanced_bytes_b64": "<base64 raw JPEG>",
        "output_size":      {"width": int, "height": int}
      },
      "error": null | "..."
    }
```

### 5.3 与现有识别端点的衔接

前端从 `final.enhanced_bytes_b64` 还原 Blob 后,作为 `file` 字段上传到 `/predict/exam`(端点零修改)。

---

## 6. 配置(`service/config.py`)

新增字段:

```python
@dataclass(frozen=True)
class Config:
    # 现有字段不变
    model_path: str
    device: str | None
    host: str
    port: int
    max_concurrent: int
    max_file_size_mb: int
    # 新增
    docrect_model_path: str | None   # None → dewarping 阶段跳过
```

环境变量:
- `DOCRECT_MODEL_PATH` — `.onnx` 或 `.pth` 文件路径。未设置时降级为跳过 dewarping。

**启动期检测**(`start.bat`):
```
if defined DOCRECT_MODEL_PATH (
    if not exist "%DOCRECT_MODEL_PATH%" (
        echo [INFO] DocRect model not found at %DOCRECT_MODEL_PATH%; dewarping stage will be skipped.
    )
)
```

**依赖清单**(新增到 `pyproject.toml`):
```toml
enhance = [
    "onnxruntime>=1.15",
]
```

`torch` 已经在 `dependencies` 里,可直接用作 DocRect PyTorch 推理的后备。

---

## 7. 前端修改(`service/static/index.html`)

### 7.1 按钮

在「开始识别」按钮**左侧/上方**新增「增强图片」按钮:

```html
<div style="display:flex;gap:8px;">
  <button id="submitExamEnhanceBtn" disabled>增强图片</button>
  <button id="submitExamBtn" disabled>开始识别</button>
</div>
```

「增强图片」和「开始识别」独立启用:**任一图片上传后两者都启用**。

### 7.2 预览区

插入到「开始识别」按钮**下方**、状态提示**上方**:

```html
<div id="examEnhancePreview" style="display:none; margin-top:12px; padding:12px; background:#f9f9f9; border-radius:4px;">
  <h3>增强预览</h3>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
    <div><p>原图:</p><img id="enhanceOriginalImg" style="max-width:100%;border:1px solid #ddd;"></div>
    <div><p>增强图:</p><img id="enhanceEnhancedImg" style="max-width:100%;border:1px solid #ddd;"></div>
  </div>
  <p id="enhanceAppliedNote" style="font-size:12px;color:#666;"></p>
  <div style="display:flex;gap:8px;margin-top:8px;">
    <button id="useEnhancedBtn">用增强图识别</button>
    <button id="useOriginalBtn">跳过,用原图识别</button>
  </div>
</div>
```

### 7.3 JS 流程

- `submitExamEnhanceBtn.click` → 调 `/predict/exam/enhance` → 拿到 `task_id` → 轮询状态(复用现有轮询样板代码)
- `done` 时填两张 `<img>`,显示预览区;`applied_stages` 渲染到「增强预览」标题下方
- 「用增强图识别」:`fetch("data:...")` → `.blob()` → 作为 `file` 提交到 `/predict/exam`
- 「跳过,用原图识别」:直接用 `examFile` 提交到 `/predict/exam`
- 任一成功提交后,隐藏预览区、显示识别状态(沿用现有逻辑)

### 7.4 状态机

```
增强预览区: 隐藏 ──(点击增强)──► 加载中(spinner + "增强中...") ──► 完成(展示对比 + 两个按钮)
                       └──失败──► 错误提示(不阻断识别流程)
识别按钮:   始终启用(原图始终可识别),但用户可主动用增强图
```

---

## 8. 测试策略

### 8.1 单元测试(`tests/test_enhance.py`)

| 用例 | 输入 | 断言 |
|---|---|---|
| `test_edge_crop_finds_quad` | 合成:白底 + 黑四边形 + 桌面背景 | `stages.edge_crop.payload.applied == true`,输出是矫正后的图 |
| `test_edge_crop_skips_on_no_quad` | 纯文字图(无四边形) | `payload.applied == false`,`error == "NO_QUAD_FOUND"`,输出与输入相同 |
| `test_deskew_corrects_small_rotation` | 5° 旋转合成图 | 输出图主轴角度 < 1° |
| `test_deskew_skips_extreme` | 20° 旋转 | `applied == false` |
| `test_clahe_preserves_shape` | 任意图 | 输出 dtype/shape 与输入一致 |
| `test_dewarp_with_mock` | monkeypatch `apply_docrect` 模拟成功 | 输出 shape 与输入一致 |
| `test_dewarp_unavailable` | monkeypatch `apply_docrect` 抛异常 | `applied == false`,`error` 含 `"DOCRECT_UNAVAILABLE"` |

### 8.2 集成测试(`tests/test_enhance_api.py`)

| 用例 | 断言 |
|---|---|
| `test_enhance_round_trip` | POST `/predict/exam/enhance` → 拿到 task_id → 轮询直到 `done` → 断言 `final.original_image`、`final.enhanced_image`、`final.applied_stages` 都存在 |
| `test_enhance_without_docrect` | 不设 `DOCRECT_MODEL_PATH` 时:增强任务仍 `done`,但 `applied_stages` 不含 `dewarping` |
| `test_enhance_rejects_bad_mime` | 上传 `.txt` → 415 `BAD_MIME` |
| `test_enhance_handles_corrupt_image` | 上传损坏 jpeg → 任务 `failed`,`error` 含 `"decode failed"` |
| `test_enhance_to_recognize_e2e` | 增强 → 拿 `enhanced_bytes_b64` → 解码 → 上传 `/predict/exam` → 任务 `done` |

### 8.3 前端检查(Playwright 可选 / 手动冒烟)

- 「增强图片」按钮存在、初始 disabled、上传图后启用
- 点击 → 状态「处理中」 → 完成时预览区出现
- 预览区出现「用增强图识别」「跳过,用原图识别」两个按钮
- 点击「跳过,用原图识别」→ 走的还是上传的原始文件

### 8.4 性能基线(记录到 README)

在 1920×1080 测试图上的预期耗时(单阶段):
- `edge_crop` ≤ 200 ms
- `deskew` ≤ 100 ms
- `dewarping`(DocTr ONNX, CPU)≤ 8000 ms
- `clahe_enhance` ≤ 100 ms
- 整体 ≤ 10 s

---

## 9. 风险与回退

| 风险 | 缓解 |
|---|---|
| DocRect 模型下载困难 / 文件大 | 不强制依赖;缺模型时仅 dewarping 跳过,其余阶段仍生效 |
| DocRect CPU 推理慢 | 默认跑单张;后续可加并发(复用 `cfg.max_concurrent`) |
| 增强后比原图更糟 | 「跳过,用原图识别」始终可用;stages 表透明展示每阶段效果 |
| 引入 ONNX 依赖增大包体积 | ONNX Runtime CPU 包 ~10 MB;若仍嫌大,可仅 fallback 到 PyTorch |

---

## 10. 实施切片(给 writing-plans 阶段使用)

**Slice 1 — 后端骨架 + decode/edge_crop/deskew/clahe 四个纯 OpenCV 阶段**:
- `service/exam/enhance.py`(不含 DocRect)
- `service/app.py` 新增两个端点
- `service/exam/tasks.py` 支持 `enhance:` 前缀
- `service/config.py` 新增字段
- `start.bat` 检测
- 单元 + 集成测试

**Slice 2 — 前端预览区 + 「用增强图/原图识别」逻辑**:
- `service/static/index.html` 改造
- 前端冒烟测试

**Slice 3 — DocRect 集成(dewarping 阶段)**:
- DocRect 模型加载与懒加载
- ONNX 优先、PyTorch fallback
- 性能基线测试

三个切片可以顺序实施,Slice 1 完成后即可在无 DocRect 模型的情况下端到端跑通。