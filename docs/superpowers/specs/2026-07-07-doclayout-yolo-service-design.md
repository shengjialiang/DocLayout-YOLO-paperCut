# DocLayout-YOLO HTTP 服务 — 设计文档

**作者:** Claude
**日期:** 2026-07-07
**状态:** 已批准 — 等待实现

---

## 目标

将 DocLayout-YOLO 部署为本地 HTTP 服务,客户端通过浏览器或 HTTP 请求上传图片,服务返回:

1. **标注后的图片**(jpeg/png),图片上的每个文档元素(标题/正文/表格/图片等)带有彩色检测框和类别标签
2. **结构化 JSON 布局数据**,包含每个检测区域的类别、坐标、置信度

支撑个人/小团队本地使用,无需 GPU,可在一台普通笔记本上运行。

---

## 部署场景

- **场景:** 个人 / 小团队本地
- **并发:** 低(单用户偶尔上传几张图片)
- **硬件:** 支持 CPU、Apple Silicon MPS、CUDA GPU(自适应)
- **规模:** 单进程 FastAPI + uvicorn,1 个 worker

---

## 架构

```
浏览器 / curl / 客户端
        │  HTTP
        ▼
┌─────────────────────────────────────┐
│  FastAPI + uvicorn (单进程)         │
│                                     │
│  启动期加载:                          │
│   • YOLOv10 模型 (全局单例)          │
│   • asyncio.Semaphore(N)            │
│                                     │
│  路由:                              │
│   GET  /              → 上传页面    │
│   POST /predict       → JSON        │
│   POST /predict/image → 图片 bytes  │
│   GET  /health        → 健康检查    │
└─────────────────────────────────────┘
        │  模型推理
        ▼
    GPU / MPS / CPU
```

### 目录布局

```
DocLayout-YOLO/
├── service/
│   ├── app.py              # FastAPI 主程序
│   └── static/
│       └── index.html      # 上传 UI(纯 HTML + fetch API)
├── tests/
│   └── test_service.py     # mock 单测
├── demo.py                 # 保留原样
├── doclayout_yolo/         # 不修改
└── pyproject.toml          # 新增可选依赖 [service]
```

### 关键技术决策

| 项目 | 决策 |
|---|---|
| Web 框架 | FastAPI |
| ASGI 服务器 | uvicorn |
| 模型单例 | 启动期 `app.on_event("startup")` 加载一次 |
| 并发控制 | `asyncio.Semaphore`,默认 `1`(GPU 独占);`MAX_CONCURRENT` 环境变量覆盖 |
| 设备选择 | CUDA → MPS → CPU 顺序自动检测;`DEVICE` 环境变量可强制指定 |
| 默认模式 | CPU 友好(无 GPU 也能完整运行);不假设 GPU 内存、pin memory、CUDA streams |
| 前端 | 静态 HTML + `fetch()` 调用,无构建步骤 |

### 依赖新增(`pyproject.toml` 的 `[service]` extras)

```toml
[project.optional-dependencies]
service = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.23",
    "python-multipart>=0.0.6",
    "httpx>=0.24",  # 仅测试需要
]
```

启动时:`pip install -e ".[service]"`。原 `pip install doclayout-yolo` 不变,服务为可选附加。

---

## 组件与接口

### 核心组件(`service/app.py`)

```
1. 配置加载
     • 从环境变量读取: MODEL_PATH, DEVICE, HOST, PORT, MAX_CONCURRENT, MAX_FILE_SIZE_MB
2. 模型管理
     • load_model() — 启动时同步加载;失败则 uvicorn 退出
     • model — 全局单例
     • semaphore — asyncio.Semaphore(MAX_CONCURRENT)
3. 推理执行
     • predict_one(image_bgr, conf, imgsz, line_width, font_size) -> dict
       ├─ async with semaphore(超时 60s)
       ├─ t0 = time.perf_counter()
       ├─ det_res = model.predict(image_bgr, imgsz=imgsz, conf=conf, device=DEVICE)
       ├─ annotated = det_res[0].plot(pil=True, line_width=line_width, font_size=font_size)
       ├─ detections = parse_results(det_res[0])  -> [{class_id, class_name, bbox_xyxy, bbox_xywh, score}]
       ├─ encoded = base64_encode_jpeg(annotated)
       └─ return {detections, annotated_base64, inference_time_ms, ...}
4. 路由
     • GET  /              -> FileResponse(static/index.html)
     • POST /predict       -> PredictResponse(JSON)
     • POST /predict/image -> Response(image/jpeg bytes)
     • GET  /health        -> JSON
5. 异常处理
     • 全局 ExceptionHandler 兜底,返回 {"detail": ..., "error_code": ...}
```

### API 详细规范

#### `POST /predict`

- **Content-Type:** `multipart/form-data`
- **表单字段:**
  | 字段 | 必填 | 默认 | 范围 | 说明 |
  |---|---|---|---|---|
  | `file` | 是 | — | ≤ 20MB,`image/*` MIME | 上传的图片 |
  | `conf` | 否 | `0.2` | `[0, 1]` | 置信度阈值 |
  | `imgsz` | 否 | `1024` | `[320, 2048]` | 推理尺寸(像素) |
  | `line_width` | 否 | `5` | `[1, 20]` | 标注框线宽 |
  | `font_size` | 否 | `20` | `[8, 50]` | 标签字号 |
- **响应:** `application/json`
  ```json
  {
    "width": 1240,
    "height": 1754,
    "annotated_image": "data:image/jpeg;base64,/9j/4AAQ...",
    "detections": [
      {
        "id": 0,
        "class_id": 0,
        "class_name": "title",
        "bbox_xyxy": [100.0, 200.0, 500.0, 250.0],
        "bbox_xywh": [300.0, 225.0, 400.0, 50.0],
        "score": 0.95
      }
    ],
    "num_detections": 8,
    "inference_time_ms": 234,
    "model_imgsz": 1024,
    "conf_threshold": 0.2,
    "device": "cpu"
  }
  ```

#### `POST /predict/image`

- **输入:** 同 `/predict`
- **响应:** 直接 `image/jpeg`(标注图字节),便于 `curl -o result.jpg`

#### `GET /`

- **响应:** `text/html`,`static/index.html`

#### `GET /health`

- **响应:**
  ```json
  {"status": "ok", "device": "cpu", "model_loaded": true, "max_concurrent": 1, "model_path": "/path/to/model.pt"}
  ```

### 前端 UI(`service/static/index.html`)

纯 HTML + JavaScript,无构建步骤。布局:

```
┌─────────────────────────────────────────────────────┐
│           DocLayout-YOLO 检测服务                    │
├─────────────────────────────────────────────────────┤
│  [拖拽上传 / 点击选择]  支持 jpg/png/webp,≤20MB      │
│                                                     │
│  检测参数:                                          │
│    conf 阈值:  [======●====] 0.2                     │
│    imgsz:      [====●======] 1024                    │
│    框线粗细:   [==●========] 5                       │
│    标签字号:   [======●====] 20                      │
│                                                     │
│           [ 开始检测 ]                              │
├─────────────────────────────────────────────────────┤
│  推理中...(CPU 模式下预计 5-15 秒)                  │
├─────────────────────────────────────────────────────┤
│  检测结果: 8 个区域,耗时 234ms                      │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐               │
│  │  标注图      │    │  原图        │               │
│  └──────────────┘    └──────────────┘               │
│                                                     │
│  [ 下载标注图 ]                                    │
│                                                     │
│  检测到的区域:                                       │
│   #  类别   置信度    坐标 (x1, y1, x2, y2)         │
│   1  title  0.95    [100, 200, 500, 250]            │
│   2  text   0.87    [...]                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**实现要点:**
- 默认参数硬编码(从后端默认值兜底,即使接口参数变化也能工作)
- 用 `<output>` 显示当前滑块值
- 检测结果用 `<img src="data:image/jpeg;base64,..."` 直接渲染
- 检测列表用 `<table>` 输出

---

## 数据流

```
浏览器                        FastAPI                    YOLOv10 模型
  │                              │                              │
  │  POST /predict (multipart)   │                              │
  │  file=exam.jpg               │                              │
  │  conf=0.2, imgsz=1024        │                              │
  │ ──────────────────────────► │                              │
  │                              │ 1. 接收 + 校验文件           │
  │                              │ 2. cv2.imdecode(BGR)         │
  │                              │ 3. await semaphore           │
  │                              │ ───────────────────────────►│
  │                              │ 4. model.predict(...) 同步   │
  │                              │ ◄───────────────────────────│
  │                              │ 5. plot() 生成 PIL 标注图    │
  │                              │ 6. 解析 boxes -> JSON       │
  │                              │ 7. base64 编码图            │
  │                              │ 8. 构造 JSON 响应           │
  │                              │ 9. release semaphore        │
  │  HTTP 200 + JSON             │                              │
  │ ◄────────────────────────── │                              │
  │                              │                              │
  │  JS 渲染 base64 图 + 列表   │                              │
```

---

## 错误处理

| 错误场景 | HTTP 状态码 | 响应 |
|---|---|---|
| 缺 `file` 字段 | 422 | `{"detail": "file is required", "error_code": "MISSING_FILE"}` |
| 文件 > 20MB | 413 | `{"detail": "file too large, max 20MB", "error_code": "FILE_TOO_LARGE"}` |
| MIME 不是 `image/*` | 415 | `{"detail": "unsupported media type", "error_code": "BAD_MIME"}` |
| 图片损坏 / cv2 解码失败 | 400 | `{"detail": "failed to decode image: ...", "error_code": "DECODE_ERROR"}` |
| `imgsz` 超出 `[320, 2048]` | 422 | `{"detail": "imgsz must be in [320, 2048]", "error_code": "INVALID_PARAM"}` |
| `conf` 不在 `[0, 1]` | 422 | `{"detail": "conf must be in [0, 1]", "error_code": "INVALID_PARAM"}` |
| semaphore 等待超时(> 60s) | 503 | `{"detail": "service busy, try again later", "error_code": "BUSY"}` |
| 推理 OOM | 500 | `{"detail": "out of memory, try smaller imgsz", "error_code": "OOM"}` |
| 任何未捕获异常 | 500 | `{"detail": "internal error", "error_code": "INTERNAL"}` |
| 模型未加载(启动竞态) | 503 | `{"detail": "model not ready", "error_code": "NOT_READY"}` |

**实现要点:**
- 统一 JSON 错误格式 `{"detail": str, "error_code": str}`
- 启动失败:`MODEL_PATH` 未设置或加载失败 → uvicorn 直接报错退出,不静默
- `/health` 区分启动期(503)和运行期(200)
- 设置请求体大小上限(`uvicorn` `--limit-max-requests` + `python-multipart` 校验)
- 日志:`uvicorn` 默认 access log + 推理耗时单独打印

### 启动失败模式

| 场景 | 行为 |
|---|---|
| `MODEL_PATH` 未设置 | 启动报错:`Set MODEL_PATH env var to your .pt file` |
| 模型文件不存在 | 启动报错:`model file not found: <path>` |
| 加载模型抛异常 | 启动报错,日志含完整 traceback |
| 端口被占用 | uvicorn 自然抛 `OSError`,用户提示换 `PORT` |

### 性能预期

| 设备 | 单张推理时间(imgsz=1024) |
|---|---|
| NVIDIA GPU(RTX 3060) | 0.2 ~ 0.5 秒 |
| Apple Silicon Mac(MPS) | 0.5 ~ 1 秒 |
| 普通笔记本 CPU(8 核) | 5 ~ 15 秒 |

---

## 测试与验证

### 1. 单元测试(`tests/test_service.py`)

**目标:** 验证业务逻辑,不依赖真实模型权重

- 使用 `unittest.mock` patch 掉 `YOLOv10`,返回固定结果
- 使用 `httpx.AsyncClient` + FastAPI 的 `TestClient`
- 用例:
  - `GET /health` 返回 200,JSON 含 `device`、`model_loaded`
  - `POST /predict` 不带 `file` → 422
  - `POST /predict` 带文本文件 → 415
  - `POST /predict` 超过 20MB 的 fake 文件 → 413
  - `POST /predict` 上传 1×1 假 PNG 触发 cv2 成功路径 → 200,JSON 含 `detections`、`num_detections`
  - `POST /predict` 损坏图片字节 → 400
  - `POST /predict` 带 `conf=2.0` → 422
  - `POST /predict` 带 `imgsz=100` → 422
  - `POST /predict/image` 正常上传 → `image/jpeg` body 非空
  - 5 个并发 `POST /predict` → 全部 200,semaphore 工作正常

### 2. 集成验证(开发机手工)

启动真实服务,用 `assets/example/` 中的样本图片:

- [ ] `pip install -e ".[service]"` 成功
- [ ] `MODEL_PATH=path/to/doclayout_yolo_docstructbench_imgsz1024.pt python service/app.py` 启动成功
- [ ] 启动日志显示 `model loaded, device=cpu` 或 `device=cuda`
- [ ] `curl http://localhost:8000/health` 返回 `model_loaded: true`
- [ ] `curl -F "file=@assets/example/0.jpg" -F "conf=0.2" http://localhost:8000/predict` 返回 JSON
- [ ] JSON 中 `detections` 数组里的 `class_name` 与 README 中 DocStructBench 类别对应
- [ ] `curl -F "file=@assets/example/0.jpg" -F "conf=0.2" -o result.jpg http://localhost:8000/predict/image` 得到标注图,可用图片查看器打开
- [ ] 浏览器打开 `http://localhost:8000`,拖入图片,得到正确检测结果
- [ ] UI 中调 `conf` 滑块从 0.2 调到 0.5,`num_detections` 减少
- [ ] Ctrl-C 终止后端口立即释放

### 3. 验证策略要点

- **CI 不跑真实模型** — 模型权重 ~100MB,CI 慢且不稳。CI 只跑 `test_service.py` 的 mock 用例
- **`demo.py` 完全独立** — 服务只是 `YOLOv10` 的新入口,不修改 `model.predict()` 行为
- **回归保护:** 任何对 `doclayout_yolo/` 源码的修改都应保证 `demo.py` 仍可工作

---

## 风险与权衡

| 风险 | 影响 | 缓解 |
|---|---|---|
| CPU 推理慢 | 用户等待时间长 | UI 显示"预计 5-15 秒"、有 loading 提示 |
| 多次上传大图导致内存压力 | OOM | 文件大小限制 + 并发限制 + 推理后立即释放 |
| 模型初次加载慢 | 启动 5-10s | `/health` 区分启动期,前端逻辑等待加载完成 |
| 浏览器缓存旧 HTML | 改 HTML 后用户看不到更新 | `<meta charset>` + 文件名加版本号或 `Cache-Control: no-store` |
| base64 嵌入大图使 JSON 过大 | 网络传输慢 | 标注图用 jpeg quality=85 压缩,典型 < 500KB;可选将来改为 `/predict/image` 直接拿图 |

---

## 范围之外(YAGNI)

为保持首版简洁,以下功能**不在本期实现**:

- 多文件批量上传
- PDF 输入(只支持图片)
- 用户认证 / API key
- 请求历史持久化
- 多模型管理(动态切换不同权重)
- WebSocket 实时进度推送
- HTTPS(本地 HTTP 即可)
- Docker/Kubernetes 部署
- 速率限制(个人用不上)

后续如需要,可在不破坏现有接口的前提下增量添加。
