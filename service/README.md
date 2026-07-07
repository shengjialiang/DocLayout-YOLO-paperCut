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
