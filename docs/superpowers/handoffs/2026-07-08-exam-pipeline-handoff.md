# 试卷流水线实施 — Handoff 文档

> **场景**：当前会话由于 context 限制无法继续执行 16-Task 计划。本文档让新 session 从 Task 4 开始无缝接力。

---

## 当前会话已完成

### Task 1（commit `3c99c38`）
- 修改 `pyproject.toml`
- 添加 `[exam]` extras（`paddleocr>=2.7`、`paddlepaddle>=2.6`）

### Task 2（commit `3eb4590`）
- 新增 `service/exam/__init__.py`
- 新增 `service/exam/schemas.py` — Pydantic 模型
- 新增 `tests/exam/__init__.py`
- 新增 `tests/exam/test_schemas.py` — 6 测试通过

**重要**: Task 2 的实现对 spec 做了两处必要调整：
1. `StageResult.started_at` 设为 Optional（spec 测试省略了它）
2. `TaskSubmitResponse.submit_url` 设为 Optional，使用 `@model_validator(mode="after")` 在构造时根据 `task_id` 自动填充（保持 `make_submit_url` classmethod 不变）

### Task 3（commit `109cd47`）
- 新增 `service/exam/expand.py` — 框放大（pixel + ratio + 4 方向 + 越界 clamp）
- 新增 `tests/exam/test_expand.py` — 3 测试通过

**当前测试总数**: 9 passed

---

## 项目当前状态

```
分支：main（领先 origin/main 18 commits，全部本地，未推送）
工作区：clean（除临时 predict_result.json 在 .gitignore 之外）
Python：3.11.9（PyTorch 2.12.1+cpu）
pytest：8.3.3（已装）
依赖：fastapi 0.115、uvicorn 0.30、huggingface_hub 1.22 已装
      paddleocr / paddlepaddle 未装（仅声明在 [exam] extras）
```

---

## 接下来要做的

直接执行 plan 中的 Task 4 开始。Plan 路径：

```
docs/superpowers/plans/2026-07-08-doclayout-yolo-exam-pipeline.md
```

完整 spec：

```
docs/superpowers/specs/2026-07-08-doclayout-yolo-exam-pipeline-design.md
```

---

## 推荐新 session 的开场白

```
我正在用 superpowers:executing-plans 继续执行之前 brainstorm 阶段生成的
实施计划。计划文件在 docs/superpowers/plans/2026-07-08-doclayout-yolo-exam-pipeline.md。
Handoff 文档在 docs/superpowers/handoffs/2026-07-08-exam-pipeline-handoff.md。

请从 Task 4 开始执行（Tasks 1-3 已完成），采用：
- Subagent-Driven：每个 Task 一个 implementer subagent + spec reviewer + code quality reviewer
- TDD：先写失败测试，再实现
- 频繁提交（每个 Task 一个 commit）
```

---

## 关键依赖与模式

- 现有 service 用 `model.predict_one(...)` 做推理（已存在 `service/inference.py`）
- 现有 Pydantic 用 `model_config = ConfigDict(protected_namespaces=())`
- pytest 配置在 `pyproject.toml` 中含 `pythonpath = ["."]`
- Task 11 会重构 `pipeline.run_pipeline` 签名（添加 `model` 和 `semaphore` kwarg）

## 注意事项

1. **Windows 路径**：用 Git Bash，路径用 `/d/aspirecn/陕西/...` 或 `D:/aspirecn/陕西/...`
2. **后台服务**：之前启动的 `python -m service.app`（task id `b77iflvzi`）可能仍在运行。如不需可停止。
3. **Task 12 的 run_in_executor 调用**：plan 已修复，使用 `functools.partial(run_pipeline, ..., model=model, semaphore=sem)` 因为 `run_pipeline` 的 model 和 semaphore 是 keyword-only。
4. **PaddleOCR 模型**：首次调用时下载（约 150MB），懒加载不影响现有 /predict。
5. **服务不会自动重启**：Task 12 改 `service/app.py` 后需手动重启服务（`TaskStop` 旧的 + 重新启动）。

## 已完成的 TaskCreate 跟踪

session 中创建的 TaskCreate ID：
- #25 Task 1 ✅
- #17 Task 2 ✅
- #20 Task 3 ✅
- 等等（详见 TaskList 输出）
