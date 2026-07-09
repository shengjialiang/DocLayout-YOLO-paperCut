# 试卷 OCR 并行化 — Handoff 文档

> **场景**：6-Task OCR 并行化计划已全部完成并通过两层 review（spec compliance + code quality）。Final reviewer 给出 "Ready to merge"。本 handoff 记录已完成工作、commit 历史、关键决策、验证结果，供新 session 接力或直接 push / PR。

---

## 当前会话已完成

### 文档 commit（先生成 spec + plan）

- `6da9602` — `docs: add exam OCR parallelization design spec` → `docs/superpowers/specs/2026-07-09-exam-ocr-parallelization-design.md`
- `44551ce` — `docs: add exam OCR parallelization implementation plan` → `docs/superpowers/plans/2026-07-09-exam-ocr-parallelization.md`

### 实施 commit（6 个 task 全部 DONE）

| Commit | Task | 描述 |
|---|---|---|
| `413ab84` | 1 | `feat(exam): add process-wide OcrEngine singleton with double-checked locking` |
| `6a84d6d` | 2 | `refactor(exam): route OcrEngine.ocr_image through process-wide singleton` |
| `dfc1490` | 3 | `perf(exam): parallelize OCR stage with ThreadPoolExecutor (env-tunable workers)` |
| `b72fc6d` | 4 | `perf(service): bump MAX_CONCURRENT default from 1 to 2` |
| `68407eb6` | 5 | `test(exam): reset OCR engine singleton in tasks.reset_for_tests` |

Task 6（端到端验证）无 commit — 只跑了 pytest 验证。

### Base / head

- Base SHA（实施前）: `c33a3e4`（Task 1 父 commit；`git rev-parse 413ab84^` 等价）
- Head SHA: `68407eb652793c5a935f6b38e67885db70c54430`

---

## 文件改动清单

### 修改（4 个生产文件 + 3 个测试文件）

```
service/exam/ocr.py          # +get_engine() + reset_engine_singleton() + OcrEngine 改走单例
service/exam/pipeline.py     # +_ocr_one_box worker + Stage 3 改 ThreadPoolExecutor + MAX_OCR_WORKERS 常量
service/exam/tasks.py        # reset_for_tests 也调 reset_engine_singleton（延迟 import）
service/config.py            # MAX_CONCURRENT 默认 "1" → "2"
tests/exam/test_ocr.py       # +2 测试（单例/并发初始化）+ autouse fixture 复位
tests/exam/test_pipeline.py  # +2 测试（ThreadPoolExecutor 验证/顺序保持）
tests/test_config.py         # test_load_config_defaults 断言 == 1 → == 2
```

### 不变（per spec §8.3）

```
service/exam/expand.py、geometry.py、question_filter.py、schemas.py
service/app.py、inference.py
service/static/index.html
pyproject.toml
```

### 完整 diff 统计

`git diff --stat c33a3e4..68407eb6`：
- 4 个生产文件、3 个测试文件
- 净增 ~150 行（生产）+ ~150 行（测试）

---

## 关键决策（与 spec 完全对齐）

| 决策 | 选择 | 备注 |
|---|---|---|
| 并行模型 | ThreadPoolExecutor | PaddleOCR C++ 推理释放 GIL |
| OCR workers | 默认 2，env `MAX_OCR_WORKERS` | CPU-only 保守值；设 1 即串行 |
| 引擎单例 | 模块级 + 双检锁 | Task 1 |
| 请求级并发 | MAX_CONCURRENT 默认 2 | env 仍可调回 1 |
| 错误聚合 | `(OcrBlock, error_str)` 元组 | 不污染 Pydantic `OcrBlock` |
| 顺序保证 | `pool.map` 顺序产出 | Task 3 第二个测试验证 |
| 资源清理 | `with ThreadPoolExecutor(...) as pool` | `__exit__` 等待 workers |

### 实施时的 3 个 spec deviation（已验证必要且最小）

1. **Task 2 — `fake_load` 签名更新**（`tests/exam/test_ocr.py:50,73,90,201`）
   - 原因：`get_engine()` 现在调 `_load_paddleocr(lang=lang)`，原 `def fake_load()` 不再兼容
   - 修复：加 `lang="ch"` 默认参数；fake body 不动

2. **Task 3 — 测试 counter 改 class-level**（`tests/exam/test_pipeline.py:520,531`）
   - 原因：spec 写的 `self.n` 是 per-instance，但 `OcrEngine(lang="ch")` 每次 lambda 调用都返回新实例，counter 永远 = 0，测试无法验证顺序
   - 修复：class-level `_shared_n`，测试开头 reset 到 0

3. **Task 4 — `tests/test_config.py:39` 断言更新**
   - 原因：spec 假设"默认改动不影响单测"不准确；`test_load_config_defaults` 直接断言默认值
   - 修复：`== 1` → `== 2`

---

## 验证结果

| 验证项 | 结果 |
|---|---|
| 全套测试（默认配置） | **122 passed, 2 skipped** in 9.26s |
| 全套测试（`MAX_OCR_WORKERS=1 MAX_CONCURRENT=1` 回退） | **122 passed, 2 skipped** in 9.47s |
| `MAX_OCR_WORKERS=4`（激进） | pipeline 测试 11/11 通过 |
| Manual smoke（`第五题/第十/第十一题.jpg`） | 3 个请求 status=done；OCR 1.2-1.6s/张 |
| 2 skipped | 预存在的 E2E 测试（需 `MODEL_PATH`），与本次改动无关 |

**未验证**：30% 提速 — 3 张测试图都只有 1 个 OCR block，无并行化机会。需要多 block 测试图才能验证提速（不是代码问题，是测试数据限制）。

---

## 环境

- 分支：`main`（领先 origin/main 56 commits，全部本地，未推送）
- 工作区：除几张测试图（`第五题.jpg` 等）和 `start.bat` 外干净
- Python / pytest / paddleocr：项目原状不变

---

## 收尾选项

1. **直接 push 到 origin/main** — 5 个 commit + 2 个 doc commit 一起推
2. **建 PR** — 在 origin 建 feature/ocr-parallelization 分支后提 PR
3. **观望** — 保留本地，等更多回归数据或代码评审

收尾动作推荐 invoke `superpowers:finishing-a-development-branch` skill 处理。
