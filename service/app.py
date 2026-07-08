"""FastAPI application: routes, middleware, entrypoint."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from service.config import Config, load_config
from service.exam import tasks as exam_tasks
from service.exam.pipeline import run_pipeline
from service.exam.question_filter import validate_regex
from service.inference import load_model
from service.schemas import HealthResponse, PredictResponse

logger = logging.getLogger("doclayout_service")


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


def create_app() -> FastAPI:
    """Build the FastAPI app. Loads model on lifespan startup."""
    cfg = load_config()
    state: dict[str, Any] = {
        "model": None,
        "config": cfg,
        "semaphore": asyncio.Semaphore(cfg.max_concurrent),
    }

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

        # Read & size check
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

        # Decode data URL -> bytes
        b64 = result["annotated_base64"].split(",", 1)[1]
        import base64
        img_bytes = base64.b64decode(b64)
        return Response(content=img_bytes, media_type="image/jpeg")

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

    @app.get("/", include_in_schema=False)
    def root():
        index_path = Path(__file__).parent / "static" / "index.html"
        return FileResponse(
            index_path,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/predict/exam", responses={
        413: {"description": "File too large"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Validation error"},
        503: {"description": "Service not ready"},
    })
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
        """Submit a new exam processing task. Returns task_id immediately."""
        # Convert and validate numeric params
        try:
            expand_top_f = float(expand_top)
            expand_bottom_f = float(expand_bottom)
            expand_left_f = float(expand_left)
            expand_right_f = float(expand_right)
        except (TypeError, ValueError):
            return JSONResponse(
                status_code=422,
                content={"detail": "expand_* must be numeric", "error_code": "INVALID_PARAM"},
            )

        err = _validate_exam_params(
            expand_mode, expand_top_f, expand_bottom_f,
            expand_left_f, expand_right_f, question_regex,
        )
        if err:
            return JSONResponse(status_code=422, content={"detail": err, "error_code": err})

        # MIME check
        if not (file.content_type or "").startswith("image/"):
            return JSONResponse(
                status_code=415,
                content={"detail": "unsupported media type", "error_code": "BAD_MIME"},
            )

        raw = await file.read()
        if len(raw) > cfg.max_file_size_mb * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"file too large, max {cfg.max_file_size_mb}MB",
                    "error_code": "FILE_TOO_LARGE",
                },
            )

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
            return JSONResponse(
                status_code=503,
                content={"detail": "model not ready", "error_code": "NOT_READY"},
            )

        # Fire-and-forget background task
        asyncio.create_task(
            _run_exam_pipeline_async(task_id, raw, params, model, sem)
        )

        from service.exam.schemas import TaskSubmitResponse
        return TaskSubmitResponse(
            task_id=task_id,
            status="queued",
            submit_url=f"/predict/exam/{task_id}/status",
        )

    @app.get("/predict/exam/{task_id}/status")
    async def exam_status(task_id: str):
        state_d = exam_tasks.get_task(task_id)
        if state_d is None:
            return JSONResponse(
                status_code=404,
                content={
                    "detail": "task not found or expired",
                    "error_code": "TASK_NOT_FOUND",
                },
            )
        from service.exam.schemas import StageResult, TaskStatusResponse
        stages = {
            k: StageResult(
                started_at=0.0,
                finished_at=0.0,
                duration_ms=v.get("duration_ms"),
                payload=v.get("payload"),
                error=v.get("error"),
            )
            for k, v in state_d["stages"].items()
        }
        return TaskStatusResponse(
            task_id=state_d["task_id"],
            status=state_d["status"],
            progress=state_d.get("progress"),
            stages=stages,
            final=state_d["final"],
            error=state_d.get("error"),
        )

    app.state.service_state = state
    return app


async def _run_exam_pipeline_async(task_id, image_bytes, params, model, sem):
    """Run pipeline in background thread (OCR is CPU-bound, blocking)."""
    import asyncio
    from functools import partial
    loop = asyncio.get_event_loop()
    # run_pipeline signature requires keyword-only model= and semaphore=;
    # functools.partial binds them so run_in_executor can call positionally.
    await loop.run_in_executor(
        None,
        partial(run_pipeline, task_id, image_bytes, params, model=model, semaphore=sem),
    )


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    app = create_app()
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")