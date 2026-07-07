"""FastAPI application: routes, middleware, entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from service.config import Config, load_config
from service.inference import load_model
from service.schemas import HealthResponse

logger = logging.getLogger("doclayout_service")


def create_app() -> FastAPI:
    """Build the FastAPI app. Loads model on lifespan startup."""
    cfg = load_config()
    state: dict[str, Any] = {"model": None, "config": cfg}

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
        import asyncio
        sem = asyncio.Semaphore(cfg.max_concurrent)
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

    app.state.service_state = state
    return app


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    app = create_app()
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")