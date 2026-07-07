"""FastAPI application: routes, middleware, entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

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

    app.state.service_state = state
    return app


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    app = create_app()
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")