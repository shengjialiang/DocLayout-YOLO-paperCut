"""Pydantic response schemas for the HTTP service."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    id: int
    class_id: int
    class_name: str
    bbox_xyxy: list[float] = Field(..., description="[x1, y1, x2, y2] in pixel coords")
    bbox_xywh: list[float] = Field(..., description="[cx, cy, w, h]")
    score: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    width: int
    height: int
    annotated_image: str = Field(..., description="Data-URL: data:image/jpeg;base64,...")
    detections: list[Detection]
    num_detections: int
    inference_time_ms: int
    model_imgsz: int
    conf_threshold: float
    device: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    device: str
    model_loaded: bool
    max_concurrent: int
    model_path: str