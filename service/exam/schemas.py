"""Pydantic models for exam pipeline."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Question(BaseModel):
    """Final extended question box."""
    index: int
    bbox_xyxy: list[float] = Field(..., min_length=4, max_length=4)
    matched_text: str
    source_detection_ids: list[int]
    ocr_score: float


class OcrBlock(BaseModel):
    """OCR result for a single cropped plain text box."""
    source_detection_id: int
    bbox_xyxy: list[float] = Field(..., min_length=4, max_length=4)
    text: str
    score: float
    is_question: bool
    crop_image: str = ""   # base64 JPEG of the cropped region


class FinalResult(BaseModel):
    """Pipeline final output."""
    annotated_image: str = Field(..., description="Data-URL: data:image/jpeg;base64,...")
    original_image: str
    yolo_annotated: str = ""   # YOLO detection image (before OCR)
    questions: list[Question]
    ocr_blocks: list[OcrBlock]


class StageResult(BaseModel):
    """Per-stage execution record."""
    model_config = ConfigDict(protected_namespaces=())
    started_at: float | None = None
    finished_at: float | None = None
    duration_ms: int | None = None
    payload: dict | None = None
    error: str | None = None


class TaskSubmitResponse(BaseModel):
    """Returned immediately after POST /predict/exam."""
    task_id: str
    status: Literal["queued"]
    submit_url: str | None = None

    @model_validator(mode="after")
    def _populate_submit_url(self) -> "TaskSubmitResponse":
        if self.submit_url is None:
            object.__setattr__(self, "submit_url", self.make_submit_url(self.task_id))
        return self

    @classmethod
    def make_submit_url(cls, task_id: str) -> str:
        return f"/predict/exam/{task_id}/status"


class TaskStatusResponse(BaseModel):
    """Returned by GET /predict/exam/{task_id}/status."""
    model_config = ConfigDict(protected_namespaces=())
    task_id: str
    status: Literal["queued", "running", "done", "failed", "expired"]
    progress: str | None = None
    stages: dict[str, StageResult]
    final: FinalResult | None = None
    error: str | None = None

    @classmethod
    def make_submit_url(cls, task_id: str) -> str:
        return f"/predict/exam/{task_id}/status"