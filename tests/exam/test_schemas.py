"""Pydantic schema validation tests."""
import pytest
from pydantic import ValidationError

from service.exam.schemas import (
    Question,
    OcrBlock,
    FinalResult,
    TaskSubmitResponse,
    TaskStatusResponse,
)


def test_question_serialization():
    q = Question(
        index=1,
        bbox_xyxy=[127.0, 871.0, 1523.0, 1501.0],
        matched_text="1. (计算题) ...",
        source_detection_ids=[1, 6],
        ocr_score=0.93,
    )
    assert q.index == 1
    assert q.ocr_score == 0.93
    d = q.model_dump()
    assert d["matched_text"] == "1. (计算题) ..."


def test_ocr_block_with_is_question_flag():
    block = OcrBlock(
        source_detection_id=1,
        bbox_xyxy=[127.0, 871.0, 1523.0, 1501.0],
        text="1. 题",
        score=0.93,
        is_question=True,
    )
    assert block.is_question is True


def test_final_result_with_questions_and_blocks():
    final = FinalResult(
        annotated_image="data:image/jpeg;base64,abc",
        original_image="data:image/jpeg;base64,xyz",
        questions=[
            Question(
                index=1, bbox_xyxy=[0, 0, 100, 100],
                matched_text="1.", source_detection_ids=[0], ocr_score=0.9,
            )
        ],
        ocr_blocks=[
            OcrBlock(
                source_detection_id=0, bbox_xyxy=[0, 0, 100, 100],
                text="1.", score=0.9, is_question=True,
            )
        ],
    )
    assert len(final.questions) == 1
    assert len(final.ocr_blocks) == 1


def test_task_submit_response_required_fields():
    r = TaskSubmitResponse(task_id="abc123", status="queued")
    d = r.model_dump()
    assert d["task_id"] == "abc123"
    assert d["status"] == "queued"
    assert d["submit_url"] == "/predict/exam/abc123/status"


def test_task_status_response_running():
    r = TaskStatusResponse(
        task_id="abc", status="running", progress="ocr",
        stages={"yolo": {"duration_ms": 6500, "payload": {}, "error": None}},
        final=None, error=None,
    )
    d = r.model_dump()
    assert d["status"] == "running"
    assert d["stages"]["yolo"]["duration_ms"] == 6500


def test_invalid_bbox_xyxy_wrong_length():
    with pytest.raises(ValidationError):
        Question(
            index=1, bbox_xyxy=[0, 0, 100],
            matched_text="x", source_detection_ids=[], ocr_score=0.0,
        )