"""Task store tests: CRUD, stage updates, LRU eviction, TTL expiration."""
import time

import pytest

from service.exam import tasks
from service.exam.schemas import FinalResult, Question, OcrBlock


@pytest.fixture(autouse=True)
def reset_store():
    """Clear the in-memory store between tests."""
    tasks._store.clear()
    yield
    tasks._store.clear()


def test_create_task_returns_id_and_queued_state():
    tid = tasks.create_task()
    assert isinstance(tid, str) and len(tid) > 0
    state = tasks.get_task(tid)
    assert state is not None
    assert state["status"] == "queued"
    assert state["stages"] == {}
    assert state["final"] is None


def test_get_missing_task_returns_none():
    assert tasks.get_task("nonexistent") is None


def test_update_stage_records_duration():
    tid = tasks.create_task()
    tasks.update_stage(tid, "yolo", duration_ms=1234, payload={"num_detections": 5})
    state = tasks.get_task(tid)
    assert state["status"] == "running"
    assert "yolo" in state["stages"]
    assert state["stages"]["yolo"]["duration_ms"] == 1234
    assert state["stages"]["yolo"]["payload"] == {"num_detections": 5}


def test_update_stage_with_error_does_not_fail_task():
    tid = tasks.create_task()
    tasks.update_stage(tid, "ocr", duration_ms=500, error="PaddleOCR failed")
    state = tasks.get_task(tid)
    # Spec: intermediate errors don't fail the task
    assert state["status"] == "running"
    assert state["stages"]["ocr"]["error"] == "PaddleOCR failed"


def test_mark_done_writes_final_and_status_done():
    tid = tasks.create_task()
    final = FinalResult(
        annotated_image="data:img",
        original_image="data:orig",
        questions=[],
        ocr_blocks=[],
    )
    tasks.mark_done(tid, final)
    state = tasks.get_task(tid)
    assert state["status"] == "done"
    assert state["final"] == final


def test_mark_failed_writes_error_and_status_failed():
    tid = tasks.create_task()
    tasks.mark_failed(tid, "boom")
    state = tasks.get_task(tid)
    assert state["status"] == "failed"
    assert state["error"] == "boom"


def test_get_task_serializable_dict():
    tid = tasks.create_task()
    tasks.update_stage(tid, "yolo", duration_ms=100, payload={"k": "v"})
    state = tasks.get_task(tid)
    # Ensure no internal references / Pydantic models in serializable form
    assert isinstance(state, dict)
    assert isinstance(state["stages"], dict)


def test_lru_eviction_when_over_max(monkeypatch):
    monkeypatch.setattr(tasks, "MAX_TASKS", 3)
    tids = [tasks.create_task() for _ in range(4)]
    # The oldest (tids[0]) should be evicted
    assert tasks.get_task(tids[0]) is None
    for tid in tids[1:]:
        assert tasks.get_task(tid) is not None


def test_lru_access_promotes_to_most_recent(monkeypatch):
    monkeypatch.setattr(tasks, "MAX_TASKS", 3)
    t1 = tasks.create_task()
    t2 = tasks.create_task()
    t3 = tasks.create_task()
    # Access t1 to promote it
    tasks.get_task(t1)
    # Adding t4 should evict t2 (now the oldest)
    t4 = tasks.create_task()
    assert tasks.get_task(t1) is not None
    assert tasks.get_task(t2) is None
    assert tasks.get_task(t3) is not None
    assert tasks.get_task(t4) is not None


def test_ttl_expiration(monkeypatch):
    monkeypatch.setattr(tasks, "TASK_TTL_SECONDS", 0.1)
    tid = tasks.create_task()
    assert tasks.get_task(tid) is not None
    time.sleep(0.2)
    assert tasks.get_task(tid) is None