"""Tests for the mark_done_enhance task-store extension."""
from __future__ import annotations

from service.exam import tasks
from service.exam.schemas import EnhancementFinal


def test_mark_done_enhance_persists_final():
    tasks.reset_for_tests()
    tid = tasks.create_task()
    final = EnhancementFinal(
        original_image="data:image/jpeg;base64,AAA",
        enhanced_image="data:image/jpeg;base64,BBB",
        applied_stages=["edge_crop"],
        enhanced_bytes_b64="BBB",
        output_size={"width": 100, "height": 50},
    )
    tasks.mark_done_enhance(tid, final)

    state = tasks.get_task(tid)
    assert state is not None
    assert state["status"] == "done"
    assert state["final"] is final
    assert state["progress"] is None


def test_mark_done_enhance_noop_on_missing_task():
    tasks.reset_for_tests()
    # No create_task — store is empty
    tasks.mark_done_enhance("nonexistent", final=None)  # type: ignore[arg-type]
    assert tasks.get_task("nonexistent") is None