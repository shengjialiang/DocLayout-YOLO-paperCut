"""In-memory task store with LRU eviction and TTL expiration."""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from typing import Any

from service.exam.schemas import FinalResult

# Tunables
MAX_TASKS = 64
TASK_TTL_SECONDS = 600  # 10 minutes

# OrderedDict preserves insertion order; we move-to-end on access for LRU semantics.
_store: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _now() -> float:
    return time.time()


def _purge_expired() -> None:
    """Remove tasks older than TASK_TTL_SECONDS."""
    cutoff = _now() - TASK_TTL_SECONDS
    expired = [tid for tid, s in _store.items() if s["updated_at"] < cutoff]
    for tid in expired:
        _store.pop(tid, None)


def create_task() -> str:
    """Create a new task and return its ID."""
    _purge_expired()
    tid = uuid.uuid4().hex[:12]
    now = _now()
    _store[tid] = {
        "task_id": tid,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "stages": {},
        "final": None,
        "error": None,
        "progress": None,
    }
    _store.move_to_end(tid)
    _evict_if_needed()
    return tid


def get_task(task_id: str) -> dict[str, Any] | None:
    """Return task state as dict, or None if missing/expired."""
    _purge_expired()
    state = _store.get(task_id)
    if state is None:
        return None
    if _now() - state["updated_at"] > TASK_TTL_SECONDS:
        _store.pop(task_id, None)
        return None
    _store.move_to_end(task_id)
    return dict(state)


def update_stage(
    task_id: str,
    stage: str,
    *,
    duration_ms: int,
    payload: dict | None = None,
    error: str | None = None,
) -> None:
    """Record a stage's completion. Intermediate errors don't fail the task."""
    state = _store.get(task_id)
    if state is None:
        return
    now = _now()
    state["stages"][stage] = {
        "duration_ms": duration_ms,
        "payload": payload,
        "error": error,
    }
    state["status"] = "running"
    state["progress"] = stage
    state["updated_at"] = now
    _store.move_to_end(task_id)
    _evict_if_needed()


def mark_done(task_id: str, final: FinalResult) -> None:
    """Mark task as done with final result."""
    state = _store.get(task_id)
    if state is None:
        return
    state["final"] = final
    state["status"] = "done"
    state["progress"] = None
    state["updated_at"] = _now()
    _store.move_to_end(task_id)


def mark_failed(task_id: str, error: str) -> None:
    """Mark task as failed with error message."""
    state = _store.get(task_id)
    if state is None:
        return
    state["error"] = error
    state["status"] = "failed"
    state["progress"] = None
    state["updated_at"] = _now()
    _store.move_to_end(task_id)


def _evict_if_needed() -> None:
    """LRU eviction: drop oldest entries if over MAX_TASKS."""
    while len(_store) > MAX_TASKS:
        _store.popitem(last=False)


def reset_for_tests() -> None:
    """Clear all tasks (test helper)."""
    _store.clear()