"""Question number detection via regex."""
from __future__ import annotations

import re

DEFAULT_QUESTION_REGEX = r"^\s*\(?\d+[\.\)](?!\d)"


def validate_regex(pattern: str) -> str:
    """Validate a Python regex. Raises ValueError if invalid or matches empty."""
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"invalid regex: {e}") from e
    if compiled.match(""):
        raise ValueError("regex matches empty string; would trivially match all boxes")
    return pattern


def is_question(text: str, regex: str = DEFAULT_QUESTION_REGEX) -> bool:
    """Return True if text starts with a question number per the regex."""
    if not text:
        return False
    return re.match(regex, text) is not None
