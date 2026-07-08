"""Question number detection via regex + OCR confusion normalisation."""
from __future__ import annotations

import re

# Matches: "1.", "1)", "1。", "1、", "1．" (fullwidth period), "(1)", etc.
DEFAULT_QUESTION_REGEX = r"^\s*\(?\d+[\.。．、\)](?!\d)"

# Map of OCR-confused characters that should be '.' in a leading-digit context.
# Includes fullwidth period ．
_QUESTION_PUNCT_NORMALIZE = re.compile(
    r'^(\s*\(?\d+)[。，．、：,;:](?!\d)',
)


def normalize_question_number(text: str) -> str:
    """Replace OCR-common punctuation errors in leading-digit patterns.

    ``1。`` → ``1.``, ``2，`` → ``2.``, ``3、 `` → ``3.``, etc.
    Does not touch body text punctuation.
    """
    return _QUESTION_PUNCT_NORMALIZE.sub(r'\1.', text)


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
