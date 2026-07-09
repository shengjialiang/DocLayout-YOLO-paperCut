"""Question number detection via regex + OCR confusion normalisation."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

# Matches: "1.", "10.", optionally preceded by whitespace.
# Note: OCR outputs that use Chinese punctuation (。" etc.) are normalized to
# '.' by ``normalize_question_number`` before this regex is applied, so
# callers don't need to extend the regex for that case.
DEFAULT_QUESTION_REGEX = r"^\s*\d+\."

# Map of OCR-confused characters that should be '.' in a leading-digit context.
# Includes fullwidth period ．
_QUESTION_PUNCT_NORMALIZE = re.compile(
    r'^(\s*\(?\d+)[。，．、：,;:](?!\d)',
)

# Extracts the leading integer from a question-number line.
# Used by the post-filter step that reclassifies "题目框" whose question
# numbers don't follow top-to-bottom order (catches OCR misreadings like
# "0" for "10", "1" for "11", etc.).
#
# The optional ``\(?`` mirrors the question regex: the number line can be
# either ``"1. ..."`` or ``"(2) ..."`` (or ``"（3）..."`` with fullwidth
# paren). When the leading char is a paren, we skip past it to reach the
# digits. This matches the spec's "从第一位开始直到非数字" intent for the
# common case while still being usable for paren-style question numbers.
_LEADING_DIGITS = re.compile(r'^\s*[\(（]?(\d+)')

if TYPE_CHECKING:
    from service.exam.schemas import OcrBlock


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
    """Return True if text starts with a question number per the regex.

    The matched portion must not be immediately followed by a digit:
    this rejects sub-numbered items like ``"1.5 ..."`` whose leading
    digits belong to a deeper-numbered child, not a top-level question.
    The default regex already encodes this via ``(?!\d)``; this
    post-check is a defensive layer that also protects user-supplied
    regexes that lack the lookahead.
    """
    if not text:
        return False
    m = re.match(regex, text)
    if m is None:
        return False
    end = m.end()
    if end < len(text) and text[end].isdigit():
        return False
    return True


def extract_question_number(text: str) -> int | None:
    """Return the leading integer from ``text`` after stripping leading spaces.

    Used to pull the question number out of an OCR line. Per spec: trim
    leading whitespace, then take all leading digits. ``"  10. 阅读"`` → 10,
    ``"本题共10分"`` → None (digits don't start at position 0).
    """
    if not text:
        return None
    m = _LEADING_DIGITS.match(text)
    if not m:
        return None
    return int(m.group(1))


def _find_question_number_in_blocks(
    block_texts: list[str],
    regex: str = DEFAULT_QUESTION_REGEX,
) -> int | None:
    """Return the question number hidden anywhere in a box's OCR blocks.

    PaddleOCR may return body text ahead of the question-number line, so we
    can't just look at block_texts[0]. Prefer the first block that matches
    the question regex; fall back to the first block with leading digits
    so the function still returns a number for odd OCR layouts.
    """
    for text in block_texts:
        if is_question(text, regex):
            num = extract_question_number(text)
            if num is not None:
                return num
    for text in block_texts:
        num = extract_question_number(text)
        if num is not None:
            return num
    return None


def filter_questions_by_order(
    question_blocks: list["OcrBlock"],
    regex: str = DEFAULT_QUESTION_REGEX,
) -> list["OcrBlock"]:
    """Reclassify question boxes whose numbers break top-to-bottom order.

    Spec: after ``is_question`` is determined per box, walk the boxes in
    y1 order (top to bottom) and extract each one's leading question
    number. Boxes whose number is not strictly greater than every prior
    number are misreadings (e.g. OCR returned "0" for "10"), so the box
    gets ``is_question=False`` and is dropped from the result.

    Mutates the input ``OcrBlock`` objects' ``is_question`` flag; returns
    the surviving (still-question) blocks in their original list order.
    """
    if not question_blocks:
        return []

    sorted_blocks = sorted(
        question_blocks,
        key=lambda b: b.bbox_xyxy[1],
    )

    prev_max: int | None = None
    bad_ids: set[int] = set()
    for block in sorted_blocks:
        num = _find_question_number_in_blocks(block.block_texts, regex)
        if num is None:
            bad_ids.add(block.source_detection_id)
            continue
        if prev_max is not None and num <= prev_max:
            bad_ids.add(block.source_detection_id)
        else:
            prev_max = num

    for block in question_blocks:
        if block.source_detection_id in bad_ids:
            block.is_question = False

    return [b for b in question_blocks if b.is_question]
