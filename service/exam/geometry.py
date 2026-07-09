"""Question box coordinate extension logic."""
from __future__ import annotations


def extend_questions(
    questions: list[dict],
    all_boxes: list[dict],
) -> list[dict]:
    """Extend each question's bottom to the next question's top.

    The LAST question's bottom = max(y2) of all plain-text-class boxes,
    per spec section 1.3 ("最后框 = 所有框中 y2 最大且类别为 plain text").

    If no plain-text box exists, the last question keeps its original bottom.

    Args:
        questions: Each dict has keys `bbox_xyxy: [x1, y1, x2, y2]`, `text`, `source_ids`.
        all_boxes: All original detections, each with `class_name` and `bbox_xyxy`.

    Returns:
        New list of questions with extended bbox_xyxy. Original `text` and `source_ids` preserved.
    """
    if not questions:
        return []

    sorted_qs = sorted(questions, key=lambda q: q["bbox_xyxy"][1])

    # Compute max y2 of plain text class for last question's bottom.
    plain_text_y2s = [
        b["bbox_xyxy"][3] for b in all_boxes
        if b.get("class_name") == "plain text"
    ]
    last_bottom = max(plain_text_y2s) if plain_text_y2s else None

    out = []
    for i, q in enumerate(sorted_qs):
        x1, y1, x2, y2 = q["bbox_xyxy"]
        new_box = dict(q)
        if i + 1 < len(sorted_qs):
            next_top = sorted_qs[i + 1]["bbox_xyxy"][1]
            new_box["bbox_xyxy"] = [x1, y1, x2, next_top]
        elif last_bottom is not None:
            new_box["bbox_xyxy"] = [x1, y1, x2, last_bottom]
        else:
            # Keep original
            new_box["bbox_xyxy"] = [x1, y1, x2, y2]
        out.append(new_box)
    return out


def align_question_right_edges(
    questions: list[dict],
    all_boxes: list[dict],
) -> list[dict]:
    """Unify every question's right edge to the rightmost plain-text x2.

    Per spec: after all question frames have been determined, every
    question's right edge (``x2``) is set to the largest ``x2`` among
    all plain-text-class annotation boxes. Catches cases where some
    question frames have a shorter right edge than the column they sit
    in (e.g. OCR-cropped plain-text boxes that don't reach the page
    margin), so question box outlines line up across the page.

    No-op when there are no plain-text-class boxes or no questions;
    in that case the input list is returned as-is.

    Args:
        questions: Each dict has keys ``bbox_xyxy: [x1, y1, x2, y2]``,
            ``text``, ``source_ids``, ``ocr_score``.
        all_boxes: All original detections, each with ``class_name`` and
            ``bbox_xyxy``.

    Returns:
        New list of dicts with each ``bbox_xyxy[2]`` updated to
        ``max(plain_text_x2)``. Original input dicts are not mutated;
        the returned list is the same object only when no alignment
        was applied.
    """
    if not questions:
        return []
    plain_x2s = [
        b["bbox_xyxy"][2] for b in all_boxes
        if b.get("class_name") == "plain text"
    ]
    if not plain_x2s:
        return questions
    max_x2 = max(plain_x2s)
    out = []
    for q in questions:
        x1, y1, _, y2 = q["bbox_xyxy"]
        new_box = dict(q)
        new_box["bbox_xyxy"] = [x1, y1, max_x2, y2]
        out.append(new_box)
    return out
