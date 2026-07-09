"""Question box coordinate extension tests."""
from service.exam.geometry import align_question_right_edges, extend_questions


def box(x1, y1, x2, y2, cls="plain text"):
    return {"class_name": cls, "bbox_xyxy": [x1, y1, x2, y2]}


def question(text, x1, y1, x2, y2, source_ids=(0,)):
    return {"text": text, "bbox_xyxy": [x1, y1, x2, y2], "source_ids": list(source_ids)}


def test_middle_question_bottom_equals_next_question_top():
    """Middle questions extend their bottom to the next question's top."""
    all_boxes = [
        box(0, 0, 100, 50, cls="figure"),       # 0: top
        box(0, 100, 100, 200),                  # 1: question 1 (plain text)
        box(0, 250, 100, 320),                  # 2: question 2
        box(0, 400, 100, 500),                  # 3: question 3
        box(0, 600, 100, 800),                  # 4: tail plain text (max-y2)
    ]
    qs = [
        question("1.", 0, 100, 100, 180, source_ids=[1]),
        question("2.", 0, 250, 100, 290, source_ids=[2]),
        question("3.", 0, 400, 100, 480, source_ids=[3]),
    ]
    out = extend_questions(qs, all_boxes)
    # Q1 bottom = Q2 top = 250
    assert out[0]["bbox_xyxy"] == [0, 100, 100, 250]
    # Q2 bottom = Q3 top = 400
    assert out[1]["bbox_xyxy"] == [0, 250, 100, 400]


def test_last_question_bottom_uses_max_y2_plain_text():
    """Last question's bottom equals max(y2) among plain-text boxes."""
    all_boxes = [
        box(0, 100, 100, 200),                  # question 1
        box(0, 250, 100, 320),                  # question 2
        box(0, 400, 100, 500),                  # question 3
        box(0, 600, 100, 800),                  # tail plain text (max y2)
    ]
    qs = [
        question("1.", 0, 100, 100, 180, source_ids=[0]),
        question("2.", 0, 250, 100, 290, source_ids=[1]),
        question("3.", 0, 400, 100, 480, source_ids=[2]),
    ]
    out = extend_questions(qs, all_boxes)
    # Q3 bottom = max(y2) plain text = 800
    assert out[2]["bbox_xyxy"] == [0, 400, 100, 800]


def test_last_question_uses_max_y2_plain_text_not_other_classes():
    all_boxes = [
        box(0, 100, 100, 200),                                  # q1
        box(0, 250, 100, 320),                                  # q2
        box(0, 400, 100, 500),                                  # q3
        box(0, 600, 100, 1500, cls="figure"),                   # 极高 figure (max y2)
        box(0, 550, 100, 700),                                  # plain text (NOT max y2)
    ]
    qs = [
        question("1.", 0, 100, 100, 180, source_ids=[1]),
        question("2.", 0, 250, 100, 290, source_ids=[2]),
        question("3.", 0, 400, 100, 480, source_ids=[3]),
    ]
    out = extend_questions(qs, all_boxes)
    # Last question bottom should be max y2 plain text = 700, not 1500
    assert out[2]["bbox_xyxy"] == [0, 400, 100, 700]


def test_single_question_uses_max_y2_plain_text():
    all_boxes = [
        box(0, 100, 100, 200),
        box(0, 500, 100, 900, cls="plain text"),
    ]
    qs = [question("1.", 0, 100, 100, 180, source_ids=[0])]
    out = extend_questions(qs, all_boxes)
    assert out[0]["bbox_xyxy"] == [0, 100, 100, 900]


def test_zero_questions_returns_empty():
    assert extend_questions([], []) == []


def test_input_order_unordered_questions_sorted_by_y1():
    all_boxes = [
        box(0, 100, 100, 200),
        box(0, 250, 100, 320),
    ]
    qs = [
        question("2.", 0, 250, 100, 290, source_ids=[1]),
        question("1.", 0, 100, 100, 180, source_ids=[0]),
    ]
    out = extend_questions(qs, all_boxes)
    # After sort by y1: Q1 (y=100), Q2 (y=250). Q1 bottom = 250.
    assert out[0]["bbox_xyxy"][1] == 100
    assert out[0]["bbox_xyxy"][3] == 250
    assert out[1]["bbox_xyxy"][1] == 250
    # last question has no "next", uses max plain text y2 = 320
    assert out[1]["bbox_xyxy"][3] == 320


def test_no_plain_text_returns_questions_unchanged_for_last():
    """When no plain text exists beyond last question, keep its original bottom."""
    all_boxes = [
        box(0, 100, 100, 200, cls="figure"),
        box(0, 250, 100, 320, cls="table"),
    ]
    qs = [question("1.", 0, 100, 100, 180, source_ids=[0])]
    out = extend_questions(qs, all_boxes)
    # No plain text at all; last question's bottom = original 180
    assert out[0]["bbox_xyxy"] == [0, 100, 100, 180]


# --- align_question_right_edges ---


def test_align_empty_questions_returns_empty():
    """No questions → no-op, return empty list."""
    out = align_question_right_edges([], [box(0, 0, 100, 50)])
    assert out == []


def test_align_no_plain_text_returns_questions_unchanged():
    """If no plain-text boxes exist, the input list is returned unchanged."""
    all_boxes = [
        box(0, 100, 100, 200, cls="figure"),
        box(0, 250, 100, 320, cls="table"),
    ]
    qs = [question("1.", 0, 100, 80, 180, source_ids=[0])]
    out = align_question_right_edges(qs, all_boxes)
    assert out is qs  # same list returned unchanged
    assert out[0]["bbox_xyxy"] == [0, 100, 80, 180]


def test_align_basic_sets_all_question_x2_to_max_plain_text():
    """All questions' x2 are unified to max(x2) of plain-text boxes."""
    all_boxes = [
        box(10, 100, 350, 200),  # q1 (x2 = 350)
        box(10, 250, 350, 320),  # q2
        box(10, 400, 350, 500),  # q3
        box(10, 600, 800, 800),  # tail plain text with widest x2
    ]
    qs = [
        question("1.", 10, 100, 350, 180, source_ids=[0]),
        question("2.", 10, 250, 350, 290, source_ids=[1]),
        question("3.", 10, 400, 350, 480, source_ids=[2]),
    ]
    out = align_question_right_edges(qs, all_boxes)
    assert all(q["bbox_xyxy"][2] == 800 for q in out)


def test_align_ignores_non_plain_text_classes():
    """figure / table x2 values must NOT influence the max."""
    all_boxes = [
        box(10, 100, 350, 200),                  # plain text q1
        box(10, 250, 350, 320),                  # plain text q2
        box(10, 400, 350, 500),                  # plain text q3
        box(10, 0, 9999, 50, cls="figure"),      # figure with huge x2 — ignored
        box(10, 600, 800, 800),                  # plain text (max plain x2 = 800)
    ]
    qs = [
        question("1.", 10, 100, 350, 180, source_ids=[0]),
        question("2.", 10, 250, 350, 290, source_ids=[1]),
        question("3.", 10, 400, 350, 480, source_ids=[2]),
    ]
    out = align_question_right_edges(qs, all_boxes)
    assert all(q["bbox_xyxy"][2] == 800 for q in out)
    assert out[0]["bbox_xyxy"][2] == 800


def test_align_uses_max_across_multiple_plain_text_boxes():
    """The maximum x2 across ALL plain-text boxes is used, not the last."""
    all_boxes = [
        box(10, 100, 400, 200),  # x2 = 400
        box(10, 250, 350, 320),  # x2 = 350
        box(10, 400, 800, 500),  # x2 = 800 (max)
        box(10, 600, 500, 800),  # x2 = 500
    ]
    qs = [
        question("1.", 10, 100, 350, 180, source_ids=[0]),
        question("2.", 10, 250, 400, 290, source_ids=[1]),
    ]
    out = align_question_right_edges(qs, all_boxes)
    assert out[0]["bbox_xyxy"] == [10, 100, 800, 180]
    assert out[1]["bbox_xyxy"] == [10, 250, 800, 290]


def test_align_preserves_y1_and_y2():
    """Vertical coordinates (y1, y2) and x1 must be preserved."""
    all_boxes = [box(10, 100, 800, 200)]
    qs = [question("1.", 50, 120, 200, 180, source_ids=[0])]
    out = align_question_right_edges(qs, all_boxes)
    assert out[0]["bbox_xyxy"] == [50, 120, 800, 180]


def test_align_does_not_mutate_input_question_dict():
    """The function must return NEW dicts, never mutate input."""
    all_boxes = [box(10, 100, 800, 200)]
    q_in = question("1.", 10, 100, 350, 180, source_ids=[0])
    original_bbox = list(q_in["bbox_xyxy"])
    out = align_question_right_edges([q_in], all_boxes)
    assert q_in["bbox_xyxy"] == original_bbox
    assert out[0] is not q_in
    assert out[0]["bbox_xyxy"] != q_in["bbox_xyxy"]
    assert out[0]["bbox_xyxy"][2] == 800


def test_align_preserves_other_question_fields():
    """Other fields (text, source_ids, ocr_score) must survive the transform."""
    all_boxes = [box(10, 100, 800, 200)]
    q_in = {
        "text": "1. (计算题)", "bbox_xyxy": [10, 100, 350, 180],
        "source_ids": [42], "ocr_score": 0.93,
    }
    out = align_question_right_edges([q_in], all_boxes)
    assert out[0]["text"] == "1. (计算题)"
    assert out[0]["source_ids"] == [42]
    assert out[0]["ocr_score"] == 0.93


def test_align_single_question_single_plain_text_box():
    all_boxes = [box(0, 100, 600, 200)]
    qs = [question("1.", 0, 100, 200, 180, source_ids=[0])]
    out = align_question_right_edges(qs, all_boxes)
    assert out[0]["bbox_xyxy"] == [0, 100, 600, 180]


def test_align_empty_all_boxes_returns_questions_unchanged():
    """Empty detection list → no plain text → no-op."""
    qs = [question("1.", 0, 100, 200, 180, source_ids=[0])]
    out = align_question_right_edges(qs, [])
    assert out is qs
    assert out[0]["bbox_xyxy"] == [0, 100, 200, 180]
