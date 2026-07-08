"""Question box coordinate extension tests."""
from service.exam.geometry import extend_questions


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
