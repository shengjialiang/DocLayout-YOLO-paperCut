"""Question number regex matching tests."""
import pytest

from service.exam.question_filter import (
    DEFAULT_QUESTION_REGEX,
    is_question,
    validate_regex,
)


def test_default_regex_matches_dotted():
    assert is_question("1. (计算题) 1+1=?", DEFAULT_QUESTION_REGEX)


def test_default_regex_matches_paren():
    # Default regex now only accepts English-period form. The paren form
    # ``"(2) ..."`` is exercised through an explicit regex here so users
    # who want it back can opt in.
    assert not is_question("(2) 已知 x = 1", DEFAULT_QUESTION_REGEX)
    assert is_question("(2) 已知 x = 1", r"^\s*\(?\d+[\.\)]")


def test_default_regex_matches_two_digit():
    assert is_question("10. 阅读理解", DEFAULT_QUESTION_REGEX)


def test_default_regex_matches_with_leading_whitespace():
    assert is_question("  3. 选择题", DEFAULT_QUESTION_REGEX)


def test_default_regex_rejects_subnumber():
    assert not is_question("1.1 子项", DEFAULT_QUESTION_REGEX)


def test_default_regex_rejects_plain_text():
    assert not is_question("本题共 10 分", DEFAULT_QUESTION_REGEX)


def test_default_regex_rejects_empty():
    assert not is_question("", DEFAULT_QUESTION_REGEX)


def test_custom_regex():
    assert is_question("Question 3: ...", r"^Question\s+\d+")


def test_validate_regex_accepts_valid():
    assert validate_regex(r"^\d+\.") == r"^\d+\."


def test_validate_regex_rejects_invalid():
    with pytest.raises(ValueError):
        validate_regex(r"[")


def test_default_regex_matches_chinese_period():
    """OCR often confuses . with 。 — but the default regex now expects only
    ``.`` (Chinese punctuation is normalized to ``.`` upstream by
    ``normalize_question_number``). Verify the normalized form matches
    the default, and the raw Chinese period only matches an explicit
    regex that opts in to those characters."""
    assert is_question("1. 题目", DEFAULT_QUESTION_REGEX)
    assert not is_question("1。 题目", DEFAULT_QUESTION_REGEX)
    assert is_question("1。 题目", r"^\s*\(?\d+[\.。．、\)]")


def test_default_regex_matches_chinese_enumeration_comma():
    assert not is_question("2、 题目", DEFAULT_QUESTION_REGEX)
    assert is_question("2、 题目", r"^\s*\(?\d+[\.。．、\)]")


def test_default_regex_matches_fullwidth_period():
    """OCR sometimes produces 4． (U+FF0E FULLWIDTH FULL STOP). The default
    regex no longer covers it; users wanting fullwidth handling must opt
    in with an explicit regex."""
    assert not is_question("4． 题目", DEFAULT_QUESTION_REGEX)
    assert is_question("4． 题目", r"^\s*\(?\d+[\.。．、\)]")


# --- is_question trailing-digit post-check ---

def test_is_question_post_check_rejects_subnumber_with_default_regex():
    """Default regex + post-check still rejects '1.1 子项'.

    The default regex already encodes ``(?!\d)`` so this is the same
    rejection exercised through a different code path; covered here
    so any future change to the default regex is caught.
    """
    assert not is_question("1.1 子项", DEFAULT_QUESTION_REGEX)


def test_is_question_post_check_rejects_digit_after_custom_regex():
    """A custom regex without ``(?!\d)`` must still reject sub-numbered
    text via the post-match digit check, e.g. ``"1.5 阅读"`` with
    ``r"^\d+\\."`` matches ``"1."`` but the next char is ``"5"``."""
    assert not is_question("1.5 阅读", r"^\d+\.")


def test_is_question_post_check_rejects_two_digit_decimal():
    """``12.5 阅读理解`` with regex ``r"^\d+\\."``: match ends at ``12.``,
    next char is ``"5"`` → reject."""
    assert not is_question("12.5 阅读理解", r"^\d+\.")


def test_is_question_post_check_accepts_match_at_end_of_text():
    """No character after the match → not digit-followed → still a question."""
    assert is_question("1.", DEFAULT_QUESTION_REGEX)


def test_is_question_post_check_accepts_space_after_match():
    """Space after the match is not a digit → still a question."""
    assert is_question("1. (计算题)", DEFAULT_QUESTION_REGEX)


def test_is_question_post_check_accepts_chinese_period_at_end():
    """The new default no longer accepts the Chinese-period form. Use an
    explicit regex (with the trailing ``(?!\d)`` check still needed if
    the user wants post-check protection) to verify that form keeps
    working when opted in."""
    assert not is_question("1。", DEFAULT_QUESTION_REGEX)
    assert not is_question("1。 题目", DEFAULT_QUESTION_REGEX)
    assert is_question("1。 题目", r"^\s*\(?\d+[\.。．、\)](?!\d)")


def test_is_question_post_check_does_not_strip_legitimate_two_digit():
    """A two-digit question number followed by a space (not a digit) must
    still be accepted. The post-check only rejects when the character
    immediately after the regex match is itself a digit."""
    assert is_question("12. 阅读", DEFAULT_QUESTION_REGEX)


def test_is_question_post_check_rejects_with_paren_form():
    """``(2)3 子项`` — the regex matches ``"(2)"`` and ``"3"`` follows."""
    assert not is_question("(2)3 子项", DEFAULT_QUESTION_REGEX)


def test_normalize_fullwidth_period_to_english_period():
    from service.exam.question_filter import normalize_question_number
    assert normalize_question_number("4． 题目") == "4. 题目"


def test_normalize_ocr_dot_to_english_period():
    from service.exam.question_filter import normalize_question_number
    assert normalize_question_number("1。 题目") == "1. 题目"


def test_normalize_ocr_comma_to_english_period():
    from service.exam.question_filter import normalize_question_number
    assert normalize_question_number("2， 题目") == "2. 题目"


def test_normalize_ocr_enumeration_comma():
    from service.exam.question_filter import normalize_question_number
    assert normalize_question_number("3、 题目") == "3. 题目"


def test_normalize_ocr_chinese_colon():
    from service.exam.question_filter import normalize_question_number
    assert normalize_question_number("4： 题目") == "4. 题目"


def test_normalize_idempotent_on_correct_format():
    from service.exam.question_filter import normalize_question_number
    assert normalize_question_number("1. 题目") == "1. 题目"


def test_normalize_leaves_paren_pattern_alone():
    from service.exam.question_filter import normalize_question_number
    assert normalize_question_number("(2) 题目") == "(2) 题目"


def test_normalize_does_not_touch_body_text():
    from service.exam.question_filter import normalize_question_number
    assert normalize_question_number("解答：因为A。") == "解答：因为A。"


def test_validate_regex_rejects_matches_empty():
    with pytest.raises(ValueError):
        validate_regex(r".*")


# --- extract_question_number ---

def test_extract_question_number_simple():
    from service.exam.question_filter import extract_question_number
    assert extract_question_number("1. (计算题) 1+1=?") == 1


def test_extract_question_number_two_digit():
    from service.exam.question_filter import extract_question_number
    assert extract_question_number("10. 阅读理解") == 10


def test_extract_question_number_strips_leading_spaces():
    from service.exam.question_filter import extract_question_number
    assert extract_question_number("   7. 题目") == 7


def test_extract_question_number_strips_leading_tabs_and_newlines():
    from service.exam.question_filter import extract_question_number
    assert extract_question_number("\t\n  3. 选择题") == 3


def test_extract_question_number_paren_pattern():
    from service.exam.question_filter import extract_question_number
    # "(2) 已知 x = 1" → leading digits = 2
    assert extract_question_number("(2) 已知 x = 1") == 2


def test_extract_question_number_fullwidth_paren():
    from service.exam.question_filter import extract_question_number
    assert extract_question_number("（3） 题目") == 3


def test_extract_question_number_chinese_period():
    from service.exam.question_filter import extract_question_number
    # Normalized text uses English '.', so digits are 4
    assert extract_question_number("4. 题目") == 4


def test_extract_question_number_no_digits_returns_none():
    from service.exam.question_filter import extract_question_number
    assert extract_question_number("本题共10分") is None


def test_extract_question_number_letters_only_returns_none():
    from service.exam.question_filter import extract_question_number
    assert extract_question_number("abc") is None


def test_extract_question_number_empty_returns_none():
    from service.exam.question_filter import extract_question_number
    assert extract_question_number("") is None


def test_extract_question_number_only_whitespace_returns_none():
    from service.exam.question_filter import extract_question_number
    assert extract_question_number("   \t  ") is None


def test_extract_question_number_very_large():
    from service.exam.question_filter import extract_question_number
    assert extract_question_number("1234. 题目") == 1234


# --- filter_questions_by_order ---

def _make_ocr_block(text: str, y1: float, source_id: int):
    """Build a minimal OcrBlock-like object for filter tests.

    Only the attributes the filter actually touches are populated; the rest
    get sensible defaults so the helper is reusable across tests.
    """
    from service.exam.schemas import OcrBlock
    # block_texts must contain at least one entry that matches the regex
    # so extract_question_number_from_box can find it.
    return OcrBlock(
        source_detection_id=source_id,
        bbox_xyxy=[0, y1, 100, y1 + 50],
        text=text,
        score=0.9,
        is_question=True,
        block_texts=[text],
    )


def test_filter_by_order_empty_list():
    from service.exam.question_filter import filter_questions_by_order
    assert filter_questions_by_order([]) == []


def test_filter_by_order_single_box_kept():
    from service.exam.question_filter import filter_questions_by_order
    blocks = [_make_ocr_block("5. 题目", 100, 0)]
    result = filter_questions_by_order(blocks)
    assert len(result) == 1
    assert result[0].source_detection_id == 0


def test_filter_by_order_monotonic_increasing_all_kept():
    from service.exam.question_filter import filter_questions_by_order
    blocks = [
        _make_ocr_block("1. 题目", 100, 0),
        _make_ocr_block("2. 题目", 200, 1),
        _make_ocr_block("3. 题目", 300, 2),
    ]
    result = filter_questions_by_order(blocks)
    assert [b.source_detection_id for b in result] == [0, 1, 2]
    assert all(b.is_question for b in blocks)


def test_filter_by_order_example_marks_001_as_out_of_order():
    """Spec example: 7/8/9/0/0/1/10 → the 0/0/1 boxes must be reclassified
    as not question boxes; 7/8/9/10 stay."""
    from service.exam.question_filter import filter_questions_by_order
    blocks = [
        _make_ocr_block("7. 题目", 100, 0),
        _make_ocr_block("8. 题目", 200, 1),
        _make_ocr_block("9. 题目", 300, 2),
        _make_ocr_block("0. 题目", 400, 3),
        _make_ocr_block("0. 题目", 500, 4),
        _make_ocr_block("1. 题目", 600, 5),
        _make_ocr_block("10. 题目", 700, 6),
    ]
    result = filter_questions_by_order(blocks)
    kept_ids = [b.source_detection_id for b in result]
    assert kept_ids == [0, 1, 2, 6], kept_ids
    # Reclassified blocks have is_question=False on the original list
    assert blocks[0].is_question is True
    assert blocks[1].is_question is True
    assert blocks[2].is_question is True
    assert blocks[3].is_question is False  # 0 after 9
    assert blocks[4].is_question is False  # 0 after 0
    assert blocks[5].is_question is False  # 1 after 0
    assert blocks[6].is_question is True   # 10 after 9


def test_filter_by_order_sorts_by_y1_not_input_order():
    """Order check uses bbox y1 (top→bottom), not list order.

    The numbers 7/8/9 are in increasing y1 order, so all boxes should be
    kept — the function must walk them in y1 order internally even when
    the input list is shuffled. (Return order is the input list order;
    we only verify is_question flags here.)"""
    from service.exam.question_filter import filter_questions_by_order
    blocks = [
        _make_ocr_block("8. 题目", 200, 1),  # y1=200
        _make_ocr_block("7. 题目", 100, 0),  # y1=100
        _make_ocr_block("9. 题目", 300, 2),  # y1=300
    ]
    result = filter_questions_by_order(blocks)
    assert [b.source_detection_id for b in result] == [1, 0, 2]  # input order
    assert all(b.is_question for b in blocks)


def test_filter_by_order_box_without_number_reclassified():
    """If we can't extract a question number at all, the box is reclassified."""
    from service.exam.question_filter import filter_questions_by_order
    blocks = [
        _make_ocr_block("1. 题目", 100, 0),
        _make_ocr_block("本题共10分", 200, 1),
        _make_ocr_block("2. 题目", 300, 2),
    ]
    result = filter_questions_by_order(blocks)
    kept_ids = [b.source_detection_id for b in result]
    assert kept_ids == [0, 2]
    assert blocks[1].is_question is False


def test_filter_by_order_uses_block_texts_not_joined_text():
    """The number can live in any block_text (e.g. when OCR returns body
    text first). filter must scan block_texts to find the question number,
    not just use the joined ``text`` field."""
    from service.exam.schemas import OcrBlock
    from service.exam.question_filter import filter_questions_by_order
    # "5." lives in the 2nd block_text, body text in the 1st.
    b0 = OcrBlock(
        source_detection_id=0, bbox_xyxy=[0, 100, 100, 150],
        text="已知x 5. 已知x=1", score=0.9, is_question=True,
        block_texts=["已知x", "5. 已知x=1"],
    )
    b1 = OcrBlock(
        source_detection_id=1, bbox_xyxy=[0, 200, 100, 250],
        text="6. 题目", score=0.9, is_question=True,
        block_texts=["6. 题目"],
    )
    result = filter_questions_by_order([b0, b1])
    assert [b.source_detection_id for b in result] == [0, 1]


def test_filter_by_order_all_invalid_reclassifies_all():
    from service.exam.question_filter import filter_questions_by_order
    blocks = [
        _make_ocr_block("本题共10分", 100, 0),
        _make_ocr_block("解答如下", 200, 1),
    ]
    result = filter_questions_by_order(blocks)
    assert result == []
    assert all(not b.is_question for b in blocks)


def test_filter_by_order_strictly_increasing_after_gap():
    """If OCR skips a question (e.g. 5, 6, 8), the skipped number is OK —
    the rule is strictly increasing, not contiguous."""
    from service.exam.question_filter import filter_questions_by_order
    blocks = [
        _make_ocr_block("5. 题目", 100, 0),
        _make_ocr_block("6. 题目", 200, 1),
        _make_ocr_block("8. 题目", 300, 2),
    ]
    result = filter_questions_by_order(blocks)
    assert [b.source_detection_id for b in result] == [0, 1, 2]


# --- new default-reg behavior docs ---


def test_default_regex_rejects_paren_form():
    """The default regex now expects digits-then-'.', no optional paren."""
    assert not is_question("(2) 已知 x = 1", DEFAULT_QUESTION_REGEX)
    assert not is_question("(10) 题目", DEFAULT_QUESTION_REGEX)


def test_default_regex_rejects_chinese_periods():
    """Chinese-period OCR output is NOT handled by the regex itself; callers
    must run ``normalize_question_number`` upstream to map 。/．/、 → '.'."""
    for text in ("1。 题目", "4． 题目", "2、 题目", "1。", "4．"):
        assert not is_question(text, DEFAULT_QUESTION_REGEX), text


def test_default_regex_accepts_only_english_period_form():
    """The default regex must still accept the canonical ``N.`` form
    (with optional leading whitespace and multi-digit numbers)."""
    for text in (
        "1.", "1. 题目",
        "  3. 选择题",
        "10. 阅读理解",
        "12. 阅读",
        "  3.5 子项",  # rejected by post-check, see above
    ):
        if text.startswith("  3.5"):
            assert not is_question(text, DEFAULT_QUESTION_REGEX), text
        else:
            assert is_question(text, DEFAULT_QUESTION_REGEX), text


def test_extract_question_number_still_handles_paren_and_chinese_period():
    """Changing DEFAULT_QUESTION_REGEX must NOT affect ``extract_question_number``
    — that function uses its own internal pattern that still accepts the
    paren form. (We verify it directly here as a regression guard.)"""
    from service.exam.question_filter import extract_question_number
    assert extract_question_number("(2) 已知 x = 1") == 2
    assert extract_question_number("（3） 题目") == 3
