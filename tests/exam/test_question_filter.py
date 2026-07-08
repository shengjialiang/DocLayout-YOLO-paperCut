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
    assert is_question("(2) 已知 x = 1", DEFAULT_QUESTION_REGEX)


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
    """OCR often confuses . with 。 — default regex must still match."""
    assert is_question("1。 题目", DEFAULT_QUESTION_REGEX)


def test_default_regex_matches_chinese_enumeration_comma():
    assert is_question("2、 题目", DEFAULT_QUESTION_REGEX)


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
