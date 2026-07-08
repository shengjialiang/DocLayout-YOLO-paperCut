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


def test_validate_regex_rejects_matches_empty():
    with pytest.raises(ValueError):
        validate_regex(r".*")
