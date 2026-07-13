import pytest

from training.reward import check_constraint, reward


def test_max_words():
    assert check_constraint({"type": "max_words", "n": 3}, "one two three")
    assert not check_constraint({"type": "max_words", "n": 3}, "one two three four")


def test_min_words():
    assert check_constraint({"type": "min_words", "n": 3}, "one two three")
    assert not check_constraint({"type": "min_words", "n": 4}, "one two three")


def test_keyword_word_boundary_not_substring():
    # "cat" inside "concatenate" must not count — reward-hacking guard.
    assert not check_constraint({"type": "keyword_include", "word": "cat"}, "we concatenate strings")
    assert check_constraint({"type": "keyword_forbid", "word": "cat"}, "we concatenate strings")
    assert check_constraint({"type": "keyword_include", "word": "cat"}, "the cat sat")


def test_sentence_count():
    assert check_constraint({"type": "sentence_count", "n": 2}, "One here. Two here.")
    assert check_constraint({"type": "sentence_count", "n": 1}, "The value is 3.14.")
    assert check_constraint({"type": "sentence_count", "n": 2}, "The value is 3.14. That is pi.")
    assert check_constraint({"type": "sentence_count", "n": 2}, "Is it 42? Yes.")


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown constraint type"):
        check_constraint({"type": "nope"}, "text")


def test_keyword_include_and_forbid():
    assert check_constraint({"type": "keyword_include", "word": "banana"}, "I like banana bread")
    assert not check_constraint({"type": "keyword_include", "word": "banana"}, "I like bread")
    assert check_constraint({"type": "keyword_forbid", "word": "banana"}, "I like bread")


def test_json_parses():
    assert check_constraint({"type": "json_parses"}, '{"a": 1}')
    assert not check_constraint({"type": "json_parses"}, "not json")


def test_reward_is_fraction_satisfied():
    cs = [{"type": "max_words", "n": 5}, {"type": "keyword_include", "word": "hi"}]
    assert reward(cs, "hi there") == 1.0
    assert reward(cs, "nope") == 0.5          # word ok, keyword missing
    assert reward([], "anything") == 1.0      # vacuously satisfied
