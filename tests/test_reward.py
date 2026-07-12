from training.reward import check_constraint, reward


def test_max_words():
    assert check_constraint({"type": "max_words", "n": 3}, "one two three")
    assert not check_constraint({"type": "max_words", "n": 3}, "one two three four")


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
