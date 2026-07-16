"""T1.1 GRPO reward wrapper tests (plan T1.1: fraction-of-constraints-passed,
dense partial credit, minus a malformed-output penalty; length cap so verbosity
can't be the win). Written test-first — reward.py does not exist yet (RED).
"""
from reward import constraint_reward


def test_all_constraints_pass_reward_is_one():
    specs = [{"type": "word_count", "op": "exact", "n": 3},
             {"type": "keyword_include", "keywords": ["cat"]}]
    assert constraint_reward("a cat sat", specs) == 1.0


def test_partial_pass_is_dense_fraction():
    specs = [{"type": "word_count", "op": "exact", "n": 3},
             {"type": "keyword_include", "keywords": ["dog"]}]
    assert constraint_reward("a cat sat", specs) == 0.5


def test_no_constraints_pass_is_zero():
    specs = [{"type": "keyword_include", "keywords": ["dog"]},
             {"type": "keyword_include", "keywords": ["fish"]}]
    assert constraint_reward("a cat sat", specs) == 0.0


def test_empty_response_is_negative_malformed_penalty():
    # empty must score below any real attempt so GRPO never prefers collapsing
    assert constraint_reward("", [{"type": "word_count", "op": "exact", "n": 3}]) < 0


def test_whitespace_only_is_malformed():
    assert constraint_reward("   \n  ", [{"type": "word_count", "op": "min", "n": 1}]) < 0


def test_overlength_is_penalized_vs_concise():
    specs = [{"type": "keyword_include", "keywords": ["cat"]}]
    concise = constraint_reward("cat", specs, max_chars=100)
    verbose = constraint_reward("cat " + "x " * 200, specs, max_chars=100)
    assert concise == 1.0
    assert verbose < concise            # padding to hit constraints must not win


def test_empty_spec_list_returns_zero_not_crash():
    assert constraint_reward("anything", []) == 0.0
