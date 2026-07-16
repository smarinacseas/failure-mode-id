"""T1.1 precision-pool verifier tests (CAUSE_B archetype: exact execution).
Written test-first — precision_pool.py does not exist yet (RED until implemented).
Positive / negative / near-miss (off-by-one, embedded-number) per checker.
"""
import precision_pool  # noqa: F401 — importing registers the precision verifiers
from base import check


def spec(t, **kw):
    return {"type": t, **kw}


# --- word_count --------------------------------------------------------------

def test_word_count_exact_passes():
    assert check("one two three four five", spec("word_count", op="exact", n=5)).passed is True


def test_word_count_exact_off_by_one_fails_and_reports_actual():
    r = check("one two three four", spec("word_count", op="exact", n=5))
    assert r.passed is False and "4" in r.detail


def test_word_count_min_passes():
    assert check("a b c d", spec("word_count", op="min", n=3)).passed is True


def test_word_count_max_over_fails():
    assert check("a b c d e", spec("word_count", op="max", n=4)).passed is False


def test_word_count_range_passes():
    assert check("a b c d", spec("word_count", op="range", min=3, max=6)).passed is True


# --- sentence_count (decimal-safe) -------------------------------------------

def test_sentence_count_three_terminators_passes():
    assert check("Foo. Bar! Baz?", spec("sentence_count", op="exact", n=3)).passed is True


def test_sentence_count_decimal_is_not_a_boundary():
    assert check("Pi is about 3.14 today.", spec("sentence_count", op="exact", n=1)).passed is True


def test_sentence_count_off_by_one_fails():
    assert check("One. Two.", spec("sentence_count", op="exact", n=3)).passed is False


# --- item_count --------------------------------------------------------------

def test_item_count_numbered_passes():
    assert check("1. alpha\n2. beta\n3. gamma", spec("item_count", op="exact", n=3)).passed is True


def test_item_count_bullets_passes():
    assert check("- x\n- y", spec("item_count", op="exact", n=2)).passed is True


def test_item_count_off_by_one_fails_and_reports_actual():
    r = check("- x\n- y\n- z", spec("item_count", op="exact", n=2))
    assert r.passed is False and "3" in r.detail


# --- arithmetic_result -------------------------------------------------------

def test_arithmetic_expected_present_passes():
    assert check("The total comes to 42 units.", spec("arithmetic_result", expected=42)).passed is True


def test_arithmetic_expression_is_evaluated():
    assert check("Answer: 14", spec("arithmetic_result", expression="3 * 4 + 2")).passed is True


def test_arithmetic_wrong_number_fails():
    assert check("The total is 41 units.", spec("arithmetic_result", expected=42)).passed is False


def test_arithmetic_embedded_in_larger_number_fails():
    # near-miss: "42" only appears inside "4200", never as the standalone value
    assert check("Revenue was 4200 dollars.", spec("arithmetic_result", expected=42)).passed is False


# --- ordering ----------------------------------------------------------------

def test_ordering_correct_order_passes():
    assert check("First the intro, then the body, finally the conclusion.",
                 spec("ordering", items=["intro", "body", "conclusion"])).passed is True


def test_ordering_wrong_order_fails():
    assert check("The conclusion came before the intro.",
                 spec("ordering", items=["intro", "conclusion"])).passed is False


def test_ordering_missing_item_fails():
    assert check("intro then body", spec("ordering", items=["intro", "body", "conclusion"])).passed is False


# --- exact_repetition --------------------------------------------------------

def test_exact_repetition_correct_count_passes():
    assert check("thank you very much, thank you again",
                 spec("exact_repetition", phrase="thank you", count=2)).passed is True


def test_exact_repetition_wrong_count_fails_and_reports_actual():
    r = check("thank you", spec("exact_repetition", phrase="thank you", count=2))
    assert r.passed is False and "1" in r.detail
