"""T1.2 contamination-screen tests (written before contamination.py). A prompt is
contaminated if it shares any 13-word n-gram with a reference set (T01 holdout,
CC-75, IFEval originals) — used both to keep train/holdout disjoint and to keep
training prompts out of the eval sets.
"""
from contamination import shares_ngram, word_ngrams


def _long(n):
    return " ".join(f"word{i}" for i in range(n))


def test_short_text_has_no_13grams():
    assert word_ngrams("only a handful of words here", n=13) == set()


def test_exactly_13_words_yields_one_ngram():
    assert len(word_ngrams(_long(13), n=13)) == 1


def test_identical_long_text_shares_an_ngram():
    ref = word_ngrams(_long(30))
    assert shares_ngram(_long(30), ref) is True


def test_distinct_text_shares_nothing():
    ref = word_ngrams(_long(30))
    assert shares_ngram("a totally unrelated and much shorter sentence", ref) is False


def test_ngrams_are_case_and_punctuation_insensitive():
    a = "The Quick Brown Fox Jumps Over The Lazy Dog And Then Runs Away Fast!"
    b = "the quick brown fox jumps over the lazy dog and then runs away fast"
    assert word_ngrams(a) & word_ngrams(b)


def test_partial_overlap_below_13_words_is_not_a_hit():
    # share only 12 consecutive words -> no 13-gram in common
    shared = " ".join(f"w{i}" for i in range(12))
    a = shared + " alpha beta gamma"
    b = shared + " delta epsilon zeta"
    assert shares_ngram(a, word_ngrams(b)) is False
