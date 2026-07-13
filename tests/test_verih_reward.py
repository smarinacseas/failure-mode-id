from pathlib import Path

from training.verih_reward import has_think_block, native_score, rl_reward, strip_think

# Real IF_FUNCTIONS_MAP entries (data/verih/RLVR/verl/utils/reward_score/if_functions.py):
#   verify_keywords(text, keyword_list) -> bool          (all keywords present, case-insensitive)
#   verify_letter_frequency(text, letter, N) -> bool      (raises ValueError if len(letter) != 1)
#   validate_json_format(text) -> bool                    (text is exactly one parseable JSON value)


def test_rl_reward_passes_with_think_block_and_satisfying_answer():
    gt = {"func_name": "verify_keywords", "keyword_list": ["alpha", "beta"]}
    text = "<think>reasoning here</think> This mentions alpha and beta."
    assert rl_reward(text, gt) == 1.0


def test_rl_reward_fails_without_think_block_even_if_answer_satisfies():
    gt = {"func_name": "verify_keywords", "keyword_list": ["alpha", "beta"]}
    text = "This mentions alpha and beta."
    assert rl_reward(text, gt) == 0.0


def test_rl_reward_fails_with_think_block_and_violating_answer():
    gt = {"func_name": "verify_keywords", "keyword_list": ["alpha", "beta"]}
    text = "<think>reasoning here</think> This only mentions alpha."
    assert rl_reward(text, gt) == 0.0


def test_native_score_returns_zero_on_unknown_func_name():
    gt = {"func_name": "not_a_real_verifier"}
    assert native_score("<think>t</think> anything", gt) == 0.0


def test_native_score_returns_zero_on_missing_func_name():
    assert native_score("<think>t</think> anything", {}) == 0.0


def test_native_score_returns_zero_on_exception_raising_input():
    # verify_letter_frequency raises ValueError when len(letter) != 1.
    gt = {"func_name": "verify_letter_frequency", "letter": "ab", "N": 2}
    assert native_score("<think>t</think> whatever text", gt) == 0.0


def test_strip_think_splits_on_last_occurrence():
    text = "<think>a</think> middle <think>b</think> final answer"
    assert strip_think(text) == "final answer"


def test_strip_think_passthrough_when_no_think_tag():
    assert strip_think("plain answer, no think tag") == "plain answer, no think tag"


def test_has_think_block_detects_dotall_span():
    assert has_think_block("<think>line one\nline two</think> answer")
    assert not has_think_block("no think tag here")


def test_none_valued_args_are_dropped_before_verifier_call():
    # Real gt rows carry every verifier's kwarg name, nulled out except the
    # ones relevant to func_name (see data/verih/RLVR .../train.json rows).
    # If native_score forwarded the None-valued keys as kwargs, verify_keywords
    # (which only accepts keyword_list) would raise TypeError -> swallowed ->
    # 0.0. Getting 1.0 back proves the None-valued keys were filtered out.
    gt = {"func_name": "verify_keywords", "keyword_list": ["hello"], "N": None, "word": None}
    text = "<think>t</think> hello world"
    assert native_score(text, gt) == 1.0


def test_native_score_never_raises_on_real_gt():
    # Real, on-disk VerIH train rows — no network. gt strings decode to the
    # ~25-verifier taxonomy; native_score must swallow every verifier's
    # exception without leaking it, per ih.py's own try/except wrapper.
    from training.data import load_verih

    train_path = Path(__file__).resolve().parents[1] / "data" / "verih" / "RLVR" / "dataset" / "verih" / "train.json"
    samples = load_verih(str(train_path), limit=500)
    gts = [s["constraints"][0] for s in samples if s["constraints"]][:200]
    assert len(gts) == 200

    for gt in gts:
        result = native_score("<think>t</think> test answer", gt)
        assert result in (0.0, 1.0)
