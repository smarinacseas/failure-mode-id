"""T1.3 GRPO reward-adapter tests (written test-first — reward_adapter.py does
not exist yet, RED).

The adapter turns the T1.1 verifier reward `constraint_reward(response, specs)
-> float` into TRL's GRPO reward signature `(prompts, completions, **kwargs) ->
list[float]` (PREREG amendment 2026-07-16 (c)): it grades `extract_final(c)` for
each completion `c`, reads the per-prompt constraint list from a `specs` column
(carried as a JSON string alongside the prompt), and applies the length cap.
"""
import json

from reward_adapter import make_constraint_reward

# specs as the GRPO dataset carries them: one JSON string per prompt
S_CAT = json.dumps([{"type": "word_count", "op": "exact", "n": 3},
                    {"type": "keyword_include", "keywords": ["cat"]}])
S_DOG = json.dumps([{"type": "word_count", "op": "exact", "n": 3},
                    {"type": "keyword_include", "keywords": ["dog"]}])


def test_returns_one_float_per_completion():
    fn = make_constraint_reward()
    out = fn(prompts=["p", "p"], completions=["a cat sat", "a cat sat"],
             specs=[S_CAT, S_CAT])
    assert out == [1.0, 1.0]


def test_partial_and_full_credit_are_per_prompt():
    fn = make_constraint_reward()
    out = fn(prompts=["p", "p"], completions=["a cat sat", "a cat sat"],
             specs=[S_CAT, S_DOG])          # 2nd prompt wants 'dog', absent -> 0.5
    assert out == [1.0, 0.5]


def test_grades_only_the_post_final_answer():
    # scaffold before ===FINAL=== must not be graded — only the final answer is.
    fn = make_constraint_reward()
    completion = "INVENTORY: dog dog dog\n===FINAL===\na cat sat"
    assert fn(prompts=["p"], completions=[completion], specs=[S_CAT]) == [1.0]


def test_empty_completion_is_negative_malformed_penalty():
    fn = make_constraint_reward()
    assert fn(prompts=["p"], completions=[""], specs=[S_CAT])[0] < 0


def test_length_cap_docks_padding():
    fn = make_constraint_reward(max_chars=50)
    concise = fn(prompts=["p"], completions=["cat"],
                 specs=[json.dumps([{"type": "keyword_include", "keywords": ["cat"]}])])[0]
    verbose = fn(prompts=["p"], completions=["cat " + "x " * 200],
                 specs=[json.dumps([{"type": "keyword_include", "keywords": ["cat"]}])])[0]
    assert concise == 1.0
    assert verbose < concise


def test_specs_accepted_as_parsed_list_too():
    # robustness: a list[dict] specs column (not JSON-stringified) also works.
    fn = make_constraint_reward()
    specs = [[{"type": "keyword_include", "keywords": ["cat"]}]]
    assert fn(prompts=["p"], completions=["a cat"], specs=specs) == [1.0]


def test_ignores_trl_extra_kwargs():
    # TRL passes trainer_state / completion_ids / log_metric etc. — must not crash.
    fn = make_constraint_reward()
    out = fn(prompts=["p"], completions=["a cat sat"], specs=[S_CAT],
             completion_ids=[[1, 2, 3]], trainer_state=object(), log_metric=lambda *a: None)
    assert out == [1.0]


def test_conversational_completion_format():
    # with a conversational dataset TRL returns completions as message lists.
    fn = make_constraint_reward()
    comp = [{"role": "assistant", "content": "a cat sat"}]
    assert fn(prompts=["p"], completions=[comp], specs=[S_CAT]) == [1.0]
