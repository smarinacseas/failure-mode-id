from training.iheval import score_conflict


class _Sampler:
    def __init__(self, fixed): self.fixed = fixed
    def generate(self, messages, max_tokens, temperature, reasoning): return self.fixed


def test_score_conflict_is_mean_native_verifier_satisfaction():
    # Real func_name from if_functions (verify_keywords: all keyword_list
    # entries must appear in the answer, case-insensitive) pins
    # score_conflict's mean logic against training/verih_reward.py's
    # native_score, not the abandoned type-keyed training/reward.py path
    # (which raises KeyError on func_name-keyed gt dicts).
    samples = [
        {"messages": [], "constraints": [{"func_name": "verify_keywords", "keyword_list": ["cat"]}]},
        {"messages": [], "constraints": [{"func_name": "verify_keywords", "keyword_list": ["dog"]}]},
    ]
    # Sampler always answers "the cat sat" — satisfies keyword_list=["cat"]
    # (1.0), violates keyword_list=["dog"] (0.0) — mean 0.5.
    assert score_conflict(samples, _Sampler("the cat sat")) == 0.5


def test_score_conflict_empty_samples_is_zero():
    assert score_conflict([], _Sampler("anything")) == 0.0


def test_score_conflict_empty_constraints_counts_as_zero():
    samples = [{"messages": [], "constraints": []}]
    assert score_conflict(samples, _Sampler("anything")) == 0.0
