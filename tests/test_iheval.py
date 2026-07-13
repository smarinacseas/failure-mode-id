from training.iheval import score_conflict


class _Sampler:
    def __init__(self, fixed): self.fixed = fixed
    def generate(self, messages, max_tokens, temperature, reasoning): return self.fixed


def test_score_conflict_is_mean_pass_rate():
    samples = [{"messages": [], "constraints": [{"type": "max_words", "n": 2}]},
               {"messages": [], "constraints": [{"type": "keyword_include", "word": "ok"}]}]
    # response "ok" → 1 word (≤2 ✓) and contains "ok" (✓): both pass.
    assert score_conflict(samples, _Sampler("ok")) == 1.0
    # response "way too many words here" → fails max_words(2); passes keyword? no "ok".
    assert score_conflict(samples, _Sampler("way too many words here")) == 0.0
