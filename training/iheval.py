"""IHEval conflict/aligned gate: native VerIH verifier satisfaction per
checkpoint, on the held-out VerIH test split.

`score_conflict` scores each sample's ANSWER against its own gt verifier
spec via `training/verih_reward.py::native_score` — the same scorer VerIH's
own eval protocol uses (check_answer's aligned/conflict branch), not the
binary think-format-gated `rl_reward` used during training. This is a
deliberate choice, not an oversight: `TinkerSampler.generate`
(training/proxy.py) returns `renderers.get_text_content(message)`, which
strips `<think>...</think>` blocks before the text ever reaches this module.
Scoring with `rl_reward` here would mean `has_think_block` sees post-strip
text and always returns False, collapsing every score to 0.0 regardless of
answer quality. `native_score` needs no think block — it strips through the
last `</think>` itself (a no-op on already-stripped text) and checks only
constraint satisfaction, so it is well-defined on exactly the text this
sampler produces.

Each sample's `constraints` list holds a single VerIH `gt` dict (see
training/data.py's `to_sample`); an empty list (gt missing/blank in the
source row) counts as 0.0 rather than being skipped, so malformed rows
still show up in the pass rate instead of silently inflating it.
"""

from __future__ import annotations

import time

from training.verih_reward import native_score


_DEFAULT_MAX_TOKENS = 2048


def score_conflict(samples: list[dict], sampler) -> float:
    """Mean native-verifier satisfaction of the sampler's answers over
    `samples` (0.0 on an empty sample list). Each sample's constraints[0] is
    the VerIH gt spec scored against the sampler's response text via
    native_score (answer-satisfaction only — think-block format is a
    training-time reward concern, not an eval-gate one; see module
    docstring)."""
    if not samples:
        return 0.0
    total = 0.0
    for s in samples:
        constraints = s["constraints"]
        if not constraints:
            continue  # 0.0 contribution — counted via len(samples) below
        text = sampler.generate(messages=s["messages"], max_tokens=_DEFAULT_MAX_TOKENS,
                                temperature=0.6, reasoning=True)  # 0.6: reasoning-on, no greedy loops
        total += native_score(text, constraints[0])
    return total / len(samples)


def main() -> None:
    import json
    import os

    import tinker
    from tinker_cookbook import renderers

    from training.data import load_verih
    from training.proxy import TinkerSampler

    limit_env = os.environ.get("IHEVAL_LIMIT")
    limit = int(limit_env) if limit_env else None
    max_tokens = int(os.environ.get("IHEVAL_MAX_TOKENS", "2048"))
    temperature = 0.6

    ckpts = json.loads(open("training/checkpoints.json").read())
    all_samples = load_verih("data/verih/RLVR/dataset/verih/test.json")

    conflict = [s for s in all_samples if "conflict" in (s["type"] or "")]
    aligned = [s for s in all_samples if "aligned" in (s["type"] or "")]
    if limit:
        conflict = conflict[:limit]
        aligned = aligned[:limit]
    print(f"test split: {len(conflict)} conflict, {len(aligned)} aligned "
          f"(of {len(all_samples)} total)")

    service = tinker.ServiceClient()
    base_client = service.create_sampling_client(model_path=ckpts["base"])
    ft_client = service.create_sampling_client(model_path=ckpts["ft"])

    # LIVE-VERIFY (gated phase): renderer/tokenizer construction mirrors
    # training/proxy.py's main() exactly — tokenizer comes from the base
    # checkpoint's sampling client and base/ft share the SAME renderer pair
    # (get_renderer("qwen3", tok) / get_renderer("qwen3_disable_thinking",
    # tok)), so only weights differ between the two samplers (fair
    # baseline). Not yet exercised against a live Tinker server.
    tokenizer = base_client.get_tokenizer()
    renderer = renderers.get_renderer("qwen3", tokenizer)
    renderer_no_thinking = renderers.get_renderer("qwen3_disable_thinking", tokenizer)

    def _score_with_progress(samples: list[dict], sampler, label: str) -> float:
        if not samples:
            return 0.0
        total = 0.0
        start = time.monotonic()
        for i, s in enumerate(samples, start=1):
            constraints = s["constraints"]
            if constraints:
                text = sampler.generate(messages=s["messages"], max_tokens=max_tokens,
                                        temperature=temperature, reasoning=True)
                total += native_score(text, constraints[0])
            if i % 50 == 0 or i == len(samples):
                elapsed = time.monotonic() - start
                print(f"  {label}: {i}/{len(samples)} scored ({elapsed:.1f}s elapsed)")
        return total / len(samples)

    clients = {"base": base_client, "ft": ft_client}
    results: dict = {"params": {"limit": limit, "max_tokens": max_tokens,
                                  "temperature": temperature}}
    for name, client in clients.items():
        sc = TinkerSampler(client, renderer, renderer_no_thinking)
        conflict_score = _score_with_progress(conflict, sc, f"{name}/conflict")
        aligned_score = _score_with_progress(aligned, sc, f"{name}/aligned")
        print(f"{name}: conflict={conflict_score:.3f} (n={len(conflict)}) "
              f"aligned={aligned_score:.3f} (n={len(aligned)})")
        results[name] = {
            "conflict": conflict_score, "aligned": aligned_score,
            "n_conflict": len(conflict), "n_aligned": len(aligned),
        }

    with open("training/iheval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
