"""IHEval conflict-set gate: deterministic pass-rate per checkpoint.

Reuses the same constraint checker as the RL reward, so the gate measures
exactly what training optimized. `score_conflict` is a pure function, unit-
tested against `type`-keyed constraints (training/reward.py's own 6-type
taxonomy), and stays that way regardless of how the real data is wired up.

Known open issue, NOT solved here: the real VerIH test split's `constraints`
are single-element lists holding the raw `gt` verifier spec, keyed
`func_name` (24 IFEval-style verifier names — see training/data.py's
docstring), not `type`. training/reward.py's `check_constraint` does not
consume `func_name` (it does `constraint["type"]`), so calling
`score_conflict(load_verih(...), ...)` against the real test split will
raise (KeyError: 'type') on every sample until a func_name→type adapter (or
a `check_constraint` extension) is decided — see the task-6 report for the
full func_name inventory and mapping gap. That decision is out of scope for
this module and is resolved at the training-phase boundary before Step 5's
live run.
"""

from __future__ import annotations

from training.reward import reward


def score_conflict(samples: list[dict], sampler) -> float:
    if not samples:
        return 0.0
    total = 0.0
    for s in samples:
        resp = sampler.generate(messages=s["messages"], max_tokens=8000,
                                temperature=0.6, reasoning=True)  # 0.6: reasoning-on, no greedy loops
        total += reward(s["constraints"], resp)
    return total / len(samples)


def main() -> None:
    import json

    import tinker
    from tinker_cookbook import renderers

    from training.data import load_verih
    from training.proxy import TinkerSampler

    ckpts = json.loads(open("training/checkpoints.json").read())
    samples = load_verih("data/verih/RLVR/dataset/verih/test.json")

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

    clients = {"base": base_client, "ft": ft_client}
    for name, client in clients.items():
        sc = TinkerSampler(client, renderer, renderer_no_thinking)
        print(f"{name}: conflict_pass_rate={score_conflict(samples, sc):.3f}")


if __name__ == "__main__":
    main()
