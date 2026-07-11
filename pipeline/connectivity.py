"""Pre-flight connectivity check. One tiny call to each candidate + the judge.

Fails fast on the first error so the user can fix model IDs against
openrouter.ai/models or anthropic docs before paying for a batch run.
"""

from __future__ import annotations

from contextlib import nullcontext

import config
from config import anthropic, router
from pipeline.monitor import RunMonitor, build_monitor
from pipeline.run_config import RunConfig


def _ping_candidate(key: str, model_id: str) -> str:
    resp = router.chat.completions.create(
        model=model_id, temperature=0, max_tokens=4,
        messages=[{"role": "user", "content": "Say 'ok'."}],
    )
    return (resp.choices[0].message.content or "").strip()


def _ping_judge(spec) -> str:
    if spec.client == "openrouter":
        resp = router.chat.completions.create(
            model=spec.model, max_tokens=4,
            messages=[{"role": "user", "content": "Say 'ok'."}],
        )
        return (resp.choices[0].message.content or "").strip()
    resp = anthropic.messages.create(
        model=spec.model, max_tokens=4,
        messages=[{"role": "user", "content": "Say 'ok'."}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()


def run(cfg: RunConfig | None = None, monitor: RunMonitor | None = None) -> None:
    from pipeline.run_config import resolve_judge
    candidates = cfg.candidates if cfg is not None else dict(config.CANDIDATES)
    # Bare-invocation defaults resolve registry keys to specs so each judge is
    # pinged against its OWN provider, never blindly against Anthropic.
    specs = list(cfg.judges) if cfg is not None else [resolve_judge(k) for k in config.JUDGES]
    # A run also calls out to judges that aren't necessarily panel members:
    # the classifier fallback chain (cfg.classifier_chain) and the diagnose
    # fallback chain (config.DIAGNOSE_CHAIN). Ping every distinct one of those
    # too, so a typo'd classifier or diagnose model fails here instead of
    # mid-run. cfg is None only for the bare/default invocation, which has no
    # per-experiment classifier override to add.
    extra = list(cfg.classifier_chain) if cfg is not None else []
    extra += [resolve_judge(k) for k in config.DIAGNOSE_CHAIN]
    # Dedup by key: a judge already pinged (as a panel member or an earlier
    # chain entry) is never pinged twice, even if the chains disagree on the
    # underlying JudgeSpec object identity.
    seen = {s.key for s in specs}
    for s in extra:
        if s.key not in seen:
            specs.append(s)
            seen.add(s.key)
    ctx = nullcontext(monitor) if monitor is not None else build_monitor("connectivity", 0)
    with ctx as mon:
        mon.start_stage("connectivity", total=len(candidates) + len(specs))
        failures: list[tuple[str, str, str]] = []
        for key, model_id in candidates.items():
            mon.item_start(model=key, prompt_id=model_id)
            try:
                txt = _ping_candidate(key, model_id)
                mon.note(f"candidate {key} ({model_id}) ok ({txt[:40]!r})")
            except Exception as e:  # noqa: BLE001
                mon.record_error(f"candidate {key} ({model_id}): {type(e).__name__}: {e}")
                failures.append((key, model_id, f"{type(e).__name__}: {e}"))
            mon.item_done()

        for spec in specs:
            mon.item_start(model="judge", prompt_id=f"{spec.key} ({spec.model})")
            try:
                txt = _ping_judge(spec)
                mon.note(f"judge {spec.key} ({spec.model}) ok ({txt[:40]!r})")
            except Exception as e:  # noqa: BLE001
                mon.record_error(f"judge {spec.key} ({spec.model}): {type(e).__name__}: {e}")
                failures.append(("judge", spec.key, f"{type(e).__name__}: {e}"))
            mon.item_done()
        mon.end_stage()

        if failures:
            for key, model_id, err in failures:
                mon.note(f"FAILED {key}: {model_id} → {err}")
            raise SystemExit(2)


if __name__ == "__main__":
    run()
