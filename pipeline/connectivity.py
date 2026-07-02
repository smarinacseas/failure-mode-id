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


def _ping_judge(judge: str) -> str:
    resp = anthropic.messages.create(
        model=judge, max_tokens=4,
        messages=[{"role": "user", "content": "Say 'ok'."}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()


def run(cfg: RunConfig | None = None, monitor: RunMonitor | None = None) -> None:
    candidates = cfg.candidates if cfg is not None else dict(config.CANDIDATES)
    judges = list(cfg.judges) if cfg is not None else list(config.JUDGES)
    ctx = nullcontext(monitor) if monitor is not None else build_monitor("connectivity", 0)
    with ctx as mon:
        mon.start_stage("connectivity", total=len(candidates) + len(judges))
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

        for judge in judges:
            mon.item_start(model="judge", prompt_id=judge)
            try:
                txt = _ping_judge(judge)
                mon.note(f"judge ({judge}) ok ({txt[:40]!r})")
            except Exception as e:  # noqa: BLE001
                mon.record_error(f"judge ({judge}): {type(e).__name__}: {e}")
                failures.append(("judge", judge, f"{type(e).__name__}: {e}"))
            mon.item_done()
        mon.end_stage()

        if failures:
            for key, model_id, err in failures:
                mon.note(f"FAILED {key}: {model_id} → {err}")
            raise SystemExit(2)


if __name__ == "__main__":
    run()
