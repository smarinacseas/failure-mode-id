"""Connectivity must ping every distinct judge spec used anywhere in a run —
panel ∪ classifier_chain ∪ diagnose chain — each on its own client, deduped
by key so a judge shared across roles (e.g. the classifier reusing a panel
member, or the diagnose judge already sitting in the panel) is pinged once."""

import pipeline.connectivity as connectivity
from conftest import make_cfg
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan


def test_pings_every_distinct_spec_on_its_client(monkeypatch):
    pinged = []
    monkeypatch.setattr(connectivity, "_ping_candidate", lambda k, m: "ok")
    monkeypatch.setattr(connectivity, "_ping_judge",
                        lambda spec: pinged.append((spec.key, spec.client)) or "ok")
    cfg = make_cfg(judges=("claude-fable-5",
                           {"key": "gpt-5", "client": "openrouter", "model": "openai/gpt-5.2"}),
                   classifier_chain=("claude-opus-4-8",))
    # Explicit RecordingSink monitor (matches every other stage test in this repo)
    # so the connectivity default-monitor path doesn't write real files under
    # outputs/logs and outputs/progress.json during a unit test.
    m = RunMonitor(WorkPlan.for_step("connectivity", 0, 1), sinks=[RecordingSink()])
    with m:
        connectivity.run(cfg, monitor=m)
    keys = [k for k, _ in pinged]
    assert keys.count("gpt-5") == 1 and ("gpt-5", "openrouter") in pinged
    assert "claude-opus-4-8" in keys          # chain member outside the panel
    assert len(keys) == len(set(keys))        # dedup by key
