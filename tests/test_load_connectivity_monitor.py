import pandas as pd
import pytest

import config
import pipeline.connectivity as connectivity
import pipeline.load as load
from conftest import make_cfg
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan


def test_load_emits_events(tmp_path, monkeypatch):
    df = pd.DataFrame([
        {"benchmark_id": "p1", "prompt": "a", "use_case": "u",
         "instruction_type": "Negative", "prompt_style": "Direct", "criterion_1": "c1"},
        {"benchmark_id": "p2", "prompt": "b", "use_case": "u",
         "instruction_type": "Multistep", "prompt_style": "Direct", "criterion_1": "c1"},
    ])
    monkeypatch.setattr(load.pd, "read_excel", lambda *_a, **_k: df)
    monkeypatch.setattr(load, "DATA_JSONL", tmp_path / "out.jsonl")

    m = RunMonitor(WorkPlan.for_step("load", 2, 1), sinks=[RecordingSink()])
    with m:
        n = load.run(limit=None, monitor=m)
    assert n == 2
    assert m.snapshot()["stages"][0]["done"] == 2


def test_connectivity_pings_all_and_reports(monkeypatch):
    class _Msg:
        def __init__(self, text): self.content = [type("B", (), {"text": text})()]
    class _Choice:
        def __init__(self, text): self.message = type("M", (), {"content": text})()
    class _FakeRouter:
        class chat:
            class completions:
                @staticmethod
                def create(**_k):
                    return type("R", (), {"choices": [_Choice("ok")]})()
    class _FakeAnthropic:
        class messages:
            @staticmethod
            def create(**_k):
                return _Msg("ok")

    monkeypatch.setattr(connectivity, "router", _FakeRouter)
    monkeypatch.setattr(connectivity, "anthropic", _FakeAnthropic)
    monkeypatch.setattr(config, "CANDIDATES", {"m1": "x", "m2": "y"})
    monkeypatch.setattr(config, "JUDGES", ["claude-j1"])
    # Connectivity also pings config.DIAGNOSE_CHAIN; pin it to the same key as
    # the lone panel judge so this test stays scoped to candidate/judge counting
    # rather than the diagnose-chain dedup (which test_connectivity_clients.py
    # covers directly).
    monkeypatch.setattr(config, "DIAGNOSE_CHAIN", ("claude-j1",))

    m = RunMonitor(WorkPlan.for_step("connectivity", 2, 2), sinks=[RecordingSink()])
    with m:
        connectivity.run(monitor=m)
    # 2 candidates + 1 judge = 3 pings
    assert m.snapshot()["stages"][0]["done"] == 3


def test_connectivity_reports_failures_and_exits(monkeypatch):
    class _FailRouter:
        class chat:
            class completions:
                @staticmethod
                def create(**_k):
                    raise RuntimeError("bad model id")

    class _OkAnthropic:
        class messages:
            @staticmethod
            def create(**_k):
                b = type("B", (), {"text": "ok"})()
                return type("R", (), {"content": [b]})()

    monkeypatch.setattr(connectivity, "router", _FailRouter)
    monkeypatch.setattr(connectivity, "anthropic", _OkAnthropic)
    monkeypatch.setattr(config, "CANDIDATES", {"m1": "x"})
    monkeypatch.setattr(config, "JUDGES", ["claude-j1"])
    # See test_connectivity_pings_all_and_reports: keep the diagnose chain
    # deduped against the single panel judge here too.
    monkeypatch.setattr(config, "DIAGNOSE_CHAIN", ("claude-j1",))

    m = RunMonitor(WorkPlan.for_step("connectivity", 1, 1), sinks=[RecordingSink()])
    with pytest.raises(SystemExit) as exc:
        with m:
            connectivity.run(monitor=m)
    assert exc.value.code == 2                        # fails loudly on bad IDs
    assert m.snapshot()["errors"] == 1                # the one failed candidate ping recorded
    assert m.snapshot()["stages"][0]["done"] == 2     # candidate + judge both item_done'd before exit


def test_connectivity_pings_cfg_models(monkeypatch):
    pinged = []
    monkeypatch.setattr(connectivity, "_ping_candidate", lambda k, mid: pinged.append(("c", k, mid)) or "ok")
    monkeypatch.setattr(connectivity, "_ping_judge", lambda spec: pinged.append(("j", spec.key)) or "ok")
    cfg = make_cfg(candidates={"x": "prov/x"}, judges=("claude-fable-5",))
    m = RunMonitor(WorkPlan.for_step("connectivity", 0, 1), sinks=[RecordingSink()])
    with m:
        connectivity.run(cfg, monitor=m)
    assert ("c", "x", "prov/x") in pinged and ("j", "claude-fable-5") in pinged
