import pandas as pd

import pipeline.connectivity as connectivity
import pipeline.load as load
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
    monkeypatch.setattr(connectivity, "CANDIDATES", {"m1": "x", "m2": "y"})

    m = RunMonitor(WorkPlan.for_step("connectivity", 2, 2), sinks=[RecordingSink()])
    with m:
        connectivity.run(monitor=m)
    # 2 candidates + 1 judge = 3 pings
    assert m.snapshot()["stages"][0]["done"] == 3
