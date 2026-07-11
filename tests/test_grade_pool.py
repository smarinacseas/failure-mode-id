"""Grade transport: OpenRouter worker pool, concurrent with Anthropic batches.

Task 3 split the plan by judge client (anth_plan / or_plan); this task adds
the pool that grinds OpenRouter cells WHILE the Anthropic batch submits,
polls, and collects — one `run()` invocation, no barrier between the two
transports (see `_run_pool` in pipeline/grade.py). `_run_batch` is
monkeypatched to `_run_sequential` below so these tests exercise the split +
pool wiring only; the real Anthropic batch API has its own coverage in
test_concurrency.py.
"""

import json
import threading

import config
import pipeline.grade as grade
from conftest import make_cfg
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _setup(tmp_path, monkeypatch, judges):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    data = tmp_path / "cc.jsonl"
    _write_jsonl(data, [
        {"id": "P1", "prompt": "p1", "criteria": ["c1", "c2"]},
        {"id": "P2", "prompt": "p2", "criteria": ["c1"]},
    ])
    monkeypatch.setattr(grade, "DATA_JSONL", data)
    cfg = make_cfg(judges=judges, candidates={"m1": "prov/m1"}, limit=None)
    # P2 is a stored loop-failure (empty response): the mechanical all-FAIL
    # short-circuit must fire for it on EVERY judge, on EITHER transport,
    # without ever reaching call_json.
    _write_jsonl(cfg.responses_path("m1"), [
        {"id": "P1", "response": "answer one"},
        {"id": "P2", "response": "", "loop_failure": {"channel": "reasoning",
                                                      "period": 7, "onset": 100}},
    ])
    return cfg


def _monitor(n):
    return RunMonitor(WorkPlan.for_step("grade", n, 1), sinks=[RecordingSink()])


def test_mixed_panel_batch_plus_pool(tmp_path, monkeypatch):
    calls = []

    def fake_call(spec, system, user, label):
        calls.append(spec.key)
        return ('[{"index":1,"verdict":"PASS","reason":""},'
                '{"index":2,"verdict":"FAIL","reason":"r"}]', "stop")
    monkeypatch.setattr(grade, "call_json", fake_call)

    # Anthropic batch transport stubbed out: route anthropic judges through
    # the sequential path so the test only exercises the split + pool.
    monkeypatch.setattr(grade, "_run_batch",
                        lambda cfg, mon, by_id, plan: grade._run_sequential(cfg, mon, by_id, plan))

    cfg = _setup(tmp_path, monkeypatch,
                 judges=("claude-fable-5",
                         {"key": "gpt-5", "client": "openrouter", "model": "openai/gpt-5.2"}))
    m = _monitor(4)
    with m:
        grade.run(cfg, monitor=m)

    for jk in ("claude-fable-5", "gpt-5"):
        rows = [json.loads(l) for l in cfg.grades_path(jk, "m1").read_text().splitlines()]
        by_id = {r["id"]: r for r in rows}
        assert by_id["P1"]["verdicts"][0]["verdict"] == "PASS"
        # loop-failure cell: mechanical all-FAIL, no judge call for P2
        assert by_id["P2"]["verdicts"][0]["reason"].startswith("loop_failure")
    assert all(k in ("claude-fable-5", "gpt-5") for k in calls)
    assert calls.count("gpt-5") == 1     # only P1 hit the model (P2 short-circuits)
    assert calls.count("claude-fable-5") == 1
    snap = m.snapshot()
    assert snap["errors"] == 0
    assert snap["stages"][0]["done"] == 4    # 2 prompts * 2 judges, one candidate


def test_pool_resume_skips_done(tmp_path, monkeypatch):
    def fake_call(spec, system, user, label):
        return ('[{"index":1,"verdict":"PASS","reason":""},'
                '{"index":2,"verdict":"PASS","reason":""}]', "stop")
    monkeypatch.setattr(grade, "call_json", fake_call)
    monkeypatch.setattr(grade, "_run_batch",
                        lambda cfg, mon, by_id, plan: grade._run_sequential(cfg, mon, by_id, plan))
    cfg = _setup(tmp_path, monkeypatch,
                 judges=({"key": "gpt-5", "client": "openrouter", "model": "openai/gpt-5.2"},))

    m1 = _monitor(2)
    with m1:
        grade.run(cfg, monitor=m1)
    first = cfg.grades_path("gpt-5", "m1").read_text()

    m2 = _monitor(2)
    with m2:
        grade.run(cfg, monitor=m2)   # resume: nothing new appended
    assert cfg.grades_path("gpt-5", "m1").read_text() == first
    assert m2.snapshot()["stages"][0]["done"] == 2   # both cells already-done, no new calls


def test_or_pool_and_anthropic_batch_run_concurrently(tmp_path, monkeypatch):
    """The property the two tests above CANNOT see (both monkeypatch
    `_run_batch` straight to `_run_sequential`, which happens to produce
    identical output whether the OR cells ran before, after, or alongside
    it): OpenRouter cells must be IN FLIGHT while the Anthropic batch
    transport is still running on the main thread — `_run_pool` submits
    before `_run_batch` is invoked, with no barrier between them.

    Proof: a 2-party threading.Barrier rendezvous inside call_json. One
    call comes from the pool worker thread (the OpenRouter cell), the other
    from the main thread (the stubbed-batch's call, via _run_sequential).
    They can only both reach the barrier if the pool was already running
    when the batch call fired. Under the pre-Task-4 code (`_run_batch(...)`
    then `_run_sequential(or_plan)` strictly in series) the first call
    would block alone until the barrier's own timeout, and — since
    `_grade_one` swallows the resulting BrokenBarrierError as a retryable
    error and tries again (also breaking immediately) — the cell would land
    as an all-FAIL judge_parse_error instead of the canned PASS verdict
    asserted below."""
    barrier = threading.Barrier(2, timeout=5.0)
    seen_keys: list[str] = []

    def fake_call(spec, system, user, label):
        seen_keys.append(spec.key)
        barrier.wait()   # both parties must arrive, or this raises/times out
        return '[{"index":1,"verdict":"PASS","reason":""}]', "stop"
    monkeypatch.setattr(grade, "call_json", fake_call)
    monkeypatch.setattr(grade, "_run_batch",
                        lambda cfg, mon, by_id, plan: grade._run_sequential(cfg, mon, by_id, plan))

    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    data = tmp_path / "cc.jsonl"
    _write_jsonl(data, [{"id": "P1", "prompt": "p1", "criteria": ["c1"]}])
    monkeypatch.setattr(grade, "DATA_JSONL", data)
    cfg = make_cfg(judges=("claude-fable-5",
                           {"key": "gpt-5", "client": "openrouter", "model": "openai/gpt-5.2"}),
                   candidates={"m1": "prov/m1"}, limit=None)
    _write_jsonl(cfg.responses_path("m1"), [{"id": "P1", "response": "answer"}])

    m = _monitor(2)
    with m:
        grade.run(cfg, monitor=m)

    assert sorted(seen_keys) == ["claude-fable-5", "gpt-5"]
    for jk in ("claude-fable-5", "gpt-5"):
        rows = [json.loads(l) for l in cfg.grades_path(jk, "m1").read_text().splitlines()]
        assert rows[0]["verdicts"][0]["verdict"] == "PASS"   # broken barrier -> FAIL instead
