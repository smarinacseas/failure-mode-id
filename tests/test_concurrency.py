"""Concurrency behavior: locked appends, the wall-clock deadline guard, the
threaded generate stage, monitor accounting under contention, and the batch
judge transport. Mocked clients only — zero real API calls.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

import config
import pipeline.generate as generate
import pipeline.grade as grade
from conftest import make_cfg
from pipeline._io import append_jsonl, read_jsonl, retry, write_jsonl
from pipeline.generate import GenerationDeadlineExceeded
from pipeline.monitor import RecordingSink, RunMonitor, StatusSink, WorkPlan


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def _chunk(content=None, reasoning=None, finish=None):
    delta = SimpleNamespace(content=content, reasoning=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


def _ok_chunks(text="RESP"):
    return [_chunk(reasoning="thinking..."), _chunk(content=text, finish="stop")]


class FakeStream:
    """Streamed-response stand-in.

    hang=True   — __next__ blocks until close() is called, then raises (the
                  half-open-socket shape: a blocked read that only a close
                  from another thread can unstick).
    endless>0   — yields a chunk every `endless` seconds forever (the
                  slow-trickle shape that resets httpx's read timer).
    """

    def __init__(self, chunks=(), hang=False, endless=0.0):
        self._chunks = list(chunks)
        self._hang = hang
        self._endless = endless
        self._closed = threading.Event()
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        self._closed.set()

    def __iter__(self):
        return self

    def __next__(self):
        if self._hang:
            self._closed.wait()
            raise RuntimeError("read raised: stream closed under a blocked recv")
        if self._closed.is_set():
            raise RuntimeError("read raised: stream closed")
        if self._chunks:
            return self._chunks.pop(0)
        if self._endless:
            time.sleep(self._endless)
            return _chunk(reasoning="x")
        raise StopIteration


def _router(stream_for_prompt):
    """Fake OpenRouter client; stream_for_prompt(prompt) -> FakeStream."""
    calls = []

    class _Completions:
        @staticmethod
        def create(**kw):
            assert kw.get("stream") is True
            calls.append(kw)
            return stream_for_prompt(kw["messages"][0]["content"])

    router = SimpleNamespace(chat=SimpleNamespace(completions=_Completions))
    router.calls = calls
    return router


def _gen_setup(tmp_path, monkeypatch, prompts, **cfg_kw):
    data = tmp_path / "prompts.jsonl"
    write_jsonl(data, [{"id": p, "prompt": p, "criteria": []} for p in prompts])
    monkeypatch.setattr(generate, "DATA_JSONL", data)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    return make_cfg(**cfg_kw)


def _gen_monitor(n_items):
    return RunMonitor(WorkPlan.for_step("generate", n_items, 1), sinks=[RecordingSink()])


def _no_backoff(monkeypatch):
    """retry() keeps its full semantics (attempt counting, retriability,
    monitor events) but sleeps zero between attempts."""
    monkeypatch.setattr("pipeline._io.time.sleep", lambda s: None)


# --------------------------------------------------------------------------- #
# 1. Interleaving: locked appends never tear or interleave lines
# --------------------------------------------------------------------------- #
def test_append_jsonl_no_interleaved_lines_under_contention(tmp_path):
    path = tmp_path / "out.jsonl"
    n_threads, per_thread = 8, 25
    barrier = threading.Barrier(n_threads)

    def worker(tid):
        barrier.wait(timeout=10)                    # maximize simultaneous writes
        for i in range(per_thread):
            payload = f"{tid}:{i}:" + ("x" * (5000 + 700 * tid))
            append_jsonl(path, {"id": f"t{tid}-r{i}", "payload": payload})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_threads * per_thread
    seen = set()
    for line in lines:
        rec = json.loads(line)                      # every line parses whole
        tid, i, x = rec["payload"].split(":", 2)
        assert rec["id"] == f"t{tid}-r{i}"          # payload not cross-contaminated
        assert x == "x" * (5000 + 700 * int(tid))
        seen.add(rec["id"])
    assert len(seen) == n_threads * per_thread      # no duplicates, none lost


def test_generate_workers_write_whole_records(tmp_path, monkeypatch):
    """Four pool workers gated on a barrier so appends overlap; every record
    in the output must parse and carry its own item's payload."""
    prompts = [f"p{i}" for i in range(8)]
    cfg = _gen_setup(tmp_path, monkeypatch, prompts)
    barrier = threading.Barrier(4)

    def fake_one(cfg_, model_id, prompt, **kw):
        barrier.wait(timeout=10)                    # 8 items / 4 workers → 2 waves
        return {"response": f"RESP-{prompt}-" + "y" * 20_000}

    monkeypatch.setattr(generate, "_generate_one", fake_one)
    m = _gen_monitor(len(prompts))
    with m:
        generate.run(cfg, monitor=m)

    written = read_jsonl(cfg.responses_path("m1"))
    assert {r["id"] for r in written} == set(prompts)
    assert len(written) == len(prompts)
    for r in written:
        assert r["response"] == f"RESP-{r['id']}-" + "y" * 20_000
    snap = m.snapshot()
    assert snap["errors"] == 0
    assert snap["stages"][0]["done"] == len(prompts)
    assert snap["current"]["in_flight"] == 0


# --------------------------------------------------------------------------- #
# 2. Deadline guard: trickling chunks and chunk-less hangs both abort
# --------------------------------------------------------------------------- #
def test_deadline_aborts_trickling_stream(tmp_path, monkeypatch):
    """Chunks keep arriving (each would reset an httpx read timer) but the
    elapsed wall clock crosses the deadline → per-chunk check aborts."""
    cfg = make_cfg()
    stream = FakeStream(endless=0.02)
    monkeypatch.setattr(config, "router", _router(lambda p: stream))
    monkeypatch.setattr(generate, "retry", lambda fn, label: fn())   # guard only

    t0 = time.monotonic()
    with pytest.raises(GenerationDeadlineExceeded, match="deadline"):
        generate._generate_one(cfg, "model-1", "p1", deadline_s=0.25)
    elapsed = time.monotonic() - t0
    assert 0.25 <= elapsed < 3.0                    # aborted at the deadline, not later
    assert stream.close_calls >= 1                  # stream actually closed


def test_deadline_aborts_hanging_stream_via_watchdog(tmp_path, monkeypatch):
    """No chunk ever arrives (half-open socket: no bytes, no EOF). The
    per-chunk check can never run; the watchdog must close the stream and
    the blocked read's resulting error is translated to the deadline error."""
    cfg = make_cfg()
    stream = FakeStream(hang=True)
    monkeypatch.setattr(config, "router", _router(lambda p: stream))
    monkeypatch.setattr(generate, "retry", lambda fn, label: fn())

    t0 = time.monotonic()
    with pytest.raises(GenerationDeadlineExceeded, match="deadline"):
        generate._generate_one(cfg, "model-1", "p1", deadline_s=0.3)
    elapsed = time.monotonic() - t0
    assert 0.3 <= elapsed < 3.0
    assert stream.close_calls >= 1                  # the watchdog fired


def test_deadline_abort_routes_into_retry_then_error(tmp_path, monkeypatch):
    """A deadline abort is a retry event; exhausted retries become one item
    error — identical to any other transient failure's accounting."""
    cfg = _gen_setup(tmp_path, monkeypatch, ["p1"])
    monkeypatch.setattr(generate, "GENERATION_DEADLINE_S", 0.15)
    monkeypatch.setattr(config, "router", _router(lambda p: FakeStream(hang=True)))
    _no_backoff(monkeypatch)

    m = _gen_monitor(1)
    with m:
        generate.run(cfg, monitor=m)

    snap = m.snapshot()
    assert snap["retries"] == 4                     # 5 attempts → 4 backoffs
    assert snap["errors"] == 1                      # then one continue-on-error item
    assert snap["stages"][0]["done"] == 1           # bar completes regardless
    assert read_jsonl(cfg.responses_path("m1")) == []


def test_deadline_error_is_retriable_by_retry():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise GenerationDeadlineExceeded("wall-clock deadline exceeded (test)")
        return "ok"

    assert retry(flaky, label="t", base_delay=0.0) == "ok"
    assert calls["n"] == 3


# --------------------------------------------------------------------------- #
# 3. Continue-on-error under concurrency
# --------------------------------------------------------------------------- #
def test_one_erroring_worker_does_not_affect_siblings(tmp_path, monkeypatch):
    prompts = ["p1", "p2", "p3", "p4"]
    cfg = _gen_setup(tmp_path, monkeypatch, prompts)

    def fake_one(cfg_, model_id, prompt, **kw):
        if prompt == "p2":
            raise RuntimeError("boom after retries")
        return {"response": f"RESP-{prompt}"}

    monkeypatch.setattr(generate, "_generate_one", fake_one)
    m = _gen_monitor(len(prompts))
    with m:
        generate.run(cfg, monitor=m)

    written = read_jsonl(cfg.responses_path("m1"))
    assert {r["id"] for r in written} == {"p1", "p3", "p4"}
    for r in written:
        assert r["response"] == f"RESP-{r['id']}"   # siblings untouched by the failure
    snap = m.snapshot()
    assert snap["errors"] == 1
    assert snap["stages"][0]["done"] == 4


# --------------------------------------------------------------------------- #
# 4. Kill mid-stage → resume via done_ids completes exactly the missing set
# --------------------------------------------------------------------------- #
def test_abrupt_kill_then_resume_completes_missing_without_duplicates(tmp_path, monkeypatch):
    prompts = [f"p{i}" for i in range(6)]
    cfg = _gen_setup(tmp_path, monkeypatch, prompts)

    def dying(cfg_, model_id, prompt, **kw):
        if prompt == "p3":
            raise KeyboardInterrupt                 # abrupt termination mid-stage
        return {"response": f"RESP-{prompt}"}

    monkeypatch.setattr(generate, "_generate_one", dying)
    with pytest.raises(KeyboardInterrupt):
        m = _gen_monitor(len(prompts))
        with m:
            generate.run(cfg, monitor=m)

    after_kill = read_jsonl(cfg.responses_path("m1"))
    survived = {r["id"] for r in after_kill}
    assert "p3" not in survived                     # the killed item wrote nothing
    assert len(after_kill) == len(survived)         # whatever landed, landed once

    resumed_calls: list[str] = []

    def healthy(cfg_, model_id, prompt, **kw):
        resumed_calls.append(prompt)
        return {"response": f"RESP-{prompt}"}

    monkeypatch.setattr(generate, "_generate_one", healthy)
    m2 = _gen_monitor(len(prompts))
    with m2:
        generate.run(cfg, monitor=m2)

    final = read_jsonl(cfg.responses_path("m1"))
    assert {r["id"] for r in final} == set(prompts)             # complete
    assert len(final) == len(prompts)                           # no duplicates
    assert set(resumed_calls) == set(prompts) - survived        # exactly the missing set
    assert m2.snapshot()["stages"][0]["done"] == len(prompts)   # resumed + processed


# --------------------------------------------------------------------------- #
# 5. Monitor accounting under concurrent mutation
# --------------------------------------------------------------------------- #
def test_monitor_counts_exact_under_concurrent_mutation(tmp_path):
    progress = tmp_path / "progress.json"
    n_threads, per_thread = 8, 25
    total = n_threads * per_thread
    m = RunMonitor(WorkPlan.for_step("generate", total, 1),
                   experiment="E99-test",
                   sinks=[RecordingSink(), StatusSink(progress, min_interval=0.0)])
    max_in_flight = []

    def worker(tid):
        for i in range(per_thread):
            m.item_start(model=f"t{tid}", prompt_id=f"i{i}")
            max_in_flight.append(m.snapshot()["current"]["in_flight"])
            m.record_retry(f"t{tid}")
            if i == 0:
                m.record_error(f"t{tid} boom")
            m.item_done(model=f"t{tid}", prompt_id=f"i{i}")

    with m:
        m.start_stage("generate", total=total)
        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

    snap = m.snapshot()
    assert snap["retries"] == total                 # one per item, none lost
    assert snap["errors"] == n_threads              # one per thread
    assert snap["stages"][0]["done"] == total
    assert snap["current"]["in_flight"] == 0        # everything checked back in
    assert max(max_in_flight) >= 1                  # concurrency actually happened
    heartbeat = json.loads(progress.read_text())    # final write is valid JSON
    assert heartbeat["retries"] == total and heartbeat["errors"] == n_threads


# --------------------------------------------------------------------------- #
# 6. Wedged worker: one item blocked past the deadline, siblings complete
# --------------------------------------------------------------------------- #
def test_wedged_worker_is_retried_then_errored_never_stalls(tmp_path, monkeypatch):
    prompts = ["p1", "p2", "p3", "p4"]
    cfg = _gen_setup(tmp_path, monkeypatch, prompts)
    monkeypatch.setattr(generate, "GENERATION_DEADLINE_S", 0.3)
    _no_backoff(monkeypatch)

    def stream_for(prompt):
        if prompt == "p2":
            return FakeStream(hang=True)            # the wedge: no bytes, no EOF
        return FakeStream(chunks=_ok_chunks(f"RESP-{prompt}"))

    monkeypatch.setattr(config, "router", _router(stream_for))

    t0 = time.monotonic()
    m = _gen_monitor(len(prompts))
    with m:
        generate.run(cfg, monitor=m)                # must return — never stall
    elapsed = time.monotonic() - t0
    assert elapsed < 15.0                           # 5 aborts × 0.3s, not forever

    written = read_jsonl(cfg.responses_path("m1"))
    assert {r["id"] for r in written} == {"p1", "p3", "p4"}
    snap = m.snapshot()
    assert snap["retries"] == 4                     # wedged item retried...
    assert snap["errors"] == 1                      # ...then errored, loudly
    assert snap["stages"][0]["done"] == 4           # stage finished
    assert snap["current"]["in_flight"] == 0


# --------------------------------------------------------------------------- #
# Batch judge transport
# --------------------------------------------------------------------------- #
def _verdict_json(n=1):
    return json.dumps([{"index": i, "verdict": "PASS", "reason": "ok"}
                       for i in range(1, n + 1)])


def _succeeded(cid, text, stop_reason="end_turn"):
    msg = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                          stop_reason=stop_reason)
    return SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="succeeded", message=msg))


def _errored(cid, etype="api_error"):
    return SimpleNamespace(custom_id=cid,
                           result=SimpleNamespace(type="errored",
                                                  error=SimpleNamespace(type=etype)))


class FakeBatches:
    """Scripted Message Batches endpoint. `script` is a list of per-submission
    dicts: {"polls": [status, ...], "results": callable(requests) -> [result]}.
    Submission N consumes script[N]."""

    def __init__(self, script):
        self.script = script
        self.created: list[list[dict]] = []          # requests per create() call
        self._by_id: dict[str, dict] = {}

    def create(self, requests):
        idx = len(self.created)
        self.created.append(list(requests))
        bid = f"batch_{idx}"
        entry = dict(self.script[idx])
        entry["polls"] = list(entry.get("polls") or ["ended"])
        entry["requests"] = list(requests)
        self._by_id[bid] = entry
        return SimpleNamespace(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        entry = self._by_id[bid]
        status = entry["polls"].pop(0) if len(entry["polls"]) > 1 else entry["polls"][0]
        rc = SimpleNamespace(processing=0 if status == "ended" else len(entry["requests"]),
                             succeeded=0, errored=0)
        return SimpleNamespace(id=bid, processing_status=status, request_counts=rc)

    def results(self, bid):
        entry = self._by_id[bid]
        return iter(entry["results"](entry["requests"]))


def _grade_setup(tmp_path, monkeypatch, script, prompts=("p1", "p2"), responses=None,
                 **cfg_kw):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg(**cfg_kw)
    write_jsonl(tmp_path / "prompts.jsonl",
                [{"id": p, "prompt": f"prompt-{p}", "criteria": ["c1"]} for p in prompts])
    monkeypatch.setattr(grade, "DATA_JSONL", tmp_path / "prompts.jsonl")
    for key in cfg.candidates:
        recs = responses if responses is not None else [
            {"id": p, "response": f"resp-{p}"} for p in prompts]
        write_jsonl(cfg.responses_path(key), recs)

    fake = FakeBatches(script)
    monkeypatch.setattr(grade, "anthropic",
                        SimpleNamespace(messages=SimpleNamespace(batches=fake)))
    sleeps: list[float] = []
    monkeypatch.setattr(grade, "_sleep", sleeps.append)
    return cfg, fake, sleeps


def _grade_monitor(n):
    return RunMonitor(WorkPlan.for_step("grade", n, 1), sinks=[RecordingSink()])


def test_batch_grade_submit_poll_collect_with_partial_error_retry(tmp_path, monkeypatch):
    """p1 succeeds in round 1; p2 comes back errored and is resubmitted alone
    in round 2, where it succeeds. Both records land; polling waited on the
    60s cadence; heartbeat carried the batch counts."""
    def round1(requests):
        cids = [r["custom_id"] for r in requests]
        return [_succeeded(cids[0], _verdict_json()), _errored(cids[1])]

    def round2(requests):
        return [_succeeded(requests[0]["custom_id"], _verdict_json())]

    script = [
        {"polls": ["in_progress", "ended"], "results": round1},
        {"polls": ["ended"], "results": round2},
    ]
    cfg, fake, sleeps = _grade_setup(tmp_path, monkeypatch, script)

    m = _grade_monitor(2)
    with m:
        grade.run(cfg, monitor=m)

    # Round 1 submitted both cells; round 2 exactly the failed one.
    assert [r["custom_id"] for r in fake.created[0]] == ["m1__p1", "m1__p2"]
    assert [r["custom_id"] for r in fake.created[1]] == ["m1__p2"]
    assert sleeps and all(s == grade.POLL_INTERVAL_S for s in sleeps)

    graded = read_jsonl(cfg.grades_path("claude-fable-5", "m1"))
    assert {g["id"] for g in graded} == {"p1", "p2"}
    assert all(g["verdicts"][0]["verdict"] == "PASS" for g in graded)
    snap = m.snapshot()
    assert snap["errors"] == 0                       # retry rescued the errored item
    assert snap["stages"][0]["done"] == 2
    assert snap["batch"] == {"submitted": 3, "pending": 0, "collected": 2, "errored": 1}


def test_batch_grade_persistent_item_error_becomes_all_fail_record(tmp_path, monkeypatch):
    """An item errored in both rounds maps to the existing continue-on-error
    accounting: an all-FAIL record with the reason, plus one recorded error —
    never a silent drop, never a crash."""
    def all_errored(requests):
        return [_errored(r["custom_id"]) for r in requests]

    script = [{"results": all_errored}, {"results": all_errored}]
    cfg, fake, _ = _grade_setup(tmp_path, monkeypatch, script, prompts=("p1",))

    m = _grade_monitor(1)
    with m:
        grade.run(cfg, monitor=m)

    graded = read_jsonl(cfg.grades_path("claude-fable-5", "m1"))
    assert [g["id"] for g in graded] == ["p1"]
    assert graded[0]["verdicts"][0]["verdict"] == "FAIL"
    assert graded[0]["verdicts"][0]["reason"].startswith("judge_parse_error")
    snap = m.snapshot()
    assert snap["errors"] == 1
    assert snap["stages"][0]["done"] == 1
    assert snap["batch"]["errored"] == 2             # one errored result per round


def test_batch_grade_truncated_result_retries_like_sequential(tmp_path, monkeypatch):
    """stop_reason=max_tokens (judge spent the budget thinking) is the batch
    analog of the sequential truncation retry: one resubmission, then an
    all-FAIL judge_truncated record."""
    def truncated(requests):
        return [_succeeded(r["custom_id"], "", stop_reason="max_tokens") for r in requests]

    script = [{"results": truncated}, {"results": truncated}]
    cfg, fake, _ = _grade_setup(tmp_path, monkeypatch, script, prompts=("p1",))

    m = _grade_monitor(1)
    with m:
        grade.run(cfg, monitor=m)

    assert len(fake.created) == 2                    # retried once
    graded = read_jsonl(cfg.grades_path("claude-fable-5", "m1"))
    assert graded[0]["verdicts"][0]["reason"].startswith("judge_truncated")
    assert m.snapshot()["errors"] == 1


def test_batch_grade_refusal_is_terminal_without_resubmission(tmp_path, monkeypatch):
    """stop_reason=refusal is sticky (7/7 identical outcomes on E05 Fable ×
    CIF-006): the batch path must write the terminal judge_refusal record
    immediately instead of burning a second batch round. Refusals are
    documented artifacts, not pipeline errors — accounting matches the
    sequential path (no monitor error)."""
    def refused(requests):
        return [_succeeded(r["custom_id"], "", stop_reason="refusal") for r in requests]

    script = [{"results": refused}, {"results": refused}]
    cfg, fake, _ = _grade_setup(tmp_path, monkeypatch, script, prompts=("p1",))

    m = _grade_monitor(1)
    with m:
        grade.run(cfg, monitor=m)

    assert len(fake.created) == 1                    # no second round
    graded = read_jsonl(cfg.grades_path("claude-fable-5", "m1"))
    assert [g["id"] for g in graded] == ["p1"]
    assert graded[0]["verdicts"][0]["verdict"] == "FAIL"
    assert graded[0]["verdicts"][0]["reason"].startswith("judge_refusal")
    snap = m.snapshot()
    assert snap["errors"] == 0
    assert snap["stages"][0]["done"] == 1


def test_batch_grade_resume_submits_only_missing_ids(tmp_path, monkeypatch):
    """done_ids re-derived from the output JSONL exactly as before: a resume
    submits a batch containing only the missing ids, and never duplicates."""
    def rest(requests):
        return [_succeeded(r["custom_id"], _verdict_json()) for r in requests]

    cfg, fake, _ = _grade_setup(tmp_path, monkeypatch, [{"results": rest}],
                                prompts=("p1", "p2", "p3"))
    write_jsonl(cfg.grades_path("claude-fable-5", "m1"),
                [{"id": "p1", "verdicts": [{"index": 1, "verdict": "PASS", "reason": ""}]}])

    m = _grade_monitor(3)
    with m:
        grade.run(cfg, monitor=m)

    assert len(fake.created) == 1
    assert [r["custom_id"] for r in fake.created[0]] == ["m1__p2", "m1__p3"]
    graded = read_jsonl(cfg.grades_path("claude-fable-5", "m1"))
    assert [g["id"] for g in graded].count("p1") == 1            # not re-graded
    assert {g["id"] for g in graded} == {"p1", "p2", "p3"}
    assert m.snapshot()["stages"][0]["done"] == 3                # 1 resumed + 2 new


def test_batch_custom_ids_disambiguate_candidates_within_one_judge(tmp_path, monkeypatch):
    """One batch per judge spans every candidate, so the prompt id alone would
    collide; the cell's stable id (candidate__prompt) routes each result to
    the right per-candidate grades file."""
    def ok(requests):
        return [_succeeded(r["custom_id"], _verdict_json()) for r in requests]

    cfg, fake, _ = _grade_setup(
        tmp_path, monkeypatch, [{"results": ok}], prompts=("p1",),
        candidates={"m1": "model-1", "m2": "model-2"},
    )

    m = _grade_monitor(1)
    with m:
        grade.run(cfg, monitor=m)

    assert sorted(r["custom_id"] for r in fake.created[0]) == ["m1__p1", "m2__p1"]
    for key in ("m1", "m2"):
        graded = read_jsonl(cfg.grades_path("claude-fable-5", key))
        assert [g["id"] for g in graded] == ["p1"]


def test_batch_request_params_match_sequential_treatment(tmp_path, monkeypatch):
    """Grading params and judge-blindness are the frozen treatment: the batch
    request must carry exactly the sequential call's params, and nothing in
    the judge-visible content may name the candidate."""
    def ok(requests):
        return [_succeeded(r["custom_id"], _verdict_json()) for r in requests]

    cfg, fake, _ = _grade_setup(
        tmp_path, monkeypatch, [{"results": ok}], prompts=("p1",),
        candidates={"candidate-x": "model-1"},
    )

    m = _grade_monitor(1)
    with m:
        grade.run(cfg, monitor=m)

    (req,) = fake.created[0]
    params = req["params"]
    assert params["model"] == "claude-fable-5"
    assert params["max_tokens"] == config.JUDGE_MAX_TOKENS
    assert params["thinking"] == {"type": "adaptive"}
    assert params["system"] == grade.JUDGE_SYSTEM
    user_msg = params["messages"][0]["content"]
    assert "prompt-p1" in user_msg and "resp-p1" in user_msg and "c1" in user_msg
    assert "candidate-x" not in user_msg             # blindness: key only in custom_id
    assert params["messages"][0]["role"] == "user"


def test_batch_grade_skips_empty_response_without_submitting(tmp_path, monkeypatch):
    def ok(requests):
        return [_succeeded(r["custom_id"], _verdict_json()) for r in requests]

    cfg, fake, _ = _grade_setup(
        tmp_path, monkeypatch, [{"results": ok}], prompts=("p1", "p2"),
        responses=[{"id": "p1", "response": "resp-p1"}, {"id": "p2", "response": "   "}],
    )

    m = _grade_monitor(2)
    with m:
        grade.run(cfg, monitor=m)

    assert [r["custom_id"] for r in fake.created[0]] == ["m1__p1"]   # p2 never submitted
    snap = m.snapshot()
    assert snap["errors"] == 1                       # skip is loud, exactly as before
    assert snap["stages"][0]["done"] == 2
