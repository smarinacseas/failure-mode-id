# Experiment Progress Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the developer a live terminal view (per-stage bars, current model·prompt, %/ETA, retries/errors) of a running eval, backed by a persistent log file and a `progress.json` heartbeat readable from any shell.

**Architecture:** An event-emitting `RunMonitor` (in `pipeline/monitor.py`) owns all counting/timing/ETA and fans each state change out to pluggable **sinks** — `ConsoleSink` (rich live bars), `LogSink` (timestamped file), `StatusSink` (`progress.json`). Pipeline steps emit semantic events (`start_stage`/`item_start`/`item_done`/`record_error`) and never import rich or touch a file. `main.py` builds a `WorkPlan` and drives one monitor per invocation; a new `status` verb renders the heartbeat.

**Tech Stack:** Python ≥3.14, `rich` (live display), `pytest` (tests), stdlib `logging`/`json`.

## Global Constraints

- **Python ≥ 3.14** (matches `requires-python`).
- **New runtime dependency:** `rich>=13`. **New dev dependency:** `pytest`. No others.
- **Steps never import `rich`** — they only call `RunMonitor` methods.
- **Paths:** heartbeat at `outputs/progress.json`; logs at `outputs/logs/<slug-or-untagged>-<YYYYMMDD-HHMMSS>.log`. Both fall under the existing `outputs/` gitignore (only `outputs/experiments/` is committed).
- **Single active run** — `progress.json` reflects one run; the pipeline is sequential by design.
- **Overall total** = sum of all stage item totals (full run = 4+75+225+225+75+1+1 = 606). Overall % uses `int()` truncation; ETA is time-weighted.
- **Every commit message ends with the trailer:** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Work happens on branch `feat/experiment-progress-monitoring` (already checked out).

---

### Task A: Project setup — dependencies, config paths, test scaffolding

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.py:61-72` (paths section)
- Create: `tests/conftest.py`
- Create: `tests/test_config_paths.py`

**Interfaces:**
- Produces: `config.LOGS_DIR: Path`, `config.PROGRESS_PATH: Path`. `rich` and `pytest` importable. `tests/` importable with project root on `sys.path`.

- [ ] **Step 1: Add dependencies**

Run:
```bash
uv add rich
uv add --dev pytest
```
Expected: `pyproject.toml` gains `rich>=…` under `dependencies` and a `pytest` dev entry; `uv.lock` updates; both install.

- [ ] **Step 2: Configure pytest path + dummy env for imports**

`config.py` reads `os.environ["OPENROUTER_API_KEY"]` and constructs `Anthropic()` at import time, so any test importing a `pipeline.*` module needs both keys present. Add pytest config to `pyproject.toml` (append at end):

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Create `tests/conftest.py`:

```python
"""Test setup: dummy API keys so importing `config` never KeyErrors.

`config.py` reads OPENROUTER_API_KEY and builds an Anthropic() client at
import time. Tests never make real calls (network is monkeypatched), but the
imports must succeed, so seed placeholder keys before anything imports config.
"""

import os

os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
```

- [ ] **Step 3: Add the two paths to `config.py`**

In `config.py`, in the `# --- Paths ---` block (after `PROMPTS_DIR = ...`, near line 62), add:

```python
LOGS_DIR = OUTPUTS_DIR / "logs"
PROGRESS_PATH = OUTPUTS_DIR / "progress.json"
```

- [ ] **Step 4: Write the failing test**

Create `tests/test_config_paths.py`:

```python
from pathlib import Path

import config


def test_new_monitoring_paths_exist():
    assert isinstance(config.LOGS_DIR, Path)
    assert isinstance(config.PROGRESS_PATH, Path)
    assert config.LOGS_DIR == config.OUTPUTS_DIR / "logs"
    assert config.PROGRESS_PATH == config.OUTPUTS_DIR / "progress.json"


def test_rich_and_pytest_available():
    import rich  # noqa: F401
```

- [ ] **Step 5: Run the test**

Run: `uv run pytest tests/test_config_paths.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock config.py tests/conftest.py tests/test_config_paths.py
git commit -m "feat: add rich/pytest deps + monitoring config paths

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B: `Stage` + `WorkPlan` (state model, no I/O)

**Files:**
- Create: `pipeline/monitor.py`
- Create: `tests/test_workplan.py`

**Interfaces:**
- Produces:
  - `Stage(name: str, total: int, done: int = 0, state: str = "pending", current_model: str|None = None, current_prompt_id: str|None = None)` with props `.pct: float`, `.remaining: int`, `.rate_s: float|None`, and methods `.record_duration(seconds: float) -> None`, `.eta_s() -> float|None`. `state ∈ {"pending","running","done"}`.
  - `WorkPlan(stages: list[Stage])` with classmethod `for_step(step: str, limit: int, n_candidates: int) -> WorkPlan` and `get(name: str) -> Stage`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workplan.py`:

```python
from pipeline.monitor import Stage, WorkPlan


def test_stage_pct_and_remaining():
    s = Stage(name="generate", total=225, done=45)
    assert s.pct == 20.0
    assert s.remaining == 180


def test_stage_pct_zero_total():
    assert Stage(name="validate", total=0).pct == 0.0


def test_stage_rate_and_eta_from_durations():
    s = Stage(name="grade", total=10, done=2)
    assert s.rate_s is None and s.eta_s() is None
    for d in (2.0, 4.0):        # mean 3.0s/item, 8 remaining
        s.record_duration(d)
    assert s.rate_s == 3.0
    assert s.eta_s() == 24.0


def test_stage_duration_window_caps_at_20():
    s = Stage(name="generate", total=100)
    for i in range(30):
        s.record_duration(float(i))
    assert len(s._durations) == 20
    assert s._durations[0] == 10.0     # oldest 10 dropped


def test_workplan_for_all_full_run():
    plan = WorkPlan.for_step("all", limit=75, n_candidates=3)
    totals = {s.name: s.total for s in plan.stages}
    assert totals == {
        "connectivity": 4, "load": 75, "generate": 225, "grade": 225,
        "classify": 75, "validate": 1, "aggregate": 1,
    }
    assert sum(totals.values()) == 606
    assert [s.name for s in plan.stages][0] == "connectivity"


def test_workplan_single_step():
    plan = WorkPlan.for_step("grade", limit=10, n_candidates=3)
    assert [s.name for s in plan.stages] == ["grade"]
    assert plan.get("grade").total == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workplan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.monitor'`.

- [ ] **Step 3: Write the implementation**

Create `pipeline/monitor.py`:

```python
"""Live progress monitoring for the eval pipeline.

A single `RunMonitor` owns all counting/timing and fans state changes out to
pluggable sinks (console bars, log file, progress.json). Pipeline steps emit
semantic events and know nothing about rendering. See
docs/superpowers/specs/2026-07-01-experiment-progress-monitoring-design.md.

This file is built up across several tasks; Task B adds Stage + WorkPlan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Stage order + how each stage's total is derived from (limit L, n_candidates M).
_STEP_ORDER = ("connectivity", "load", "generate", "grade", "classify", "validate", "aggregate")


def _stage_total(name: str, limit: int, m: int) -> int:
    return {
        "connectivity": m + 1,
        "load": limit,
        "generate": limit * m,
        "grade": limit * m,
        "classify": limit,
        "validate": 1,
        "aggregate": 1,
    }[name]


@dataclass
class Stage:
    name: str
    total: int
    done: int = 0
    state: str = "pending"                     # pending | running | done
    current_model: str | None = None
    current_prompt_id: str | None = None
    _durations: list[float] = field(default_factory=list, repr=False)

    WINDOW = 20

    @property
    def pct(self) -> float:
        return round(100.0 * self.done / self.total, 1) if self.total else 0.0

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.done)

    @property
    def rate_s(self) -> float | None:
        """Mean seconds/item over the rolling window, or None if no data yet."""
        if not self._durations:
            return None
        return sum(self._durations) / len(self._durations)

    def record_duration(self, seconds: float) -> None:
        self._durations.append(seconds)
        if len(self._durations) > self.WINDOW:
            self._durations.pop(0)

    def eta_s(self) -> float | None:
        r = self.rate_s
        return r * self.remaining if r is not None else None


@dataclass
class WorkPlan:
    stages: list[Stage]

    @classmethod
    def for_step(cls, step: str, limit: int, n_candidates: int) -> "WorkPlan":
        names = list(_STEP_ORDER) if step == "all" else [step]
        return cls([Stage(name=n, total=_stage_total(n, limit, n_candidates)) for n in names])

    def get(self, name: str) -> Stage:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_workplan.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/monitor.py tests/test_workplan.py
git commit -m "feat: add Stage + WorkPlan progress state model

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task C: `RunMonitor` core — events, snapshot, render, ACTIVE

**Files:**
- Modify: `pipeline/monitor.py` (append)
- Create: `tests/test_monitor.py`

**Interfaces:**
- Consumes: `Stage`, `WorkPlan` (Task B).
- Produces:
  - `class Sink` base with no-op `update(monitor)`, `log(monitor, level, message)`, `close(monitor)`.
  - `class RecordingSink(Sink)` — test helper; `.snapshots: list[dict]`, `.logs: list[tuple[str,str]]`.
  - `RunMonitor(plan, experiment=None, sinks=None)`: context manager. Methods `start_stage(name, total, already_done=0)`, `item_start(model=None, prompt_id=None)`, `item_done()`, `end_stage()`, `record_retry(label)`, `record_error(context)`, `note(message)`; props `elapsed_s`; `snapshot() -> dict`.
  - Module: `ACTIVE: RunMonitor|None`, `note_retry(label)`, `note_error(context)`.
  - `render_lines(snapshot: dict) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_monitor.py`:

```python
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan, render_lines
import pipeline.monitor as monitor


def _mon(step="all", limit=2, n=1):
    return RunMonitor(WorkPlan.for_step(step, limit, n), experiment="E99-test",
                      sinks=[RecordingSink()])


def test_item_events_advance_stage_and_overall():
    rec = RecordingSink()
    m = RunMonitor(WorkPlan.for_step("generate", limit=2, n_candidates=1), sinks=[rec])
    with m:
        m.start_stage("generate", total=2, already_done=0)
        for pid in ("p1", "p2"):
            m.item_start(model="m1", prompt_id=pid)
            m.item_done()
        m.end_stage()
    snap = m.snapshot()
    assert snap["stages"][0]["done"] == 2
    assert snap["stages"][0]["state"] == "done"
    assert snap["overall"]["done"] == 2
    assert snap["state"] == "done"
    assert rec.snapshots, "sink received updates"


def test_already_done_seeds_resumed_progress():
    m = _mon(step="generate", limit=10, n=1)
    with m:
        m.start_stage("generate", total=10, already_done=6)
        assert m.snapshot()["stages"][0]["done"] == 6


def test_retry_and_error_counters_via_module_helpers():
    m = _mon()
    with m:
        assert monitor.ACTIVE is m
        monitor.note_retry("openrouter:x")
        monitor.note_error("generate m1 p1: boom")
    snap = m.snapshot()
    assert snap["retries"] == 1 and snap["errors"] == 1
    assert monitor.ACTIVE is None            # cleared on exit


def test_eta_is_time_weighted_across_remaining_stages():
    m = RunMonitor(WorkPlan.for_step("all", limit=1, n_candidates=1), sinks=[])
    with m:
        m.start_stage("generate", total=1)
        m.plan.get("generate").record_duration(10.0)   # 10s/item observed
        eta = m.snapshot()["eta_s"]
    # remaining items across not-done stages × observed/fallback rate (10s):
    # load1 + generate0(after done? not ended) ... at least > 0 and finite
    assert eta is not None and eta > 0


def test_error_state_on_exception_in_context():
    m = _mon()
    try:
        with m:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert m.snapshot()["state"] == "error"


def test_render_lines_contains_key_fields():
    snap = {
        "experiment": "E02-v1-75p", "state": "running",
        "elapsed_s": 4340, "eta_s": 10800,
        "current": {"stage": "generate", "model": "qwen-397b", "prompt_id": "prompt-50"},
        "overall": {"done": 227, "total": 606, "pct": 37.5},
        "stages": [
            {"name": "load", "state": "done", "done": 75, "total": 75, "pct": 100.0},
            {"name": "generate", "state": "running", "done": 148, "total": 225, "pct": 65.8},
            {"name": "grade", "state": "pending", "done": 0, "total": 225, "pct": 0.0},
        ],
        "retries": 3, "errors": 0,
    }
    out = render_lines(snap)
    assert "E02-v1-75p" in out
    assert "148/225" in out
    assert "65%" in out                # int() truncation of 65.8
    assert "37%" in out                # overall
    assert "qwen-397b" in out and "prompt-50" in out
    assert "retries: 3" in out and "errors: 0" in out
    assert "✓ done" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_monitor.py -v`
Expected: FAIL with `ImportError: cannot import name 'RunMonitor'`.

- [ ] **Step 3: Write the implementation (append to `pipeline/monitor.py`)**

Add these imports to the top of `pipeline/monitor.py` (merge with the existing `from __future__` / dataclass imports):

```python
import os
import time
from datetime import datetime, timezone
```

Append to `pipeline/monitor.py`:

```python
# --------------------------------------------------------------------------- #
# Sinks (base + test recorder). Concrete file/console sinks land in later tasks.
# --------------------------------------------------------------------------- #
class Sink:
    """Receives monitor updates. Subclasses override only what they need."""

    def update(self, monitor: "RunMonitor") -> None: ...
    def log(self, monitor: "RunMonitor", level: str, message: str) -> None: ...
    def close(self, monitor: "RunMonitor") -> None: ...


class RecordingSink(Sink):
    """In-memory sink for tests: captures every snapshot + log line."""

    def __init__(self) -> None:
        self.snapshots: list[dict] = []
        self.logs: list[tuple[str, str]] = []

    def update(self, monitor: "RunMonitor") -> None:
        self.snapshots.append(monitor.snapshot())

    def log(self, monitor: "RunMonitor", level: str, message: str) -> None:
        self.logs.append((level, message))


# --------------------------------------------------------------------------- #
# Module-level active monitor — lets deep helpers (e.g. _io.retry) report
# without threading the monitor through every call signature.
# --------------------------------------------------------------------------- #
ACTIVE: "RunMonitor | None" = None


def note_retry(label: str) -> None:
    if ACTIVE is not None:
        ACTIVE.record_retry(label)


def note_error(context: str) -> None:
    if ACTIVE is not None:
        ACTIVE.record_error(context)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunMonitor:
    def __init__(self, plan: WorkPlan, experiment: str | None = None,
                 sinks: list[Sink] | None = None) -> None:
        self.plan = plan
        self.experiment = experiment
        self.sinks = sinks if sinks is not None else []
        self.state = "idle"
        self.retries = 0
        self.errors = 0
        self.started_at = ""
        self._start_monotonic: float | None = None
        self._item_start: float | None = None
        self._current: Stage | None = None

    # -- lifecycle ---------------------------------------------------------- #
    def __enter__(self) -> "RunMonitor":
        global ACTIVE
        ACTIVE = self
        self.state = "running"
        self.started_at = _now_iso()
        self._start_monotonic = time.monotonic()
        self._emit()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        global ACTIVE
        self.state = "error" if exc_type else "done"
        self._emit()
        for s in self.sinks:
            s.close(self)
        ACTIVE = None
        return False                          # never suppress exceptions

    # -- events ------------------------------------------------------------- #
    def start_stage(self, name: str, total: int, already_done: int = 0) -> None:
        st = self.plan.get(name)
        st.total = total
        st.done = min(already_done, total)
        st.state = "running"
        self._current = st
        self._log("INFO", f"stage {name} started (total={total}, resumed={st.done})")
        self._emit()

    def item_start(self, model: str | None = None, prompt_id: str | None = None) -> None:
        if self._current is None:
            return
        self._current.current_model = model
        self._current.current_prompt_id = prompt_id
        self._item_start = time.monotonic()
        self._emit()

    def item_done(self) -> None:
        st = self._current
        if st is None:
            return
        if self._item_start is not None:
            st.record_duration(time.monotonic() - self._item_start)
            self._item_start = None
        st.done = min(st.done + 1, st.total)
        self._log("INFO", f"{st.name} {st.current_model} {st.current_prompt_id} done")
        self._emit()

    def end_stage(self) -> None:
        st = self._current
        if st is not None:
            st.state = "done"
            st.done = st.total
            st.current_model = st.current_prompt_id = None
        self._emit()

    def record_retry(self, label: str) -> None:
        self.retries += 1
        self._log("WARNING", f"retry {label}")
        self._emit()

    def record_error(self, context: str) -> None:
        self.errors += 1
        self._log("ERROR", context)
        self._emit()

    def note(self, message: str) -> None:
        self._log("NOTE", message)
        self._emit()

    # -- math + snapshot ---------------------------------------------------- #
    @property
    def elapsed_s(self) -> float:
        return 0.0 if self._start_monotonic is None else time.monotonic() - self._start_monotonic

    def _overall(self) -> tuple[int, int]:
        done = sum(s.done for s in self.plan.stages)
        total = sum(s.total for s in self.plan.stages)
        return done, total

    def eta_s(self) -> float | None:
        fallback = self._current.rate_s if self._current is not None else None
        total = 0.0
        seen = False
        for s in self.plan.stages:
            if s.state == "done":
                continue
            rate = s.rate_s if s.rate_s is not None else fallback
            if rate is None:
                continue
            seen = True
            total += rate * s.remaining
        return total if seen else None

    def snapshot(self) -> dict:
        cur = self._current
        done, total = self._overall()
        eta = self.eta_s()
        return {
            "experiment": self.experiment,
            "state": self.state,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": _now_iso(),
            "elapsed_s": round(self.elapsed_s, 1),
            "eta_s": round(eta, 1) if eta is not None else None,
            "current": {
                "stage": cur.name if cur else None,
                "model": cur.current_model if cur else None,
                "prompt_id": cur.current_prompt_id if cur else None,
            },
            "overall": {"done": done, "total": total,
                        "pct": round(100.0 * done / total, 1) if total else 0.0},
            "stages": [
                {"name": s.name, "state": s.state, "done": s.done,
                 "total": s.total, "pct": s.pct}
                for s in self.plan.stages
            ],
            "retries": self.retries,
            "errors": self.errors,
        }

    # -- fan-out ------------------------------------------------------------ #
    def _emit(self) -> None:
        for s in self.sinks:
            s.update(self)

    def _log(self, level: str, message: str) -> None:
        for s in self.sinks:
            s.log(self, level, message)


# --------------------------------------------------------------------------- #
# Rendering — shared by ConsoleSink (Task E) and `main.py status` (Task L).
# --------------------------------------------------------------------------- #
def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m"
    return f"{sec}s"


def _bar(pct: float, width: int = 16) -> str:
    filled = max(0, min(width, int(round(width * pct / 100.0))))
    return "█" * filled + "░" * (width - filled)


def render_lines(snap: dict) -> str:
    o = snap["overall"]
    cur = snap.get("current") or {}
    header = (f"{snap.get('experiment') or 'untagged'}   {snap['state']} · "
              f"{int(o['pct'])}% · elapsed {_fmt_dur(snap['elapsed_s'])} · "
              f"ETA {_fmt_dur(snap['eta_s'])}")
    lines = [header, ""]
    for st in snap["stages"]:
        name = st["name"]
        if st["state"] == "done":
            lines.append(f"{name:<12} ✓ done   {st['done']}/{st['total']}")
        elif st["state"] == "pending":
            lines.append(f"{name:<12} pending  {st['done']}/{st['total']}")
        else:
            suffix = ""
            if cur.get("stage") == name and cur.get("model"):
                suffix = f" · {cur['model']} · {cur.get('prompt_id') or ''}"
            lines.append(f"{name:<10}{_bar(st['pct'])}  {st['done']}/{st['total']}  "
                         f"{int(st['pct'])}%{suffix}")
    lines.append("")
    lines.append(f"retries: {snap['retries']}   errors: {snap['errors']}   "
                 f"elapsed: {_fmt_dur(snap['elapsed_s'])}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_monitor.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/monitor.py tests/test_monitor.py
git commit -m "feat: add RunMonitor core, snapshot, render, ACTIVE helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task D: File sinks — `StatusSink` + `LogSink`

**Files:**
- Modify: `pipeline/monitor.py` (append)
- Create: `tests/test_file_sinks.py`

**Interfaces:**
- Consumes: `Sink`, `RunMonitor` (Task C).
- Produces:
  - `StatusSink(path: Path, min_interval: float = 1.0)` — writes `snapshot()` to `path` atomically, throttled; forces a write on `close`.
  - `LogSink(path: Path)` — a `logging.Logger` + `FileHandler`; `log()` writes `<ts> <LEVEL> <message>`; `close()` flushes/detaches. `NOTE` maps to `INFO`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_file_sinks.py`:

```python
import json

from pipeline.monitor import LogSink, RunMonitor, StatusSink, WorkPlan


def test_status_sink_writes_valid_snapshot(tmp_path):
    path = tmp_path / "progress.json"
    m = RunMonitor(WorkPlan.for_step("grade", 2, 1), experiment="E99-test",
                   sinks=[StatusSink(path, min_interval=0.0)])
    with m:
        m.start_stage("grade", total=2)
        m.item_start(model="m1", prompt_id="p1")
        m.item_done()
    snap = json.loads(path.read_text())
    assert snap["experiment"] == "E99-test"
    assert snap["stages"][0]["name"] == "grade"
    assert snap["state"] == "done"                 # forced write on close
    assert set(snap) >= {"overall", "current", "retries", "errors", "eta_s"}


def test_status_sink_throttles(tmp_path):
    path = tmp_path / "progress.json"
    sink = StatusSink(path, min_interval=999.0)     # effectively "never again"
    m = RunMonitor(WorkPlan.for_step("grade", 5, 1), sinks=[sink])
    with m:                                          # __enter__ writes once
        first = path.read_text()
        m.start_stage("grade", total=5)
        m.item_start(); m.item_done()               # throttled — file unchanged
        assert path.read_text() == first
    # close() forces a final write regardless of throttle
    assert json.loads(path.read_text())["state"] == "done"


def test_log_sink_records_events(tmp_path):
    path = tmp_path / "run.log"
    m = RunMonitor(WorkPlan.for_step("grade", 1, 1), sinks=[LogSink(path)])
    with m:
        m.start_stage("grade", total=1)
        m.record_error("grade m1 p1: boom")
        m.note("skipped 2 incomplete prompts")
    text = path.read_text()
    assert "stage grade started" in text
    assert "ERROR" in text and "boom" in text
    assert "skipped 2 incomplete prompts" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_file_sinks.py -v`
Expected: FAIL with `ImportError: cannot import name 'StatusSink'`.

- [ ] **Step 3: Write the implementation (append to `pipeline/monitor.py`)**

Add to the imports at the top of `pipeline/monitor.py`:

```python
import json
import logging
from pathlib import Path
```

Append to `pipeline/monitor.py`:

```python
class StatusSink(Sink):
    """Write snapshot() to a JSON heartbeat, throttled, with atomic replace."""

    def __init__(self, path: Path, min_interval: float = 1.0) -> None:
        self.path = path
        self.min_interval = min_interval
        self._last: float | None = None         # None → the first update always writes

    def _write(self, monitor: "RunMonitor") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(monitor.snapshot(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self.path)              # atomic: readers never see a partial file
        self._last = time.monotonic()

    def update(self, monitor: "RunMonitor") -> None:
        if self._last is None or time.monotonic() - self._last >= self.min_interval:
            self._write(monitor)

    def close(self, monitor: "RunMonitor") -> None:
        self._write(monitor)                    # always flush the terminal state


_LEVELS = {"NOTE": logging.INFO, "INFO": logging.INFO,
           "WARNING": logging.WARNING, "ERROR": logging.ERROR}


class LogSink(Sink):
    """Append every event to a timestamped log file via stdlib logging."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.logger = logging.getLogger(f"failure_mode_id.monitor.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._handler = logging.FileHandler(path, encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(self._handler)

    def log(self, monitor: "RunMonitor", level: str, message: str) -> None:
        self.logger.log(_LEVELS.get(level, logging.INFO), message)

    def close(self, monitor: "RunMonitor") -> None:
        self._handler.flush()
        self.logger.removeHandler(self._handler)
        self._handler.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_file_sinks.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/monitor.py tests/test_file_sinks.py
git commit -m "feat: add StatusSink (progress.json) + LogSink (log file)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task E: `ConsoleSink` (rich live display) + monitor factories

**Files:**
- Modify: `pipeline/monitor.py` (append)
- Create: `tests/test_console_sink.py`

**Interfaces:**
- Consumes: `Sink`, `RunMonitor`, `render_lines` (Task C), `StatusSink`/`LogSink` (Task D), `config.CANDIDATES`, `config.LOGS_DIR`, `config.PROGRESS_PATH`.
- Produces:
  - `ConsoleSink(console=None)` — rich `Live`; `update()` re-renders bars; `log()` prints `WARNING`/`ERROR`/`NOTE` above the live region; `close()` stops the live and prints a one-line summary. Degrades to plain lines when not a TTY.
  - `default_sinks(experiment: str | None) -> list[Sink]`.
  - `build_monitor(step: str, limit: int, experiment: str | None = None, sinks: list[Sink] | None = None) -> RunMonitor`.
  - `stage_ctx(monitor: RunMonitor | None, step: str, limit_hint: int)` — returns `nullcontext(monitor)` when a monitor is supplied, else a fresh standalone `build_monitor(step, limit_hint)`. **All step tasks (G–J) import this from `pipeline.monitor`.**

- [ ] **Step 1: Write the failing test**

Create `tests/test_console_sink.py`. The rich rendering itself isn't asserted pixel-for-pixel; we drive a `ConsoleSink` over a non-TTY `Console` and confirm it consumes events without error and emits notices.

```python
import io

from rich.console import Console

from pipeline.monitor import (
    ConsoleSink, RunMonitor, WorkPlan, build_monitor, default_sinks,
)


def test_console_sink_runs_without_error_and_prints_notices():
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    m = RunMonitor(WorkPlan.for_step("grade", 2, 1), experiment="E99-test",
                   sinks=[ConsoleSink(console=console)])
    with m:
        m.start_stage("grade", total=2)
        m.item_start(model="m1", prompt_id="p1")
        m.item_done()
        m.record_error("grade m1 p2: boom")        # ERROR notice -> printed
        m.note("heads up")                          # NOTE notice -> printed
    out = buf.getvalue()
    assert "boom" in out
    assert "heads up" in out


def test_default_sinks_shape(tmp_path, monkeypatch):
    import pipeline.monitor as monitor
    monkeypatch.setattr(monitor, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(monitor, "PROGRESS_PATH", tmp_path / "progress.json")
    sinks = default_sinks("E02-v1-75p")
    kinds = {type(s).__name__ for s in sinks}
    assert kinds == {"ConsoleSink", "LogSink", "StatusSink"}


def test_build_monitor_uses_workplan_and_candidate_count(monkeypatch):
    import pipeline.monitor as monitor
    monkeypatch.setattr(monitor, "CANDIDATES", {"a": "x", "b": "y"})
    m = build_monitor("generate", limit=4, experiment=None, sinks=[])
    assert m.plan.get("generate").total == 8      # 4 * 2 candidates
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_console_sink.py -v`
Expected: FAIL with `ImportError: cannot import name 'ConsoleSink'`.

- [ ] **Step 3: Write the implementation (append to `pipeline/monitor.py`)**

Add to the imports at the top of `pipeline/monitor.py`:

```python
from contextlib import nullcontext

from rich.console import Console
from rich.live import Live
from rich.text import Text

from config import CANDIDATES, LOGS_DIR, PROGRESS_PATH
```

Append to `pipeline/monitor.py`:

```python
_NOTICE_LEVELS = {"WARNING": "yellow", "ERROR": "red", "NOTE": "cyan"}


class ConsoleSink(Sink):
    """Live rich bars in the terminal; notices print above the live region."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._live: Live | None = None

    def _render(self, monitor: "RunMonitor") -> Text:
        return Text(render_lines(monitor.snapshot()))

    def update(self, monitor: "RunMonitor") -> None:
        if self._live is None:
            self._live = Live(self._render(monitor), console=self.console,
                              refresh_per_second=8, transient=False)
            self._live.start()
        else:
            self._live.update(self._render(monitor))

    def log(self, monitor: "RunMonitor", level: str, message: str) -> None:
        color = _NOTICE_LEVELS.get(level)
        if color is None:                       # INFO item ticks: bars already show them
            return
        self.console.print(f"[{color}]{level}[/{color}] {message}")

    def close(self, monitor: "RunMonitor") -> None:
        if self._live is not None:
            self._live.update(self._render(monitor))
            self._live.stop()
            self._live = None
        done, total = monitor._overall()
        self.console.print(
            f"{monitor.experiment or 'untagged'} {monitor.state} · {done}/{total} items "
            f"· {monitor.retries} retries · {monitor.errors} errors "
            f"· {_fmt_dur(monitor.elapsed_s)}"
        )


def default_sinks(experiment: str | None) -> list[Sink]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOGS_DIR / f"{experiment or 'untagged'}-{stamp}.log"
    return [ConsoleSink(), LogSink(log_path), StatusSink(PROGRESS_PATH)]


def build_monitor(step: str, limit: int, experiment: str | None = None,
                  sinks: list[Sink] | None = None) -> RunMonitor:
    plan = WorkPlan.for_step(step, limit, len(CANDIDATES))
    return RunMonitor(plan, experiment=experiment,
                      sinks=default_sinks(experiment) if sinks is None else sinks)


def stage_ctx(monitor: "RunMonitor | None", step: str, limit_hint: int):
    """Reuse the caller's monitor, or build a standalone one for this step.

    Steps call this so they run both under `main`'s shared monitor (passed in)
    and standalone (`python -m pipeline.generate` / tests) with a fresh one.
    """
    if monitor is not None:
        return nullcontext(monitor)
    return build_monitor(step, limit_hint)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_console_sink.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the whole suite so far**

Run: `uv run pytest -q`
Expected: PASS (all green).

- [ ] **Step 6: Commit**

```bash
git add pipeline/monitor.py tests/test_console_sink.py
git commit -m "feat: add ConsoleSink live display + monitor factories

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task F: Route `_io.retry` through the monitor; drop dead helper

**Files:**
- Modify: `pipeline/_io.py:39-81` (`retry`), remove `iter_progress` (`:88-93`)
- Create: `tests/test_io_retry.py`

**Interfaces:**
- Consumes: `pipeline.monitor.note_retry` (Task C).
- Produces: `retry()` calls `monitor.note_retry(...)` on each backoff instead of `print`. Signature unchanged.

- [ ] **Step 1: Confirm `iter_progress` is unused**

Run: `grep -rn "iter_progress" --include=*.py .`
Expected: only its definition in `pipeline/_io.py` (no callers) — safe to remove.

- [ ] **Step 2: Write the failing test**

Create `tests/test_io_retry.py`:

```python
import pytest

import pipeline.monitor as monitor
from pipeline._io import retry
from pipeline.monitor import RunMonitor, WorkPlan


def test_retry_reports_to_active_monitor():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 rate limited")
        return "ok"

    m = RunMonitor(WorkPlan.for_step("generate", 1, 1), sinks=[])
    with m:
        assert retry(flaky, label="test", base_delay=0.0) == "ok"
    assert m.snapshot()["retries"] == 2          # two backoffs before success


def test_retry_without_active_monitor_is_safe():
    monitor.ACTIVE = None
    assert retry(lambda: "ok", label="test") == "ok"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_io_retry.py -v`
Expected: FAIL — `retries` is `0` because `retry` still `print`s instead of reporting.

- [ ] **Step 4: Edit `pipeline/_io.py`**

In `retry()`, replace the backoff `print` line:

```python
            delay = base_delay * (2**i) + random.uniform(0, 0.5)
            print(f"  retry {label} in {delay:.1f}s ({type(e).__name__}: {e})")
            time.sleep(delay)
```

with a lazy import + monitor report (lazy import avoids any import cycle):

```python
            delay = base_delay * (2**i) + random.uniform(0, 0.5)
            from pipeline import monitor
            monitor.note_retry(f"{label} in {delay:.1f}s ({type(e).__name__})")
            time.sleep(delay)
```

Delete the now-unused `iter_progress` function (the last def in the file) and its `Iterator`/`Iterable` imports if they become unused (`from typing import Callable, Iterable, Iterator, TypeVar` → `from typing import Callable, TypeVar`).

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_io_retry.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add pipeline/_io.py tests/test_io_retry.py
git commit -m "feat: report retries to the active monitor; drop unused iter_progress

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task G: Instrument `generate` + make it continue-on-error

**Files:**
- Modify: `pipeline/generate.py:49-63` (`run`)
- Create: `tests/test_generate_monitor.py`

**Interfaces:**
- Consumes: `RunMonitor`, `build_monitor` (Tasks C/E).
- Produces: `generate.run(limit=None, monitor=None)`. On exhausted-retry failure of a candidate call: log via `monitor.record_error`, skip writing that response, `item_done()`, continue.

- [ ] **Step 1: Write the failing test**

Create `tests/test_generate_monitor.py`:

```python
import pipeline.generate as generate
from pipeline._io import write_jsonl
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan


def test_generate_continue_on_error(tmp_path, monkeypatch):
    data = tmp_path / "prompts.jsonl"
    write_jsonl(data, [{"id": "p1", "prompt": "a", "criteria": []},
                       {"id": "p2", "prompt": "b", "criteria": []}])
    monkeypatch.setattr(generate, "DATA_JSONL", data)
    monkeypatch.setattr(generate, "RESPONSES_DIR", tmp_path)
    monkeypatch.setattr(generate, "CANDIDATES", {"m1": "model-1"})

    def fake_one(model_id, prompt):
        if prompt == "b":
            raise RuntimeError("boom after retries")
        return "RESP"
    monkeypatch.setattr(generate, "_generate_one", fake_one)

    m = RunMonitor(WorkPlan.for_step("generate", 2, 1), sinks=[RecordingSink()])
    with m:
        generate.run(limit=None, monitor=m)

    from pipeline._io import read_jsonl
    written = read_jsonl(tmp_path / "m1.jsonl")
    assert [r["id"] for r in written] == ["p1"]      # p2 skipped, not written
    snap = m.snapshot()
    assert snap["errors"] == 1
    assert snap["stages"][0]["done"] == 2            # both attempted → bar completes


def test_generate_seeds_already_done(tmp_path, monkeypatch):
    data = tmp_path / "prompts.jsonl"
    write_jsonl(data, [{"id": "p1", "prompt": "a", "criteria": []},
                       {"id": "p2", "prompt": "b", "criteria": []}])
    write_jsonl(tmp_path / "m1.jsonl", [{"id": "p1", "response": "done earlier"}])
    monkeypatch.setattr(generate, "DATA_JSONL", data)
    monkeypatch.setattr(generate, "RESPONSES_DIR", tmp_path)
    monkeypatch.setattr(generate, "CANDIDATES", {"m1": "model-1"})
    monkeypatch.setattr(generate, "_generate_one", lambda m, p: "RESP")

    m = RunMonitor(WorkPlan.for_step("generate", 2, 1), sinks=[RecordingSink()])
    with m:
        generate.run(limit=None, monitor=m)
    # 1 pre-existing + 1 processed == 2 total
    assert m.snapshot()["stages"][0]["done"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate_monitor.py -v`
Expected: FAIL — `generate.run()` has no `monitor` parameter (`TypeError`).

- [ ] **Step 3: Rewrite `generate.run` (and add a helper)**

In `pipeline/generate.py`, add imports near the top (`CANDIDATES` is already imported — keep it):

```python
from pipeline.monitor import RunMonitor, stage_ctx
```

Replace the whole `run` function (`:49-63`) with:

```python
def run(limit: int | None = None, monitor: RunMonitor | None = None) -> None:
    records = limited(read_jsonl(DATA_JSONL), limit)
    if not records:
        raise RuntimeError(f"No records in {DATA_JSONL}. Run `load` first.")

    with stage_ctx(monitor, "generate", len(records)) as mon:
        total = len(records) * len(CANDIDATES)
        plan: list[tuple[str, str, list[dict]]] = []
        already = 0
        for key, model_id in CANDIDATES.items():
            done_ids = {r["id"] for r in read_jsonl(_response_path(key))}
            todo = [r for r in records if r["id"] not in done_ids]
            already += len(done_ids)
            plan.append((key, model_id, todo))

        mon.start_stage("generate", total=total, already_done=already)
        for key, model_id, todo in plan:
            out_path = _response_path(key)
            for rec in todo:
                mon.item_start(model=key, prompt_id=rec["id"])
                try:
                    text = _generate_one(model_id, rec["prompt"])
                except Exception as e:  # noqa: BLE001 — post-retry failure
                    mon.record_error(f"generate {key} {rec['id']}: {type(e).__name__}: {e}")
                    mon.item_done()
                    continue
                append_jsonl(out_path, {"id": rec["id"], "response": text})
                mon.item_done()
        mon.end_stage()
```

`stage_ctx` comes from `pipeline.monitor` (Task E) — Tasks H–J import it the same way.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate_monitor.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/generate.py tests/test_generate_monitor.py
git commit -m "feat: instrument generate + continue-on-error

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task H: Instrument `grade` + `classify`

**Files:**
- Modify: `pipeline/grade.py:97-119` (`run`)
- Modify: `pipeline/classify.py:76-106` (`run`, `_classify_one`)
- Create: `tests/test_grade_classify_monitor.py`

**Interfaces:**
- Consumes: `RunMonitor`, `stage_ctx` (from `pipeline.monitor`, Task E).
- Produces: `grade.run(limit=None, monitor=None)` and `classify.run(limit=None, monitor=None)`, both emitting per-item events; a judge/classifier parse failure calls `monitor.record_error`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_grade_classify_monitor.py`:

```python
import pipeline.classify as classify
import pipeline.grade as grade
from pipeline._io import write_jsonl
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan


def test_grade_emits_events(tmp_path, monkeypatch):
    write_jsonl(tmp_path / "prompts.jsonl",
                [{"id": "p1", "prompt": "a", "criteria": ["c1"]}])
    write_jsonl(tmp_path / "m1.jsonl", [{"id": "p1", "response": "r"}])
    monkeypatch.setattr(grade, "DATA_JSONL", tmp_path / "prompts.jsonl")
    monkeypatch.setattr(grade, "RESPONSES_DIR", tmp_path)
    monkeypatch.setattr(grade, "GRADES_DIR", tmp_path / "grades")
    monkeypatch.setattr(grade, "CANDIDATES", {"m1": "model-1"})
    monkeypatch.setattr(grade, "_grade_one",
                        lambda p, r, c: [{"index": 1, "verdict": "PASS", "reason": ""}])

    m = RunMonitor(WorkPlan.for_step("grade", 1, 1), sinks=[RecordingSink()])
    with m:
        grade.run(limit=None, monitor=m)
    assert m.snapshot()["stages"][0]["done"] == 1


def test_classify_emits_events(tmp_path, monkeypatch):
    write_jsonl(tmp_path / "prompts.jsonl",
                [{"id": "p1", "prompt": "a", "criteria": ["c1", "c2"]}])
    monkeypatch.setattr(classify, "DATA_JSONL", tmp_path / "prompts.jsonl")
    monkeypatch.setattr(classify, "CRITERIA_TAGS_PATH", tmp_path / "tags.jsonl")
    monkeypatch.setattr(classify, "_classify_one",
                        lambda criteria: [{"index": 1, "verifiability": "auto",
                                           "gameable": False, "reward_hack": "",
                                           "ambiguous": False}])
    m = RunMonitor(WorkPlan.for_step("classify", 1, 1), sinks=[RecordingSink()])
    with m:
        classify.run(limit=None, monitor=m)
    assert m.snapshot()["stages"][0]["done"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grade_classify_monitor.py -v`
Expected: FAIL — neither `run` accepts `monitor`.

- [ ] **Step 3: Rewrite `grade.run`**

In `pipeline/grade.py` add imports:

```python
from pipeline.monitor import RunMonitor, stage_ctx
```

Replace `run` (`:97-119`) with:

```python
def run(limit: int | None = None, monitor: RunMonitor | None = None) -> None:
    records = limited(read_jsonl(DATA_JSONL), limit)
    if not records:
        raise RuntimeError(f"No records in {DATA_JSONL}. Run `load` first.")
    by_id = {r["id"]: r for r in records}

    with stage_ctx(monitor, "grade", len(records)) as mon:
        total = len(records) * len(CANDIDATES)
        plan: list[tuple[str, list[dict]]] = []
        already = 0
        for key in CANDIDATES:
            responses = read_jsonl(RESPONSES_DIR / f"{key}.jsonl")
            done_ids = {r["id"] for r in read_jsonl(_grade_path(key))}
            todo = [r for r in responses if r["id"] in by_id and r["id"] not in done_ids]
            already += len(done_ids)
            if not responses:
                mon.note(f"grade {key}: no responses yet — skipping.")
            plan.append((key, todo))

        mon.start_stage("grade", total=total, already_done=already)
        for key, todo in plan:
            out_path = _grade_path(key)
            for resp_rec in todo:
                rid = resp_rec["id"]
                rec = by_id[rid]
                mon.item_start(model=key, prompt_id=rid)
                verdicts = _grade_one(rec["prompt"], resp_rec["response"], rec["criteria"])
                append_jsonl(out_path, {"id": rid, "verdicts": verdicts})
                if verdicts and str(verdicts[0].get("reason", "")).startswith("judge_parse_error"):
                    mon.record_error(f"grade {key} {rid}: judge parse failure")
                mon.item_done()
        mon.end_stage()
```

- [ ] **Step 4: Rewrite `classify.run` and route `_classify_one` prints**

In `pipeline/classify.py` add imports:

```python
from pipeline.monitor import RunMonitor, stage_ctx
```

Replace `run` (`:93-106`) with:

```python
def run(limit: int | None = None, monitor: RunMonitor | None = None) -> None:
    records = limited(read_jsonl(DATA_JSONL), limit)
    if not records:
        raise RuntimeError(f"No records in {DATA_JSONL}. Run `load` first.")

    with stage_ctx(monitor, "classify", len(records)) as mon:
        done_ids = {r["id"] for r in read_jsonl(CRITERIA_TAGS_PATH)}
        todo = [r for r in records if r["id"] not in done_ids]
        mon.start_stage("classify", total=len(records), already_done=len(done_ids))
        for rec in todo:
            mon.item_start(prompt_id=rec["id"])
            tags = _classify_one(rec["criteria"])
            append_jsonl(CRITERIA_TAGS_PATH, {"id": rec["id"], "tags": tags})
            mon.item_done()
        mon.end_stage()
```

Also in `_classify_one` (`:88-89`), replace the two `print(...)` lines with `note_error` reporting so failures reach the monitor without corrupting the live display:

```python
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt == 2:
                break
    from pipeline.monitor import note_error
    note_error(f"classifier failed twice: {last_err} — defaulting all tags.")
    return _normalize_tags([], len(criteria))
```

(Delete the two old `print(...)` lines at the end of `_classify_one`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_grade_classify_monitor.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add pipeline/grade.py pipeline/classify.py tests/test_grade_classify_monitor.py
git commit -m "feat: instrument grade + classify stages

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task I: Instrument `load` + `connectivity`

**Files:**
- Modify: `pipeline/load.py:35-55` (`run`)
- Modify: `pipeline/connectivity.py` (all of `run`, `_ping_candidate`, `_ping_judge`)
- Create: `tests/test_load_connectivity_monitor.py`

**Interfaces:**
- Consumes: `RunMonitor`, `stage_ctx` (from `pipeline.monitor`, Task E).
- Produces: `load.run(limit=None, monitor=None) -> int` and `connectivity.run(monitor=None)`, both emitting per-item events and routing summaries through `monitor.note`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_load_connectivity_monitor.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_load_connectivity_monitor.py -v`
Expected: FAIL — neither `run` accepts `monitor`.

- [ ] **Step 3: Rewrite `load.run`**

In `pipeline/load.py` add imports:

```python
from pipeline.monitor import RunMonitor, stage_ctx
```

Replace `run` (`:35-55`) with:

```python
def run(limit: int | None = None, monitor: RunMonitor | None = None) -> int:
    df = pd.read_excel(DATA_XLSX)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")
    if limit is not None:
        df = df.head(limit)

    with stage_ctx(monitor, "load", len(df)) as mon:
        mon.start_stage("load", total=len(df))
        records: list[dict] = []
        for _, row in df.iterrows():
            mon.item_start(prompt_id=str(row["benchmark_id"]))
            records.append({
                "id": str(row["benchmark_id"]),
                "prompt": str(row["prompt"]),
                "use_case": str(row["use_case"]),
                "instruction_type": str(row["instruction_type"]),
                "prompt_style": str(row["prompt_style"]),
                "criteria": _clean_criteria(row),
            })
            mon.item_done()
        write_jsonl(DATA_JSONL, records)
        mon.note(f"load: wrote {len(records)} prompts → {DATA_JSONL}")
        mon.end_stage()
    return len(records)
```

- [ ] **Step 4: Rewrite `connectivity`**

Replace the body of `pipeline/connectivity.py` (keep the module docstring) with:

```python
from __future__ import annotations

from contextlib import nullcontext

from config import CANDIDATES, JUDGE, anthropic, router
from pipeline.monitor import RunMonitor, build_monitor


def _ping_candidate(key: str, model_id: str) -> str:
    resp = router.chat.completions.create(
        model=model_id, temperature=0, max_tokens=4,
        messages=[{"role": "user", "content": "Say 'ok'."}],
    )
    return (resp.choices[0].message.content or "").strip()


def _ping_judge() -> str:
    resp = anthropic.messages.create(
        model=JUDGE, max_tokens=4,
        messages=[{"role": "user", "content": "Say 'ok'."}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()


def run(monitor: RunMonitor | None = None) -> None:
    ctx = nullcontext(monitor) if monitor is not None else build_monitor("connectivity", 0)
    with ctx as mon:
        mon.start_stage("connectivity", total=len(CANDIDATES) + 1)
        failures: list[tuple[str, str, str]] = []
        for key, model_id in CANDIDATES.items():
            mon.item_start(model=key, prompt_id=model_id)
            try:
                txt = _ping_candidate(key, model_id)
                mon.note(f"candidate {key} ({model_id}) ok ({txt[:40]!r})")
            except Exception as e:  # noqa: BLE001
                mon.record_error(f"candidate {key} ({model_id}): {type(e).__name__}: {e}")
                failures.append((key, model_id, f"{type(e).__name__}: {e}"))
            mon.item_done()

        mon.item_start(model="judge", prompt_id=JUDGE)
        try:
            txt = _ping_judge()
            mon.note(f"judge ({JUDGE}) ok ({txt[:40]!r})")
        except Exception as e:  # noqa: BLE001
            mon.record_error(f"judge ({JUDGE}): {type(e).__name__}: {e}")
            failures.append(("judge", JUDGE, f"{type(e).__name__}: {e}"))
        mon.item_done()
        mon.end_stage()

        if failures:
            for key, model_id, err in failures:
                mon.note(f"FAILED {key}: {model_id} → {err}")
            raise SystemExit(2)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_load_connectivity_monitor.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add pipeline/load.py pipeline/connectivity.py tests/test_load_connectivity_monitor.py
git commit -m "feat: instrument load + connectivity stages

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task J: Instrument `validate` + `aggregate`

**Files:**
- Modify: `pipeline/validate.py:63-78` (`sample`), `:149-155` (`run`)
- Modify: `pipeline/aggregate.py:191-282` (`run`)
- Create: `tests/test_validate_aggregate_monitor.py`

**Interfaces:**
- Consumes: `RunMonitor`, `stage_ctx` (from `pipeline.monitor`, Task E).
- Produces: `validate.run(mode="sample", monitor=None)` and `aggregate.run(..., monitor=None)`, each wrapping its work in a `validate`/`aggregate` stage and routing prints through `monitor.note`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_validate_aggregate_monitor.py`:

```python
import pipeline.aggregate as aggregate
import pipeline.validate as validate
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan


def test_validate_marks_stage_done(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "sample", lambda mon=None: mon.note("sampled 0"))
    m = RunMonitor(WorkPlan.for_step("validate", 1, 1), sinks=[RecordingSink()])
    with m:
        validate.run(mode="sample", monitor=m)
    assert m.snapshot()["stages"][0]["state"] == "done"


def test_aggregate_marks_stage_done(monkeypatch):
    # Stub the whole extracted body so no real git/file I/O runs; we only
    # assert the monitor stage lifecycle here.
    called = {}
    monkeypatch.setattr(aggregate, "_run",
                        lambda limit, exp, desc, rr, mon: called.setdefault("ok", True))
    m = RunMonitor(WorkPlan.for_step("aggregate", 1, 1), sinks=[RecordingSink()])
    with m:
        aggregate.run(limit=0, monitor=m)
    assert called["ok"] and m.snapshot()["stages"][0]["state"] == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validate_aggregate_monitor.py -v`
Expected: FAIL — neither `run` accepts `monitor`.

- [ ] **Step 3: Instrument `validate`**

In `pipeline/validate.py` add imports:

```python
from pipeline.monitor import RunMonitor, stage_ctx
```

Change `sample()` to accept and use a monitor for its summary line. Replace its signature and the final `print(...)` (`:63`, `:74-78`):

```python
def sample(mon: "RunMonitor | None" = None) -> None:
    pool = _build_pool()
    if not pool:
        raise RuntimeError("No graded rows found. Run `grade` first.")
    n = min(VALIDATE_SAMPLE_TARGET, len(pool))
    rng = random.Random(VALIDATE_SEED)
    rows = rng.sample(pool, n)
    JUDGE_VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_VALIDATION_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    msg = (f"validate sample: wrote {n} rows → {JUDGE_VALIDATION_PATH}; "
           f"fill the `human` field (PASS/FAIL), then run `validate --mode score`.")
    mon.note(msg) if mon is not None else print(msg)
```

Replace `run` (`:149-155`) with:

```python
def run(mode: str = "sample", monitor: RunMonitor | None = None) -> None:
    if mode not in {"sample", "score"}:
        raise ValueError(f"validate mode must be 'sample' or 'score', got {mode!r}")
    with stage_ctx(monitor, "validate", 1) as mon:
        mon.start_stage("validate", total=1)
        mon.item_start()
        if mode == "sample":
            sample(mon)
        else:
            score()
        mon.item_done()
        mon.end_stage()
```

(`score()` keeps its own `print`s — it is a terminal, human-facing report typically run alone, not inside the live `all` display.)

- [ ] **Step 4: Instrument `aggregate.run`**

In `pipeline/aggregate.py` add imports:

```python
from pipeline.monitor import RunMonitor, stage_ctx
```

Wrap the existing `run` body in a stage context and convert its `print(...)` calls to `mon.note(...)`. Change the signature and bracket the body:

```python
def run(
    limit: int | None = None,
    experiment: str | None = None,
    description: str | None = None,
    run_report: str | None = None,
    monitor: RunMonitor | None = None,
) -> None:
    with stage_ctx(monitor, "aggregate", 1) as mon:
        mon.start_stage("aggregate", total=1)
        mon.item_start()
        _run(limit, experiment, description, run_report, mon)
        mon.item_done()
        mon.end_stage()


def _run(limit, experiment, description, run_report, mon) -> None:
    # ... existing body of the old run(), UNCHANGED except every `print(...)`
    # becomes `mon.note(...)`. For example:
    #   print(f"aggregate: wrote {…} → {RESULTS_PATH}")
    # becomes:
    #   mon.note(f"aggregate: wrote {…} → {RESULTS_PATH}")
    # (8 print sites total: the run-report warning, the skip summary + per-item
    #  loop, the results write, the tagged/index writes, the untagged note, and
    #  the manifest-merged note.)
```

Apply that mechanical `print(...) → mon.note(...)` change to the entire moved body. Leave all non-print logic identical.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_validate_aggregate_monitor.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add pipeline/validate.py pipeline/aggregate.py tests/test_validate_aggregate_monitor.py
git commit -m "feat: instrument validate + aggregate stages

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task K: Wire the monitor into `main.py`

**Files:**
- Modify: `main.py` (whole file)
- Create: `tests/test_main_wiring.py`

**Interfaces:**
- Consumes: `build_monitor`, `render_lines` (Tasks C/E); every step's `run(..., monitor=...)`.
- Produces: `main()` drives one `RunMonitor` per invocation (both `all` and single-step) and passes it into each step.

- [ ] **Step 1: Write the failing test**

Create `tests/test_main_wiring.py`:

```python
import main


def test_all_requires_limit(capsys):
    assert main.main(["all"]) == 2
    assert "requires --limit" in capsys.readouterr().err


def test_single_step_builds_monitor_and_passes_it(monkeypatch):
    captured = {}

    def fake_grade_run(limit=None, monitor=None):
        captured["limit"] = limit
        captured["has_monitor"] = monitor is not None
    monkeypatch.setattr(main.grade, "run", fake_grade_run)
    # keep the console quiet in tests: no-op sinks
    monkeypatch.setattr(main, "build_monitor",
                        lambda step, limit, experiment=None: _NoopMonitor(step))

    assert main.main(["grade", "--limit", "5"]) == 0
    assert captured == {"limit": 5, "has_monitor": True}


class _NoopMonitor:
    def __init__(self, step): self.step = step
    def __enter__(self): return self
    def __exit__(self, *a): return False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_main_wiring.py -v`
Expected: FAIL — `build_monitor` isn't referenced in `main` yet (`AttributeError` on the monkeypatch) / steps aren't passed a monitor.

- [ ] **Step 3: Rewrite `main.py`**

Replace the whole file with (the CLI surface is unchanged except a new `status` verb; every dispatch now runs inside a monitor):

```python
"""CLI orchestrator for the ComplexConstraints v1 eval pipeline.

Usage:
    uv run python main.py <step> [--limit N] [--mode sample|score]
                                 [--experiment SLUG] [--description STR]
                                 [--run-report PATH]
    uv run python main.py status          # print the live progress heartbeat

Steps: load · generate · grade · classify · validate · aggregate · all
       · connectivity · status

`all` runs connectivity → load → generate → grade → classify →
validate(sample) → aggregate, all under one live progress display.
`--limit N` is honored by every step; `all` REQUIRES `--limit`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from config import CANDIDATES, PROGRESS_PATH
from pipeline import (
    aggregate, classify, connectivity, generate, grade, load, validate,
)
from pipeline.monitor import build_monitor, render_lines

STEPS = ("load", "generate", "grade", "classify", "validate", "aggregate")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description=__doc__)
    p.add_argument("step", choices=(*STEPS, "all", "connectivity", "status"),
                   help="Pipeline step to run.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N prompts. Required for `all`.")
    p.add_argument("--mode", choices=("sample", "score"), default="sample",
                   help="For `validate`: sample (default) or score.")
    p.add_argument("--experiment", default=None,
                   help="Experiment slug E<NN>-<label> (tags the run + heartbeat).")
    p.add_argument("--description", default=None,
                   help="One-liner describing what makes this experiment distinct.")
    p.add_argument("--run-report", default=None, dest="run_report",
                   help="Path to the meta/ run report for this experiment.")
    return p


def _run_all(limit, experiment, description, run_report) -> None:
    with build_monitor("all", limit, experiment) as mon:
        connectivity.run(monitor=mon)
        load.run(limit=limit, monitor=mon)
        generate.run(limit=limit, monitor=mon)
        grade.run(limit=limit, monitor=mon)
        classify.run(limit=limit, monitor=mon)
        validate.run(mode="sample", monitor=mon)
        aggregate.run(limit=limit, experiment=experiment,
                      description=description, run_report=run_report, monitor=mon)


def _print_status() -> None:
    if not PROGRESS_PATH.exists():
        print("no run found (outputs/progress.json missing).")
        return
    snap = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    print(render_lines(snap))
    if snap.get("state") == "running":
        try:
            updated = datetime.fromisoformat(snap["updated_at"])
            age = (datetime.now(timezone.utc) - updated).total_seconds()
        except (KeyError, ValueError):
            age = None
        if age is not None and age > 90:
            print(f"\n⚠ possibly stalled — last update {int(age)}s ago (pid {snap.get('pid')}).")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.step == "status":
        _print_status()
        return 0

    if args.step == "connectivity":
        with build_monitor("connectivity", 0, args.experiment) as mon:
            connectivity.run(monitor=mon)
        return 0

    if args.step == "all":
        if args.limit is None:
            print("error: `all` requires --limit (e.g. `--limit 3` for a smoke test, "
                  "`--limit 75` for the full set).", file=sys.stderr)
            return 2
        _run_all(args.limit, args.experiment, args.description, args.run_report)
        return 0

    with build_monitor(args.step, args.limit or 0, args.experiment) as mon:
        if args.step == "load":
            load.run(limit=args.limit, monitor=mon)
        elif args.step == "generate":
            generate.run(limit=args.limit, monitor=mon)
        elif args.step == "grade":
            grade.run(limit=args.limit, monitor=mon)
        elif args.step == "classify":
            classify.run(limit=args.limit, monitor=mon)
        elif args.step == "validate":
            validate.run(mode=args.mode, monitor=mon)
        elif args.step == "aggregate":
            aggregate.run(limit=args.limit, experiment=args.experiment,
                          description=args.description, run_report=args.run_report,
                          monitor=mon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_main_wiring.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main_wiring.py
git commit -m "feat: drive one RunMonitor per invocation in main.py

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task L: `status` command rendering + staleness (integration)

**Files:**
- Create: `tests/test_status_command.py`

**Interfaces:**
- Consumes: `main._print_status`, `main.PROGRESS_PATH` (Task K).
- Produces: a passing end-to-end check of the status reader against a written heartbeat, including the stale case. (No production code beyond Task K — this task hardens the `status` path.)

- [ ] **Step 1: Write the failing test**

Create `tests/test_status_command.py`:

```python
import json
from datetime import datetime, timedelta, timezone

import main


def _snap(state, updated):
    return {
        "experiment": "E02-v1-75p", "state": state, "pid": 4242,
        "started_at": "2026-07-02T14:30:00+00:00", "updated_at": updated,
        "elapsed_s": 4340, "eta_s": 10800,
        "current": {"stage": "generate", "model": "qwen-397b", "prompt_id": "prompt-50"},
        "overall": {"done": 227, "total": 606, "pct": 37.5},
        "stages": [{"name": "generate", "state": "running",
                    "done": 148, "total": 225, "pct": 65.8}],
        "retries": 3, "errors": 0,
    }


def test_status_renders_heartbeat(tmp_path, monkeypatch, capsys):
    path = tmp_path / "progress.json"
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(_snap("running", now)))
    monkeypatch.setattr(main, "PROGRESS_PATH", path)
    main._print_status()
    out = capsys.readouterr().out
    assert "E02-v1-75p" in out and "148/225" in out
    assert "stalled" not in out                       # fresh heartbeat


def test_status_flags_stale_running_run(tmp_path, monkeypatch, capsys):
    path = tmp_path / "progress.json"
    old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    path.write_text(json.dumps(_snap("running", old)))
    monkeypatch.setattr(main, "PROGRESS_PATH", path)
    main._print_status()
    assert "possibly stalled" in capsys.readouterr().out


def test_status_missing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main, "PROGRESS_PATH", tmp_path / "nope.json")
    main._print_status()
    assert "no run found" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_status_command.py -v`
Expected: If Task K is complete, these may PASS immediately. If any fail (e.g. stale threshold), fix `main._print_status` accordingly. If they pass first try, that confirms the integration — proceed.

- [ ] **Step 3: Full suite + a real smoke render**

Run: `uv run pytest -q`
Expected: PASS (all green).

Then a manual sanity check of the live display end-to-end (uses real API keys + network; skip if keys absent):

Run: `uv run python main.py all --limit 1 --experiment E00-monitor-smoke --description "monitoring smoke"`
Expected: live bars advance through connectivity → load → generate → grade → classify → validate → aggregate; afterward `outputs/progress.json` exists and `uv run python main.py status` prints the final `done` snapshot; a timestamped file exists under `outputs/logs/`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_status_command.py
git commit -m "test: cover status command rendering + stale detection

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task(s) |
| --- | --- |
| §3 RunMonitor + sinks architecture | B, C, D, E |
| §4.1 monitor.py (Stage/WorkPlan/RunMonitor/ACTIVE) | B, C |
| §4.2 ConsoleSink / LogSink / StatusSink | D, E |
| §4.3 main.py WorkPlan + `status` verb | K, L |
| §4.4 step instrumentation + already_done seeding | G, H, I, J |
| §4.5 config paths | A |
| §4.6 rich dependency | A |
| §5 WorkPlan totals (606) | B |
| §6 progress.json schema | C (snapshot), D (StatusSink) |
| §7 ETA rolling window + throttling | B (Stage.eta), C (RunMonitor.eta), D (StatusSink throttle) |
| §8 console layout | C (render_lines), E (ConsoleSink) |
| §9 generate continue-on-error | G |
| §10 testing via recording fake sink | C (RecordingSink) + every step task |
| §11 files touched | all tasks; `_io.iter_progress` removed in F |
| §12 timestamped log filenames, gitignore | E (default_sinks), A (paths under outputs/) |

No spec requirement is unassigned.

**2. Placeholder scan** — every code step contains complete, runnable code. The one prose-described change (Task J Step 4, mechanical `print → mon.note` over aggregate's moved body) names each of the 8 print sites explicitly rather than leaving it vague; the transformation is uniform and shown by example.

**3. Type consistency** — `RunMonitor` method names (`start_stage`, `item_start`, `item_done`, `end_stage`, `record_error`, `record_retry`, `note`, `snapshot`), the `Sink` triple (`update`/`log`/`close`), module helpers (`note_retry`/`note_error`/`ACTIVE`), `render_lines`, `build_monitor`, `default_sinks`, and `stage_ctx` (defined in Task E's monitor.py, imported by G/H/I/J) are used identically across tasks. Step signatures converge on `run(..., monitor: RunMonitor | None = None)` everywhere; `load.run` additionally keeps its `-> int` return.

## Notes for the implementer

- Run `uv run pytest -q` after each task; the suite must stay green.
- Steps import `stage_ctx` from `pipeline.monitor` (Task E) — it needs `build_monitor`, so Task E precedes the step tasks.
- The live display and raw `print()` cannot coexist — that's why every step's per-item `print` is replaced by monitor events and every summary `print` by `mon.note`. If you spot a stray `print` in an instrumented module, route it through `mon.note`.
