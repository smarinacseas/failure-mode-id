# Experiment Progress Monitoring — Design

- **Date:** 2026-07-01
- **Status:** Approved (ready for implementation planning)
- **Author:** Stefan Marinac (with Claude)

## 1. Problem

A full eval run (`main.py all --limit 75`) is ~525 sequential LLM calls over
4–6 hours. Today all progress reporting is ephemeral `print()` to stdout:

- Per-step headers (`generate · qwen-9b: 12 todo / 75 total`)
- Per-item ticks (`gen prompt-01 · qwen-9b ✓`)
- Retry lines from `_io.retry`

There is **no percentage, no ETA, no current-stage view, no persisted record**,
and nothing survives closing the terminal. For a multi-hour job the developer
cannot answer "how far along is it, what is it doing right now, and when will it
finish?" without scrolling raw stdout.

## 2. Goals / Non-goals

**Goals**
- A **live terminal display** (primary surface): per-stage progress bars, the
  current model + prompt, per-stage and overall %, ETA, retry/error counts, and
  elapsed time.
- A **persistent timestamped log file** per run for durable, greppable history.
- A **`progress.json` heartbeat** written continuously, readable from any shell.
- A **`main.py status`** command that renders the heartbeat as a snapshot —
  lets the developer check "any given experiment" from a second terminal.
- Instrumentation that works for both `all` and any single step
  (`main.py grade --limit 75`).
- Resumability is **visible**: a resumed run shows already-completed work as
  done immediately.

**Non-goals (YAGNI)**
- Web-dashboard live view — deferred. The sink architecture makes it a later
  add with no step changes.
- Concurrent/parallel runs — v1 is sequential by design; `progress.json`
  reflects a single active run.
- Cost/token tracking, historical progress time-series — out of scope.

## 3. Chosen approach — event-emitting `RunMonitor` with pluggable sinks

Pipeline steps emit **semantic events** to one `RunMonitor`. The monitor owns
all counting, timing, and ETA math, and fans each event out to **sinks**:

| Sink | Output |
| --- | --- |
| `ConsoleSink` | `rich` live multi-bar display (the primary surface) |
| `LogSink` | timestamped log file via stdlib `logging` |
| `StatusSink` | `progress.json` heartbeat |

Steps are fully decoupled from rendering — they never import `rich` and never
touch a file. **The sinks are the monitoring surfaces**, so the two deferred
surfaces (status command, web view) become new sinks/readers with zero changes
to step code. This is the observer pattern; its payoff is testability (§10).

Rejected alternatives: passing a `rich.Progress` into every step (couples steps
to `rich`, log/status become bolt-ons, hard to test); a bare callback hook (a
strictly lighter version of the monitor object).

## 4. Components

### 4.1 `pipeline/monitor.py` (new)

- **`Stage`** — `name`, `total`, `done`, `state` (`pending|running|done`),
  rolling window of recent item durations, `current` (`{model, prompt_id}`).
- **`WorkPlan`** — ordered list of `Stage`s for the invocation.
- **`RunMonitor`** — context manager holding the plan, start time, retry/error
  counts, and the sink list. API surface:
  - `start_stage(name, total, already_done=0)` — seed a stage; `already_done`
    pre-counts resumed work so the bar starts partway.
  - `item_start(model=None, prompt_id=None)` / `item_done()`
  - `note_retry(label)` / `note_error(context)`
  - `note(message)` — free-text notice (skip summaries, warnings) → log + a
    console line above the live area.
  - `finish(state)` — flush sinks, render final summary.
- **Module-level `ACTIVE: RunMonitor | None`** plus `note_retry()/note_error()`
  helpers. `RunMonitor.__enter__` sets `ACTIVE = self`; `__exit__` clears it.
  This lets the deep `_io.retry` helper report without threading the monitor
  through every call signature (lazy import in `_io` avoids an import cycle).

### 4.2 Sinks

- **`ConsoleSink`** — a `rich.live.Live` wrapping a `Group(Progress, summary)`.
  `Progress` holds one task per stage; the active task shows
  `current model · prompt · ETA` in a custom column. The summary line renders
  `retries · errors · elapsed`. Refresh ≤ 10 Hz. Non-item notices print above
  the live region (rich supports this). When stdout is not a TTY, degrade to
  periodic plain-text lines (no ANSI).
- **`LogSink`** — stdlib `logging` `FileHandler`, formatter
  `%(asctime)s %(levelname)s %(message)s`. Logs stage start/end, each item
  completion with duration, retries (WARNING), errors (ERROR), and the final
  summary.
- **`StatusSink`** — writes `progress.json` (§6) on every `item_done` and on
  stage/run transitions, throttled to at most ~1 write/second. Atomic write
  (temp file + `os.replace`) so a concurrent `status` read never sees a partial
  file.

### 4.3 `main.py`

- Build the `WorkPlan` from `--limit` × `CANDIDATES` (§5).
- Wrap the invocation in `with RunMonitor(plan) as monitor:` and pass `monitor`
  into each step's `run(...)`. Both `all` and single-step paths use it.
- New verb **`status`** — read `outputs/progress.json`, render a static
  snapshot of the same layout plus `state` and `updated_at`. If `updated_at` is
  older than a threshold (e.g. 90 s) while `state == "running"`, flag
  `possibly stalled` (the process may have died). Missing file → "no run found".

### 4.4 Step instrumentation

Each `pipeline/*.py` `run()` gains `monitor: RunMonitor | None = None`. If
`None`, it constructs a default single-stage monitor so standalone runs still
show a bar. Per-item `print()` calls are replaced by
`monitor.item_start(...)` / `item_done()`. At stage start, seed
`already_done` = count of ids already in the output JSONL so resumed runs render
correctly. Existing non-item prints (aggregate's skip summary, connectivity
results) route through `monitor.note(...)`.

### 4.5 `config.py`

Add paths: `LOGS_DIR = OUTPUTS_DIR / "logs"`, `PROGRESS_PATH = OUTPUTS_DIR /
"progress.json"`.

### 4.6 `pyproject.toml`

Add `rich>=13` (Python 3.14-compatible).

## 5. WorkPlan construction

For limit `L` and `M = len(CANDIDATES)` (3):

| Stage | total | per-item cost |
| --- | --- | --- |
| `connectivity` | `M + 1` | tiny ping |
| `load` | `L` | instant (no network) |
| `generate` | `L × M` | ~30–60 s |
| `grade` | `L × M` | ~20–40 s |
| `classify` | `L` | ~10–20 s |
| `validate` | `1` | instant |
| `aggregate` | `1` | instant |

A single-step invocation yields a one-stage plan. Fast stages (`load`,
`validate`, `aggregate`) simply flash to done.

**Overall %** = items done ÷ total items across all stages (for the full run,
total = 4 + 75 + 225 + 225 + 75 + 1 + 1 = **606**). This is a simple proxy: the
near-instant stages (`connectivity`, `load`) make the bar jump early, and
per-item cost varies. `generate`+`grade` are 450 of 606 items and dominate
wall-clock almost entirely. **ETA** is computed time-weighted (§7) and is the
number to trust for "how long left."

## 6. `progress.json` schema

```jsonc
{
  "experiment": "E02-v1-75p",          // or null for untagged
  "state": "running",                   // idle | running | done | error
  "pid": 12345,
  "started_at": "2026-07-02T14:30:00Z",
  "updated_at": "2026-07-02T15:42:20Z",
  "elapsed_s": 4340,
  "eta_s": 10800,                        // null until a rate is observed
  "current": { "stage": "generate", "model": "qwen-397b", "prompt_id": "prompt-50" },
  "overall": { "done": 227, "total": 606, "pct": 37.5 },  // connectivity(4)+load(75)+generate(148)
  "stages": [
    { "name": "generate", "state": "running", "done": 148, "total": 225, "pct": 65.8 }
    // …one entry per stage in the plan; done stages 100%, later stages 0%
  ],
  "retries": 3,
  "errors": 0
}
```

## 7. ETA and throttling

- **ETA:** each stage keeps a rolling mean of its last ~20 item durations.
  Remaining time = Σ over stages of `remaining_items × rate`, where `rate` is
  the stage's observed mean, or — for not-yet-started stages — the current
  stage's observed rate as a fallback. `null` until at least one item completes.
  Labeled approximate; provider latency varies between `generate` and `grade`.
- **Throttling:** console refresh ≤ 10 Hz; `progress.json` at most ~1 write/s.
  Items are seconds apart, so neither is a bottleneck.

## 8. Console layout (approved)

```
E02-v1-75p   running · 37% · elapsed 1h12m · ETA ~3h

connectivity  ✓ done    4/4
load          ✓ done   75/75
generate  ██████████░░░░░░  148/225  65% · qwen-397b · prompt-50
grade     ░░░░░░░░░░░░░░░░    0/225   0%
classify  ░░░░░░░░░░░░░░░░    0/75    0%
validate      pending   0/1
aggregate     pending   0/1

retries: 3   errors: 0   elapsed: 1h12m
```

The header carries the **whole-run** overall % and ETA; each active stage row
carries its own bar + the current model · prompt. Completed stages collapse to
`✓ done`; not-yet-started stages show `pending`. (This refines the earlier
sketch: ETA moved to the header as a single whole-run figure; the active row now
shows the current model · prompt instead.) On finish, the live region is
replaced by a one-line summary.

## 9. Error-handling behavior (confirmed change)

Unify all batch steps on **continue-on-error**:

- **`generate`** (changed): on exhausted retries, log the error, increment
  `errors`, and **skip writing that response** — leaving the id for a later
  resume. Today it raises and kills the whole run; the change means one bad call
  no longer wastes hours, and the "errors" counter is meaningful.
- **`grade`** (unchanged): already records all-FAIL with a `judge_parse_error`
  reason on persistent failure.
- **`aggregate`** (unchanged): already skips prompts missing any model's
  response/grades — coherent with `generate` skipping a failed id.

Net: a run always completes; failures surface as the `errors` count + log
entries, and a rerun resumes the skipped ids.

## 10. Testing

- **`RunMonitor` against a recording fake sink**: feed event sequences, assert
  per-stage `done`/`total`/`state`, overall %, retry/error counts, ETA math, and
  the emitted `progress.json` shape. No terminal, no `rich`, no snapshot tests.
- **`WorkPlan` construction**: totals from `limit` × `CANDIDATES`, single-step
  plans, `already_done` seeding.
- **`status` command**: render from fixed `progress.json` fixtures, including
  the stale-heartbeat case.
- `ConsoleSink` stays thin (rich rendering not unit-tested); the fully-tested
  state model sits underneath it.
- Add a `tests/` directory (none exists yet).

## 11. Files touched

**New**
- `pipeline/monitor.py`
- `tests/test_monitor.py` (+ `tests/` scaffolding)

**Modified**
- `main.py` — WorkPlan build, `RunMonitor` lifecycle, `status` verb
- `pipeline/generate.py` — instrument + continue-on-error
- `pipeline/grade.py`, `pipeline/classify.py`, `pipeline/validate.py`,
  `pipeline/aggregate.py`, `pipeline/load.py`, `pipeline/connectivity.py` —
  instrument (replace per-item prints; route notices through `monitor.note`)
- `pipeline/_io.py` — `retry()` reports via `monitor.note_retry`; remove the
  unused `iter_progress` helper
- `config.py` — `LOGS_DIR`, `PROGRESS_PATH`
- `pyproject.toml` — add `rich>=13`

## 12. Rollout notes

- `outputs/logs/` and `outputs/progress.json` follow the existing gitignore
  policy for `outputs/` (only `outputs/experiments/` is committed).
- Log filenames are timestamped per run
  (`outputs/logs/<slug-or-untagged>-<YYYYMMDD-HHMMSS>.log`) so reruns never
  clobber history; `progress.json` is a single latest-run heartbeat.
```
