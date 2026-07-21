# Concurrency design

How the pipeline runs work in parallel without changing what any experiment
measures. Implements E04's suggested next steps #1 (parallelize generate) and
#4 (wall-clock deadline); see `meta/2026-07-03-reasoning-smoke.md`.

Concurrency is transport, not treatment: model params, provider routing
preference, prompt selection (`select_prompts`), judge params/blindness, and
continue-on-error semantics are byte-identical to the sequential path. Every
knob below is recorded in the run manifest (`meta.config`).

## Generation wall-clock deadline

**Value: `GENERATION_DEADLINE_S = 2030 s` (~33.8 min), floor 900 s (15 min).**

Why it exists: `timeout_s` is an httpx read timeout. It resets on every
streamed chunk, so it bounds dead sockets only, not slow-trickle
generations, and not the half-open socket a mid-run client sleep leaves
behind (no bytes, no EOF, no exception). E04 lost 10.3 hours to one such
call. Generation calls are therefore streamed (`stream=True`; the only
request-shape change) with two guards in `pipeline/generate.py`:

1. **per-chunk elapsed check**: wall clock recorded at call start; every
   received chunk compares elapsed against the deadline (catches trickle);
2. **watchdog timer**: closes the stream at the deadline, which makes a
   read blocked on a silent socket raise (catches the no-chunks hang the
   per-chunk check can never see).

Either abort raises `GenerationDeadlineExceeded` into the existing
`retry()` path: a deadline abort is a retry event, and exhausted retries
become one per-item error, identical to any other transient failure.

**Basis (derived 2026-07-06):** deadline = max(2 × p99, 900 s). p99
(nearest-rank, ceil(0.99·n)) over the n=69 observed generation-call
durations under the frozen reasoning-on treatment (48k budget, temp 0.6,
`provider_sort=throughput`):

- `outputs/logs/E04-reasoning-smoke-3p-20260702-235443.log`: 9 calls
- `outputs/logs/E05-reasoning-rand20p-20260706-115918.log`: 60 calls
  (generate stage complete as of the 2026-07-06 16:24 snapshot)

Durations are per-item deltas between consecutive `generate <model> <id>
done` log lines (items ran strictly sequentially), retries included: an
upper bound on any single call, i.e. conservative in the safe direction.
p99 = 1014.6 s (`qwen-9b/CIF-024`, 16.9 min); 2 × 1014.6 = 2029.3 → 2030 s.
Excluded: abandoned-config E04 attempts (greedy+thinking hangs, 32k
cap-outs, max 45.6 min; no current freeze can reproduce them; all would
have been caught by this guard) and reasoning-off E01 to E03 (8k budgets, a
much faster regime that would only dilute the tail). Re-derive against new
logs whenever the generation treatment changes materially; the 15-minute
floor keeps the guard sane if a future config's p99 collapses.

## Lock design

- **`append_jsonl` (pipeline/_io.py):** one `threading.Lock` per output
  file, held in a module-level dict keyed by resolved path (dict access
  guarded by its own lock). The record is serialized *before* the lock;
  under the lock, one `write()` of line+newline, then flush. Contract: a
  record is appended whole on success and not at all on failure, never
  half-written, never interleaved. No per-worker shard files, no writer
  thread; `done_ids`/resume logic untouched.
- **`RunMonitor` (pipeline/monitor.py):** a single re-entrant lock around
  all state mutation *and* every sink write (progress.json included), since
  `note_retry()`/`note_error()`/`item_*()` arrive from worker threads. The
  old single-item `current.model`/`prompt_id` heartbeat fields are replaced
  by an in-flight map; the heartbeat reports in-flight count (+ items),
  completed counts, retry total, error total, and (during batch grading)
  submitted/pending/collected/errored. `main.py status` renders both the
  new and the pre-concurrency heartbeat shapes.

## Generate stage threading

`(candidate × prompt)` work items run on a `ThreadPoolExecutor` with
**`GENERATION_WORKERS = 4`** (config.py; recorded in the manifest). Items
are filtered by `done_ids` before submission, exactly as the sequential
path did; each worker wraps its call in `retry()` and continues on error.
Abrupt termination cancels queued items and re-raises; completed appends
are already on disk, so a re-run resumes exactly the missing set.

## Batch judge (`judge_mode`)

Frozen per-experiment param `judge_mode` ∈ {`batch`, `sequential`},
default `batch` (pre-concurrency freezes load as `batch`; `sequential`
keeps the old one-streamed-call-per-cell path fully functional).

Batch mode restructures `grade.py` into submit → poll → collect against the
Anthropic Message Batches API, one batch per judge:

- **`custom_id` = the item's stable id**: `<candidate>__<prompt-id>`. A
  judge's batch spans every candidate, so the prompt id alone would collide;
  the grading cell's identity is candidate + prompt. Deterministic, derived
  only from the item, never random. custom_ids are request metadata and
  never enter the judge's context (blindness preserved).
- **Resumability**: `done_ids` are re-derived from the output JSONL exactly
  as before, and a (re)run submits a batch containing only the missing ids.
  No batch id is persisted: a run killed mid-poll simply resubmits the
  still-missing set on resume.
- Polls every 60 s with logged status; on completion each result is written
  through the same locked `append_jsonl` path (aggregate/classify unchanged).
- Per-item errors in batch results map to the existing continue-on-error
  accounting: one resubmission (mirroring the sequential path's two grade
  attempts), then an all-FAIL record carrying the reason plus a recorded
  error. Never a silent drop.

## Ordering

Output JSONL is no longer in prompt order. All consumers key by `id` and
depend on set membership only: `aggregate` builds id-keyed dicts,
`classify`/`grade`/`generate` resume via id sets, and results.json prompt
order comes from `select_prompts` over the canonical dataset, never from
file order. One consumer was order-dependent and is fixed:
`validate._build_pool` now sorts the human-scoring pool on cell identity
before its fixed-seed `rng.sample`, so the sampled subset is a pure function
of content, not of grade-file arrival order. `select_prompts` remains the
single source of the prompt subset.

## Live-run rule

Live runs launch **only from the primary working directory at a clean,
recorded commit** (the manifest stamps `git.commit` + `dirty`). Development
happens in isolated worktrees on feature branches, never edit, run, or
write in the primary working directory while an experiment is live, and
never run `main.py` against real APIs from a worktree. Exactly one live
`main.py` process tree may exist (check `ps aux | grep main.py` before
launching; orphans clobber `outputs/progress.json` and burn credits), and
long runs go under `caffeinate -is`.
