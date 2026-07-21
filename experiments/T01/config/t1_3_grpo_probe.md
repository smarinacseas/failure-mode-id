# T1.3 GRPO LR probe: results & selection (provenance)

One per-method probe (PREREG amendment 2026-07-16 (b)): coverage/RA, 50 prompts,
2 epochs, **k=6 frozen**, sweep **LR only** over the pre-registered 5e-6 to 1e-5 range.
7.5e-6 added because 5e-6 and 1e-5 both finished far inside the 3-hour cap. Backend:
stock `model.generate()` (`use_vllm=False`), GC-on + cached generation, temp 0.9, β 0.04.
All three ran a full **50 steps** (equal basis: no LR truncated by the cap).

## Sweep table (per-step CSVs in results/probe/grpo_RA_coverage_lr*.csv)

| LR | steps | reward@start | reward@end | slope/step | reward_std@end | format_ok | KL@end | len@0→N | cap-hit | tok/s |
|----|-------|-------------|-----------|-----------|----------------|-----------|--------|---------|---------|-------|
| 5e-6   | 50 | 0.766 | 0.849 | +0.00045 | 0.178 | 1.00 | 0.0007 | 296→341 | 0% | 226 |
| 7.5e-6 | 50 | 0.743 | 0.830 | **+0.00097** | **0.104** | 1.00 | 0.0007 | 296→346 | 0% | 230 |
| 1e-5   | 50 | 0.767 | 0.849 | +0.00064 | 0.130 | 1.00 | 0.0008 | 296→332 | 0% | 226 |

## Selection: **LR = 7.5e-6**

Chosen on **reward slope + stability**, not final reward (reviewer criterion):
7.5e-6 has the steepest reward slope (+0.00097/step) and the lowest end-of-run
reward_std (0.104 = most stable), with KL bounded (~0.0007) and format_ok 1.0.
5e-6 and 1e-5 tie on final reward (0.849) but 5e-6 has the weakest slope and
noisiest end (std 0.178). Frozen for **both** RL arms (RA, RB); no per-arm tuning.

## Health flags (carried to the STOP report)
- **Reward starts high (~0.77) with limited headroom**: coverage base difficulty
  is above the PREREG §5 30 to 70% band (see reward-calibration / base-difficulty check).
- **Mild length drift**: mean completion 296→~340 tok as reward rises ~0.08. Modest
  (answers ≪ M=2800-char cap; format_ok 1.0), but a pre-committed artifact tell
  (PREREG §8) to watch on the full run.
- KL tiny throughout (β=0.04 default healthy: no collapse/blow-up; no β change needed).
- Cap-hit 0% at max_completion_length=1536; no bump needed.

## Throughput
~226 to 230 tok/s (generate(), LoRA-taxed, KV-cache ON). Confirmed the cache path:
base+LoRA+cache = 260 tok/s vs base-no-LoRA = 477 (LoRA ≈1.8× tax); GC+use_cache
matches GC-off speed. At ~19 s/step, a 50-step run ≈ 16 min ≪ 3-hour cap. Full-run
projection: 300 prompts × 2 epochs ≈ 300 steps ≈ ~95 min/arm (length-drift dependent).

---

## Re-probe on the HARDENED coverage pool (2026-07-16): LR pick RE-CONFIRMED 7.5e-6

The pick above was made on the *easy* coverage pool (base reward ~0.77). That pool was
recalibrated into band (coverage-recal amendment; base criterion-pass 86.4%→61.0%), so
the LR probe was re-run on the hardened pool. Same protocol: RA/coverage, 50 prompts,
2 epochs, k=6, LR-only over {5e-6, 7.5e-6, 1e-5}, stock generate(), temp 0.9, β 0.04.
Per-step CSVs: `results/probe/grpo_RA_coverage_lr*_hard.csv`; summaries `*_hard_summary.json`.

**Base reward now ~0.435 at step 0 (was ~0.77): good headroom, confirms the recal.**

### Robust (windowed) table: single-endpoint metrics are noise at 2 prompts/step
Reward is measured over 12 rollouts/step (2 unique prompts × k=6), so per-step reward is
very noisy (bounces 0.28 to 0.71). Ranking is therefore on 10-step windows, NOT single rows.

| LR | mean r[0:10] | mean r[40:50] | Δ(last−first) | slope (5-step smoothed) | end reward_std [40:50] | format_ok | KL@end |
|----|-------------|--------------|---------------|-------------------------|------------------------|-----------|--------|
| 5e-6   | 0.511 | 0.493 | −0.018 | −0.00046 | 0.196 | 1.00 | 0.0006 |
| 7.5e-6 | 0.487 | 0.455 | −0.032 | −0.00020 | 0.202 | 1.00 | 0.0009 |
| 1e-5   | 0.510 | 0.473 | −0.037 | −0.00020 | 0.188 | 1.00 | 0.0012 |

### Selection: **7.5e-6 RE-CONFIRMED (frozen, no revision)**; the probe does not discriminate
Unlike the easy pool (all three clearly improved; 7.5e-6 won both slope and stability), on the
hardened pool **none of the three improve**; all are flat-to-declining and statistically
indistinguishable (per-step reward Δ across LRs ≈ ±0.1 ≈ reward_std). The pre-committed rule
("steepest slope + lowest end-std") presumes a clean positive slope to rank; that premise is not
met, and the raw last-row std that first appeared to favour 7.5e-6 (0.124) is itself noise: the
windowed end-std makes 7.5e-6 the *highest* (0.202). With **no discriminating signal to revise**,
7.5e-6 is kept: mid-range of the pre-registered 5e-6 to 1e-5 window, and the amendment commits to a
single frozen LR for both arms with no per-arm tuning. Switching to 1e-5 on a 0.014 windowed-std
difference would be exactly the researcher-DoF the stopping rule exists to prevent. (Reviewer
decision, 2026-07-16.)

### Health flag carried to the STOP report
- **No reward learning at the 50-step probe scale on hardened coverage** (contrast easy pool:
  +0.08 over 50 steps). Plausibly probe underpowering (50 steps × 2 prompts vs the full run's
  ~300 steps), but it must be watched on the full RA arm. Length did NOT drift up (355→~300 tok),
  so the easy-pool mild-length-drift flag does not reproduce here.

### Pre-committed check for the REAL RA run (committed 2026-07-16, BEFORE the run starts)
This is a monitoring gate on the full RA (coverage GRPO) run, **not** a re-opening of the LR
choice, which is frozen at 7.5e-6. At **step 50** of the real RA run (matching the probe scale),
compute the windowed mean reward exactly as in the probe analysis: `mean(reward[0:10])` vs
`mean(reward[40:50])` from the run's per-step log.
- If the trend is **flat-or-declining** (not a clean positive slope), **PAUSE and report** before
  continuing: this tests whether the probe's non-learning pattern replicates at the real run's
  start (candidate remedies then: more steps, more prompts/step).
- If it shows a **clear positive slope by step 50**, proceed uninterrupted to completion.
