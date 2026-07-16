# T1.3 GRPO LR probe — results & selection (provenance)

One per-method probe (PREREG amendment 2026-07-16 (b)): coverage/RA, 50 prompts,
2 epochs, **k=6 frozen**, sweep **LR only** over the pre-registered 5e-6–1e-5 range.
7.5e-6 added because 5e-6 and 1e-5 both finished far inside the 3-hour cap. Backend:
stock `model.generate()` (`use_vllm=False`), GC-on + cached generation, temp 0.9, β 0.04.
All three ran a full **50 steps** (equal basis — no LR truncated by the cap).

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
- **Reward starts high (~0.77) with limited headroom** — coverage base difficulty
  is above the PREREG §5 30–70% band (see reward-calibration / base-difficulty check).
- **Mild length drift**: mean completion 296→~340 tok as reward rises ~0.08. Modest
  (answers ≪ M=2800-char cap; format_ok 1.0), but a pre-committed artifact tell
  (PREREG §8) to watch on the full run.
- KL tiny throughout (β=0.04 default healthy — no collapse/blow-up; no β change needed).
- Cap-hit 0% at max_completion_length=1536; no bump needed.

## Throughput
~226–230 tok/s (generate(), LoRA-taxed, KV-cache ON). Confirmed the cache path:
base+LoRA+cache = 260 tok/s vs base-no-LoRA = 477 (LoRA ≈1.8× tax); GC+use_cache
matches GC-off speed. At ~19 s/step, a 50-step run ≈ 16 min ≪ 3-hour cap. Full-run
projection: 300 prompts × 2 epochs ≈ 300 steps ≈ ~95 min/arm (length-drift dependent).
