"""Compact GRPO LR-sweep report (STOP report). For each LR: reward mean/std
trajectory (windowed), format_ok, KL, response-length drift (mean_length first vs
last window — a PREREG §8 artifact tell), cap-hit, tok/s, steps. Picks the LR by
reward SLOPE + stability, not final reward."""
import csv, glob, json, os, re

PROBE = "/workspace/failure-mode-id/results/probe"


def col(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v not in ("", None):
            try: out.append(float(v))
            except ValueError: pass
    return out


def window(vals, k):
    if not vals: return None
    return sum(vals[:k]) / min(k, len(vals)), sum(vals[-k:]) / min(k, len(vals))


def slope(vals):
    n = len(vals)
    if n < 2: return 0.0
    xs = list(range(n)); mx = sum(xs)/n; my = sum(vals)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, vals)); den = sum((x-mx)**2 for x in xs)
    return num/den if den else 0.0


print(f"{'LR':>8} {'steps':>5} {'R@start':>8} {'R@end':>7} {'slope/step':>11} "
      f"{'R_std@end':>9} {'fmt_ok':>7} {'KL@end':>8} {'len0':>6} {'lenN':>6} {'cap%':>5} {'tok/s':>7}")
picks = []
for f in sorted(glob.glob(f"{PROBE}/grpo_RA_coverage_lr*.csv")):
    m = re.search(r"_lr([0-9.e-]+)\.csv", f)
    lr = m.group(1) if m else os.path.basename(f)
    rows = list(csv.DictReader(open(f)))
    R = col(rows, "reward"); Rs = col(rows, "reward_std")
    fmt = col(rows, "format_ok"); kl = col(rows, "kl")
    ln = col(rows, "completions/mean_length"); cap = col(rows, "completions/clipped_ratio")
    r0, rN = (window(R, 5) or (0, 0))
    sm = f.replace(".csv", "_summary.json").replace("grpo_RA_coverage_lr", "grpo_RA_coverage_lr")
    sm = f"{PROBE}/grpo_RA_coverage_lr{lr}_summary.json"
    toks = json.load(open(sm))["throughput"]["tok_per_s"] if os.path.exists(sm) else float("nan")
    sl = slope(R)
    print(f"{lr:>8} {len(rows):>5} {r0:>8.3f} {rN:>7.3f} {sl:>+11.5f} "
          f"{(Rs[-1] if Rs else 0):>9.3f} {(fmt[-1] if fmt else 0):>7.3f} "
          f"{(kl[-1] if kl else 0):>8.4f} {(ln[0] if ln else 0):>6.0f} {(ln[-1] if ln else 0):>6.0f} "
          f"{(100*cap[-1] if cap else 0):>5.1f} {toks:>7.1f}")
    picks.append((lr, sl, (Rs[-1] if Rs else 9), rN, kl[-1] if kl else 0))

if picks:
    # healthiest: positive slope, low end-std (stable), bounded KL
    ranked = sorted(picks, key=lambda p: (-(p[1] > 0), -p[1], p[2]))
    print(f"\nRecommended LR (max reward slope with stability): {ranked[0][0]}")
    print("ranking (lr, slope, end_std, R@end, KL@end):")
    for p in ranked: print("  ", p)
