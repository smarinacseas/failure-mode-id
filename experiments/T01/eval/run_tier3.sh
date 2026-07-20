#!/usr/bin/env bash
# T3 Tier-3 regression guard (H3): score all 6 arms on the MMLU subset
# (resumable) -> aggregate -> verdict. Mirrors run_tier1.sh.
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root

echo "===== SMOKE (arm 0, 4 q) ($(date -u +%H:%M:%SZ)) ====="
python3 experiments/T01/eval/tier3_mmlu.py --arm 0 --limit 4 || { echo "SMOKE FAILED"; exit 1; }

for arm in 0 P SA SB RA RB; do
  echo "===== ARM $arm ($(date -u +%H:%M:%SZ)) ====="
  python3 experiments/T01/eval/tier3_mmlu.py --arm "$arm" --batch-size 32 \
    || { echo "ARM $arm FAILED"; exit 1; }
done

echo "===== AGGREGATE / VERDICT ($(date -u +%H:%M:%SZ)) ====="
python3 experiments/T01/eval/tier3_aggregate.py || exit 1

echo "===== T3 PIPELINE DONE ($(date -u +%H:%M:%SZ)) ====="
