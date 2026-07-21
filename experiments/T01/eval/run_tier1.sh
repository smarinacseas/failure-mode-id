#!/usr/bin/env bash
# T1.4 Tier-1 full run: decode all 6 arms (resumable) -> grade -> analysis.
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root

for arm in 0 P SA SB RA RB; do
  echo "===== ARM $arm ($(date -u +%H:%M:%SZ)) ====="
  python3 experiments/T01/eval/tier1_decode.py --arm "$arm" --batch-size 48 \
    || { echo "ARM $arm FAILED"; exit 1; }
done

echo "===== GRADE ($(date -u +%H:%M:%SZ)) ====="
python3 experiments/T01/eval/grade_tier1.py || exit 1

echo "===== ANALYSIS ($(date -u +%H:%M:%SZ)) ====="
python3 experiments/T01/analysis/tier1_analysis.py || exit 1

echo "===== T1.4 PIPELINE DONE ($(date -u +%H:%M:%SZ)) ====="
