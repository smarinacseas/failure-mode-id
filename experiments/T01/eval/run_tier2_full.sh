#!/usr/bin/env bash
# Tier-2 full pass: per arm decode (system python / torch) -> grade (.venv / opus),
# then recovery analysis. Both stages resumable, so a disconnect can resume in place.
# Arms: local 0 (baseline reference) + SA SB RA RB. opus-4.8 single judge, k=3.
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root

for arm in 0 SA SB RA RB; do
  echo "===== DECODE $arm ($(date -u +%H:%M:%SZ)) ====="
  python3 experiments/T01/eval/tier2_decode.py --arm "$arm" --k 3 --batch-size 12 \
    || { echo "DECODE $arm FAILED"; exit 1; }
  echo "===== GRADE $arm ($(date -u +%H:%M:%SZ)) ====="
  .venv/bin/python experiments/T01/eval/tier2_grade.py --arm "$arm" --workers 12 \
    || { echo "GRADE $arm FAILED"; exit 1; }
done

echo "===== ANALYSIS ($(date -u +%H:%M:%SZ)) ====="
python3 experiments/T01/eval/tier2_analysis.py || exit 1
echo "===== TIER2 FULL DONE ($(date -u +%H:%M:%SZ)) ====="
