#!/usr/bin/env bash
# GRPO probe LR sweep (PREREG amendment 2026-07-16 (b)): ~50 prompts, LR only,
# k=6 FROZEN. Order per operator note: 5e-6, 1e-5 first; 7.5e-6 added because
# each run finishes well inside the 3-hour cap. Stock generate() (clean tok/s).
# All artifacts quarantined under results/probe/ — real adapter dirs untouched.
set -uo pipefail   # NOT -e: one run's failure must not abort the remaining LRs
cd /workspace/failure-mode-id/experiments/T01/training
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM
PROBE=/workspace/failure-mode-id/results/probe
mkdir -p "$PROBE/adapters"

for spec in "5e-6:5e-6" "1e-5:1e-5" "7.5e-6:7.5e-6"; do
  LR="${spec%%:*}"; TAG="${spec##*:}"
  echo "######## GRPO PROBE lr=$LR  $(date -u +%H:%M:%S) ########"
  python grpo.py --arm RA --cause coverage --limit 50 --epochs 2 --lr "$LR" \
    --time-budget-sec 10800 --logging-steps 1 --sanity --run-tag "lr${TAG}" \
    --output-dir "$PROBE/adapters/RA-lr${TAG}" \
    > "$PROBE/grpo_sweep_lr${TAG}.stdout" 2>&1 \
    && echo "done lr=$LR -> summary written" || echo "FAILED lr=$LR (see stdout)"
done
echo "ALL SWEEPS DONE $(date -u +%H:%M:%S)"
