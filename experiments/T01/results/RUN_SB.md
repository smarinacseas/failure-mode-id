# Arm SB: SFT · precision (run record)

Real-arm training run, T1.3. Frozen config: `config/t1_3_frozen.md` (SFT block).
Adapter weights + raw logs live on the `/workspace` volume (`results/adapters/T01-SB/`,
`results/logs/sft_SB_precision.*`), gitignored by design (`/results/`); this tracked
record is the committed provenance.

- **Date:** 2026-07-18
- **Env:** identical to SA (torch 2.8.0+cu128 · trl 1.8.0 · transformers 5.14.1 · peft 0.19.1).
- **Data:** `data/sft/precision.jsonl`, parity manifest `data/sft_manifests/SB.json`
  → **target_n = 123 / 270** accepted (down-sampled to cross-cause parity; seed 20260715).
- **Config (as run):** identical to SA: LR 1e-4 cosine, warmup_ratio 0.03; effective batch 16
  (4 × 4); max_length 4096, packing off; completion_only_loss True; bf16 + gradient checkpointing;
  2 epochs; seed 20260715. Only the data differs (within-row data-fairness preserved).
- **Token lengths:** n=123, min 165 / med 344 / p95 475 / max 621; **truncation @4096 = 0.00%**.

## Result: GT3 SFT pass bar: **PASS** (training loss clearly decreasing across 2 epochs)

- Steps: 16 (8/epoch × 2). Runtime 29.8 s; 8.26 samples/s.
- **Loss:** mean(first 3) **2.5162** → mean(last 3) **1.9933** (Δ **−0.5229**), decreasing.
  Per-step: 2.543 2.457 2.548 2.372 2.311 2.136 2.121 2.219 2.002 2.106 2.004 2.017 1.927 1.952
  1.986 2.042. `train_loss` (epoch-avg) = 2.1714.
- **Mean token accuracy:** 0.496 → 0.530 (rising).
- **Artifacts (volume):** `results/adapters/T01-SB/adapter_model.safetensors` (97 MB) + config;
  live metrics `results/logs/sft_SB_precision.{csv,jsonl}`.

No kill-switch / hardcap in play. No config deviation.

**Note on absolute loss vs SA.** SB's loss band (~2.5→2.0) sits above SA's (~1.9→1.4); this is a
cross-*cause* datagen difference (precision completions are shorter, max 621 vs 1096 tok, and the
tasks differ), not a method or fairness issue. GT3 is a per-arm *learning* bar (slope), not a
cross-arm level comparison; both arms show a clear negative slope of comparable magnitude (≈−0.52).
