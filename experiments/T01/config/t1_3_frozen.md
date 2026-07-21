# T1.3 frozen training config (provenance)

Frozen after the one-per-method probe (PREREG amendment 2026-07-16 (c)). These are
**execution parameters**: the interaction estimand (§§3-7) is immune to them.
Environment: A100-SXM4-80GB · torch 2.8.0+cu128 · trl 1.8.0 · transformers 5.14.1
· peft 0.19.1 (see `/requirements-t01.txt`). Subject weights:
`/workspace/models/llama-3.2-3b-instruct` (local; the bare HF id would re-download).

## Shared LoRA (identical all four arms, amendment (c))
- r=16, α=32, dropout=0.05, bias=none, task_type=CAUSAL_LM
- target_modules = q,k,v,o_proj + gate,up,down_proj (all attention+MLP linears)

## SFT (SA, SB): `sft.py`
- data: `data/sft/{coverage,precision}.jsonl` (conversational prompt-completion)
- completion_only_loss=True (user prompt masked to -100; verified on collated batches)
- max_length=4096 · packing=False   (probe truncation @4096 = 0.00% both causes → NOT bumped to 8192)
- LR=1e-4 · cosine · warmup_ratio=0.03   (probe: peak 1e-4 reached at step 2, then decays, verified)
- effective batch 16 = per_device 4 × grad_accum 4
- bf16=True · gradient_checkpointing=True (use_reentrant=False)
- save_steps=50 · save_strategy=steps · save_total_limit=3 · seed=20260715
- output_dir=`results/adapters/T01-{SA,SB}/`

## GRPO (RA, RB): `grpo.py`   [LR=7.5e-6 selected by probe: see t1_3_grpo_probe.md]
- data: `data/train/{coverage,precision}.jsonl` (+ per-prompt `specs` JSON column)
- **LR = 7.5e-6** (probe pick: steepest reward slope +0.00097/step AND lowest end-std 0.104;
  within pre-registered 5e-6 to 1e-5 range; frozen for BOTH RL arms, no per-arm tuning).
  **RE-CONFIRMED on the hardened coverage pool** (2026-07-16): the re-probe does not
  discriminate (all 3 LRs flat/declining, within noise), so no revision; 7.5e-6 kept.
  See `t1_3_grpo_probe.md` § "Re-probe on the HARDENED coverage pool".
- **Pre-committed RA step-50 check** (committed before the real run): at step 50 of the real
  RA run compute `mean(reward[0:10])` vs `mean(reward[40:50])`; if flat-or-declining, PAUSE
  and report; if clear positive slope, proceed. Monitoring gate only: LR stays 7.5e-6.
- backend: `GRPOConfig(use_vllm=False)`; stock `model.generate()` (amendment (d), no vLLM)
- num_generations k=6 FROZEN (amendment (b), NOT swept) · rollout temp=0.9 · β=0.04 (TRL default)
- max_completion_length=1536 (probe cap-hit = 0%) · reward length cap M=2800 chars
- attn_implementation=sdpa
- gradient_checkpointing=**True** (use_reentrant=False) + `generation_kwargs={"use_cache": True}`:
  GC's `gradient_checkpointing_enable()` sets config.use_cache=False (would give no-KV-cache
  rollouts); generation_kwargs re-enables the cache for `generate()` only, so rollouts stay
  cached (263 tok/s w/ LoRA, same as GC-off) while the training forward runs checkpointed.
  GC-off OOM'd at ~75 GB on lr=1e-5 (length drift), so checkpointing is required for headroom.
  Env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (anti-fragmentation).
- per_device 6 × grad_accum 2 → generation_batch_size 12 = 2 unique prompts/step
- lr_scheduler=constant_with_warmup · warmup_ratio=0.03
- reward: `reward_adapter.make_constraint_reward(max_chars=2800)` grading `extract_final(c)`
  via `verifiers/reward.py::constraint_reward` (malformed penalty 1.0, length penalty 0.5)
- save_steps=50 · save_total_limit=3 · seed=20260715 · output_dir=`results/adapters/T01-{RA,RB}/`

### Deviation logged (PREREG §9, no silent drift)
- TRL 1.8.0 `GRPOConfig` has **no `max_prompt_length`** field (amendment (c) named 512).
  Inert: measured prompt lengths ≤166 tokens (coverage) / ≤127 (precision) ≪ 512, so
  nothing is truncated. Documented rather than silently dropped.
