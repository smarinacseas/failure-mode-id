"""VerIH-style RLVR (GRPO) on Tinker — runbook script, not a library.

Trains `BASE_MODEL` with the native VerIH reward (training/verih_reward.py,
faithful to data/verih/RLVR/verl/utils/reward_score/ih.py's compute_score)
over the real VerIH train split (data/verih/RLVR/dataset/verih/train.json,
loaded via training/data.py::load_verih). GRPO mechanics (Datum construction,
forward_backward/optim_step call shapes) are copied from the tinker_cookbook
recipe at .venv/lib/python3.14/site-packages/tinker_cookbook/recipes/rl_loop.py
(tinker==0.22.7, tinker_cookbook==0.4.3 — same versions proxy.py cites), with
`get_reward(response, answer)` swapped for `rl_reward(text, gt)` and the
GSM8K-specific dataset/convo-prefix machinery dropped.

Every deviation from VerIH's own train_grpo.sh hyperparameters, and every
place this script's Tinker call shapes had to diverge from what rl_loop.py
does (SDK behavior discovered by reconciling against the installed package,
not assumed), is documented inline below and summarized in the module-level
CONFIG block.

Run:
    caffeinate -is uv run python -m training.train_rlvr

All hyperparameters are env-var overridable (see CONFIG below) so a smoke
run can shrink STEPS/PROMPTS_PER_STEP/MAX_TOKENS without editing code.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import renderers

from training.data import load_verih
from training.verih_reward import has_think_block, rl_reward

# --------------------------------------------------------------------------
# CONFIG (env-var overridable; every default cites its VerIH source and any
# deliberate deviation from it)
# --------------------------------------------------------------------------

# VerIH's own checkpoint — confirmed live against Tinker's get_server_capabilities()
# this session (not assumed from VerIH's paper/repo).
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")

# DEVIATION: VerIH's train_grpo.sh does full fine-tuning (no LoRA). Full-FT
# isn't how Tinker training clients work for a pilot of this size/budget —
# create_lora_training_client is the primitive available, so we LoRA.
LORA_RANK = int(os.environ.get("LORA_RANK", "32"))

# DEVIATION (pilot wall-clock): VerIH trains 12 epochs at batch_size=128
# prompts/step over ~7,192 rows (~56 steps/epoch * 12 ~= 674 steps total).
# 225 steps * 32 prompts/step ~= 7,200 draws ~= one epoch over the (smaller,
# post-filter) pool at a much smaller per-step batch, sized for pilot
# wall-clock rather than matching VerIH's full schedule.
STEPS = int(os.environ.get("STEPS", "225"))
PROMPTS_PER_STEP = int(os.environ.get("PROMPTS_PER_STEP", "32"))

# FAITHFUL: VerIH's rollout.n (GRPO group size).
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "4"))

# FAITHFUL: VerIH's max_response_length.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))

# FAITHFUL: VerIH's max_prompt_length / filter_overlong_prompts cutoff.
MAX_PROMPT_TOKENS = int(os.environ.get("MAX_PROMPT_TOKENS", "1024"))

# DEVIATION: VerIH used lr=1e-6 for full-FT. LoRA adapters need a much
# higher lr to move the (much smaller) trainable parameter count; 1e-5 is
# the tinker_cookbook LoRA convention (rl_loop.py's own Config defaults to
# 4e-5 for a from-scratch base model — we're more conservative here since
# we start from an already-instruction-tuned checkpoint).
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))

# FAITHFUL: verl's rollout sampling temperature default.
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))

SEED = int(os.environ.get("SEED", "20260712"))

DATA_PATH = os.environ.get("DATA_PATH", "data/verih/RLVR/dataset/verih/train.json")
CHECKPOINTS_PATH = os.environ.get("CHECKPOINTS_PATH", "training/checkpoints.json")

# DEVIATION: VerIH's GRPO loss adds kl_loss_coef=0.001 against a frozen
# reference model. rl_loop.py — the tinker_cookbook recipe this script's
# Datum/forward_backward/optim_step shapes are reconciled against — has NO
# reference-model KL term: forward_backward(datums, loss_fn="importance_sampling")
# is the entire loss. We replicate the recipe's loss mechanism exactly rather
# than bolt on a KL term the recipe doesn't support, so there is no KL
# penalty in this training loop. (kl_loss_coef is small enough — 0.001 — that
# VerIH's own ablations describe it as a mild regularizer, not core to the
# method; omitting it is a scope cut, not a correctness bug.)

BASE_CHECKPOINT_NAME = "qwen3-8b-base"
FT_CHECKPOINT_NAME = "qwen3-8b-ihrlvr"
PARTIAL_CHECKPOINT_NAME = "qwen3-8b-ihrlvr-partial"


def _gt_of(sample: dict) -> dict | None:
    constraints = sample.get("constraints") or []
    return constraints[0] if constraints else None


def _is_trainable_row(sample: dict) -> bool:
    """Mirrors ih.py's compute_score dispatch: only rows whose `type` string
    contains "aligned" or "conflict" go through the IF_FUNCTIONS_MAP branch
    (ih.py also handles adversarial_*/folio/lsat types we don't train on
    here); and the gt dict must actually carry a func_name to be scoreable."""
    gt = _gt_of(sample)
    if gt is None or "func_name" not in gt:
        return False
    sample_type = sample.get("type") or ""
    return ("aligned" in sample_type) or ("conflict" in sample_type)


def _build_pool(
    samples: list[dict], renderer, max_prompt_tokens: int
) -> tuple[list[dict], int, int]:
    """Filter to trainable rows, render each prompt once (precompute — the
    per-step loop reuses these ModelInputs rather than re-tokenizing), and
    drop prompts over the length cap. Returns (pool, n_skipped_untrainable,
    n_skipped_overlong)."""
    n_skipped_untrainable = 0
    n_skipped_overlong = 0
    pool: list[dict] = []
    for sample in samples:
        if not _is_trainable_row(sample):
            n_skipped_untrainable += 1
            continue
        model_input = renderer.build_generation_prompt(sample["messages"])
        if model_input.length > max_prompt_tokens:
            n_skipped_overlong += 1
            continue
        pool.append({"model_input": model_input, "gt": _gt_of(sample), "type": sample.get("type")})
    return pool, n_skipped_untrainable, n_skipped_overlong


def _batch_for_step(pool: list[dict], step: int, prompts_per_step: int) -> list[dict]:
    """Cycles through `pool` with wraparound so STEPS * PROMPTS_PER_STEP can
    exceed len(pool) without index errors — the pool was shuffled once at
    startup (seeded), so wraparound repeats rows in the same relative order
    rather than reshuffling per lap. Acceptable for a pilot; a longer run
    should reshuffle each epoch boundary."""
    n = len(pool)
    start = (step * prompts_per_step) % n
    return [pool[(start + i) % n] for i in range(prompts_per_step)]


def _save_named_checkpoint(training_client, name: str) -> str:
    """Durable, retrievable-by-path checkpoint — NOT
    save_weights_and_get_sampling_client(name=...). LIVE-VERIFY finding:
    against the installed tinker==0.22.7, TrainingClient.save_weights_and_get_sampling_client's
    `name` kwarg is deprecated and has NO effect ("checkpoints are always
    ephemeral" — confirmed via inspect.signature + the method's own
    DeprecationWarning text this session); its docstring directs callers who
    need a persistent, path-addressable checkpoint to
    save_weights_for_sampler(name=...) + create_sampling_client(model_path=...)
    instead, which is what proxy.py's BASE_CHECKPOINT/FT_CHECKPOINT env vars
    already assume (model_path=os.environ["BASE_CHECKPOINT"]). So: named,
    durable checkpoints (base/ft/partial) go through save_weights_for_sampler;
    the per-step rollout sampling client below intentionally does NOT use a
    name (it's genuinely ephemeral — only used within that one step)."""
    result = training_client.save_weights_for_sampler(name, ttl_seconds=None).result()
    return result.path


def _write_checkpoints(path: str, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()  # TINKER_API_KEY et al. live in .env (repo convention, cf. config.py)
    service = tinker.ServiceClient()  # reads TINKER_API_KEY
    training_client = service.create_lora_training_client(base_model=BASE_MODEL, rank=LORA_RANK)

    # Tokenizer straight off the training client (TrainingClient.get_tokenizer
    # exists and avoids spinning up a throwaway SamplingClient just to read
    # it — proxy.py reads it off a SamplingClient because that's the only
    # client it holds; here we already have the training client). Renderer
    # choice ("qwen3") matches proxy.py's live-confirmed thinking renderer
    # for Qwen3-8B, and model_info.get_recommended_renderer_name(BASE_MODEL)
    # returns "qwen3" too, confirmed offline this session. Only the thinking
    # renderer is needed here — VerIH's reward requires a <think> block, so
    # training always renders with thinking on; proxy.py's non-thinking
    # renderer pairing only matters for eval-time serving (Task 10).
    tokenizer = training_client.get_tokenizer()
    renderer = renderers.get_renderer("qwen3", tokenizer)

    checkpoints: dict = {}
    base_path = _save_named_checkpoint(training_client, BASE_CHECKPOINT_NAME)
    checkpoints["base"] = base_path
    _write_checkpoints(CHECKPOINTS_PATH, checkpoints)
    print(f"base checkpoint saved: {base_path}")

    samples = load_verih(DATA_PATH)
    random.Random(SEED).shuffle(samples)
    pool, n_skipped_untrainable, n_skipped_overlong = _build_pool(samples, renderer, MAX_PROMPT_TOKENS)
    print(
        f"pool={len(pool)} rows "
        f"(skipped {n_skipped_untrainable} non-aligned/conflict-or-no-func_name, "
        f"{n_skipped_overlong} over {MAX_PROMPT_TOKENS} prompt tokens; "
        f"{len(samples)} rows total)"
    )
    if not pool:
        raise RuntimeError("no trainable rows survived filtering — check DATA_PATH / gt schema")

    sampling_params = tinker.SamplingParams(
        max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop=renderer.get_stop_sequences()
    )
    adam_params = types.AdamParams(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.95, eps=1e-8)

    printed_debug_sample = False
    last_mean_reward = 0.0

    def _finalize(ft_name: str, partial: bool) -> None:
        ft_path = _save_named_checkpoint(training_client, ft_name)
        checkpoints["ft"] = ft_path
        checkpoints["run"] = {
            "params": {
                "base_model": BASE_MODEL,
                "lora_rank": LORA_RANK,
                "steps": STEPS,
                "prompts_per_step": PROMPTS_PER_STEP,
                "group_size": GROUP_SIZE,
                "max_tokens": MAX_TOKENS,
                "max_prompt_tokens": MAX_PROMPT_TOKENS,
                "learning_rate": LEARNING_RATE,
                "temperature": TEMPERATURE,
                "seed": SEED,
            },
            "last_mean_reward": last_mean_reward,
            "partial": partial,
        }
        _write_checkpoints(CHECKPOINTS_PATH, checkpoints)
        print(f"{'partial ' if partial else ''}ft checkpoint saved: {ft_path}")
        print(f"checkpoints written to {CHECKPOINTS_PATH}: base={checkpoints['base']} ft={ft_path}")

    try:
        for step in range(STEPS):
            t_start = time.time()
            # Fresh, EPHEMERAL sampling client reflecting current weights —
            # see _save_named_checkpoint's docstring for why no `name=` here.
            sampling_client = training_client.save_weights_and_get_sampling_client()

            batch = _batch_for_step(pool, step, PROMPTS_PER_STEP)
            futures_P = [
                sampling_client.sample(
                    prompt=row["model_input"], num_samples=GROUP_SIZE, sampling_params=sampling_params
                )
                for row in batch
            ]

            datums_D: list[types.Datum] = []
            rewards_P: list[float] = []
            n_format_ok = 0
            n_samples_total = 0
            n_degenerate_groups = 0

            for future, row in zip(futures_P, batch):
                prompt = row["model_input"]
                gt = row["gt"]
                sample_result = future.result()

                rewards_G: list[float] = []
                sampled_tokens_G_T: list[list[int]] = []
                logprobs_G_T: list[list[float]] = []
                for sequence in sample_result.sequences:
                    sampled_tokens = sequence.tokens
                    sampled_logprobs = sequence.logprobs
                    assert sampled_logprobs is not None

                    text = tokenizer.decode(sampled_tokens)
                    reward = rl_reward(text, gt)

                    if not printed_debug_sample:
                        print(f"[step 0 sample] reward={reward} text[:300]={text[:300]!r}")
                        printed_debug_sample = True

                    n_samples_total += 1
                    if has_think_block(text):
                        n_format_ok += 1

                    sampled_tokens_G_T.append(sampled_tokens)
                    logprobs_G_T.append(sampled_logprobs)
                    rewards_G.append(reward)

                mean_reward_prompt = sum(rewards_G) / len(rewards_G)
                advantages_G = [r - mean_reward_prompt for r in rewards_G]
                rewards_P.append(mean_reward_prompt)

                # Standard GRPO practice: a group with zero-variance rewards
                # (all pass or all fail) gives every advantage 0 — no
                # gradient signal — so skip building datums for it.
                if all(a == 0.0 for a in advantages_G):
                    n_degenerate_groups += 1
                    continue

                for sampled_tokens, logprobs, advantage in zip(
                    sampled_tokens_G_T, logprobs_G_T, advantages_G
                ):
                    ob_len = prompt.length - 1
                    model_input = prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))
                    target_tokens = [0] * ob_len + sampled_tokens
                    padded_logprobs = [0.0] * ob_len + logprobs
                    padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                    assert (
                        model_input.length
                        == len(target_tokens)
                        == len(padded_logprobs)
                        == len(padded_advantages)
                    ), (
                        f"model_input.length: {model_input.length}, len(target_tokens): {len(target_tokens)}, "
                        f"len(padded_logprobs): {len(padded_logprobs)}, len(padded_advantages): {len(padded_advantages)}"
                    )
                    datum = types.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                        },
                    )
                    datums_D.append(datum)

            if datums_D:
                fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
                optim_step_future = training_client.optim_step(adam_params)
                fwd_bwd_future.result()
                optim_step_future.result()

            mean_reward = sum(rewards_P) / len(rewards_P) if rewards_P else 0.0
            frac_format_ok = n_format_ok / n_samples_total if n_samples_total else 0.0
            frac_degenerate_groups = n_degenerate_groups / len(batch) if batch else 0.0
            elapsed = time.time() - t_start
            last_mean_reward = mean_reward

            print(
                f"step={step:04d} mean_reward={mean_reward:.4f} "
                f"frac_format_ok={frac_format_ok:.4f} "
                f"frac_degenerate_groups={frac_degenerate_groups:.4f} "
                f"elapsed={elapsed:.1f}s"
            )
    except KeyboardInterrupt:
        print("\ninterrupted — saving partial ft checkpoint before exiting (training compute is money)")
        _finalize(PARTIAL_CHECKPOINT_NAME, partial=True)
        raise SystemExit(130)

    _finalize(FT_CHECKPOINT_NAME, partial=False)


if __name__ == "__main__":
    main()
