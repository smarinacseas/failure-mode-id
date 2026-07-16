"""Shared training utilities for T1.3 (SFT + GRPO runners).

Frozen knobs live here so both methods use identical LoRA + the estimand-relevant
values are single-sourced (PREREG amendment 2026-07-16 (c)). GPU-touching helpers
(load_model/load_tokenizer) import torch/transformers lazily so this module — and
the pure helpers extract_final / lora_config — import cheaply for unit tests.
"""
from __future__ import annotations

import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
_T01 = _HERE.parents[1]
# flat verifier package (base, reward, coverage_pool, precision_pool)
sys.path.insert(0, str(_T01 / "verifiers"))

# Subject weights live locally on the pod volume; the bare HF id would try to
# re-download 6 GB. Overridable for a different host.
MODEL_PATH = os.environ.get("T01_MODEL_PATH", "/workspace/models/llama-3.2-3b-instruct")
CAUSES = ("coverage", "precision")
FINAL_MARKER = "===FINAL==="

# LoRA target modules — all attention + MLP linears, identical for every arm
# (PREREG amendment 2026-07-16 (c)).
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]


def extract_final(text: str) -> str:
    """The gradeable answer: everything after the last ===FINAL=== marker, or the
    whole text if the marker is absent (a malformed attempt graded as-is).

    Twin of datagen.teacher_gen.extract_final — re-implemented here (2 lines)
    rather than imported, because teacher_gen imports openai/dotenv, which are
    absent in the frozen training env. test_common keeps the two in agreement.
    """
    return text.rsplit(FINAL_MARKER, 1)[-1].strip() if FINAL_MARKER in text else text.strip()


def lora_config():
    """The one LoRA config shared by all four arms (r=16, α=32, attn+MLP)."""
    from peft import LoraConfig
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model():
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        # sdpa is ~2.6x faster than no-cache generation and the modern default;
        # env-overridable if a host lacks the kernel.
        attn_implementation=os.environ.get("T01_ATTN_IMPL", "sdpa"),
    )
