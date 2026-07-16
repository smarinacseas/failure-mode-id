"""T01 Gate GT0 smoke test — validate the training env on the pod.

Exercises the whole stack end-to-end so a broken env fails HERE (cheap) rather
than mid-training (expensive):
  1. CUDA GPU visible
  2. load meta-llama/Llama-3.2-3B-Instruct in bf16 (validates HF auth + gated
     license + weights)
  3. 5 generations (validates inference)
  4. attach LoRA (r=16, a=32, attn+MLP — the frozen adapter config, PREREG §5)
     and take 1 optimizer step (validates peft + backprop)
  5. import trl + vllm (validates the training/rollout stack is installed)

Run on the pod:  uv run python experiments/T01/smoke_test.py
Exit 0 + "GT0 SMOKE: PASS" ⇒ Gate GT0 env check clears.
"""

from __future__ import annotations

import sys

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# LoRA target = attention + MLP projections (Llama), frozen across all 4 arms.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: no CUDA GPU visible", file=sys.stderr)
        return 1
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")

    # 2. load
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model.to("cuda")
    print(f"loaded {MODEL} ({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params)")

    # 3. five generations
    prompts = ["Count to three.", "Name one primary color.", "Say hello.",
               "What is 2 + 2?", "Give a one-word answer: sky color?"]
    for p in prompts:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, return_tensors="pt").to("cuda")
        out = model.generate(ids, max_new_tokens=16, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"  gen[{p!r:34s}] -> {text[:50]!r}")

    # 4. LoRA attach + one optimizer step
    peft_model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGETS,
                   lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    peft_model.print_trainable_parameters()
    peft_model.train()
    batch = tok("Requirements: exactly three items. 1. a 2. b 3. c",
                return_tensors="pt").to("cuda")
    loss = peft_model(**batch, labels=batch["input_ids"]).loss
    loss.backward()
    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=1e-4)
    opt.step()
    opt.zero_grad()
    print(f"  LoRA optimizer step OK (loss={loss.item():.3f})")

    # 5. training stack. trl is the training core (required). vllm is NOT used for
    #    training — GRPO rollouts run on model.generate() (use_vllm=False, PREREG
    #    amendment 2026-07-16), and it can't coexist with this torch/CUDA pin anyway.
    #    So a missing vllm is the EXPECTED state, not a GT0 failure.
    import trl
    print(f"trl {trl.__version__}")
    try:
        import vllm
        print(f"vllm {vllm.__version__} (present, but unused in training)")
    except Exception as e:  # noqa: BLE001
        print(f"  NOTE: vllm absent ({type(e).__name__}) — expected; training uses "
              f"generate(), do NOT install vllm (it breaks the torch/CUDA pin)")

    print("GT0 SMOKE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
