"""T1.3 shared-utility tests (test-first, RED). extract_final + LoRA config are
the estimand-relevant knobs frozen by PREREG amendment 2026-07-16 (c); assert
them here so a silent drift fails a test (PREREG §9: no silent config drift)."""
from common import LORA_TARGET_MODULES, extract_final, lora_config


def test_extract_final_takes_post_marker_text():
    assert extract_final("scaffold\n===FINAL===\nthe answer") == "the answer"


def test_extract_final_uses_last_marker():
    assert extract_final("a\n===FINAL===\nb\n===FINAL===\nc") == "c"


def test_extract_final_no_marker_returns_whole_text():
    assert extract_final("  just an answer  ") == "just an answer"


def test_lora_targets_all_attention_and_mlp_linears():
    # PREREG amendment (c): q,k,v,o_proj + gate,up,down_proj — nothing else.
    assert set(LORA_TARGET_MODULES) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def test_lora_config_frozen_hyperparameters():
    cfg = lora_config()
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.bias == "none"
    assert set(cfg.target_modules) == set(LORA_TARGET_MODULES)
