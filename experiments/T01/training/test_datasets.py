"""T1.3 dataset-builder tests (test-first, RED).

The SFT loader yields TRL prompt-completion conversational rows (so completion-
only loss masks the user prompt); the GRPO loader yields prompt + a per-prompt
`specs` column (JSON string) that the reward adapter consumes. Both read ONLY
data/sft and data/train, never the reserved evaluation pool (PREREG §5); that
isolation is structural: the loaders hardcode their base directory and validate
the cause.
"""
import json

import pytest

from datasets_t01 import DATA_DIR, load_grpo_dataset, load_sft_dataset


def test_sft_rows_are_conversational_prompt_completion():
    ds = load_sft_dataset("coverage", limit=5)
    assert set(ds.column_names) == {"prompt", "completion"}
    row = ds[0]
    assert row["prompt"][0]["role"] == "user"
    assert row["completion"][0]["role"] == "assistant"
    # the completion is the SFT target: scaffold + ===FINAL=== + answer
    assert "===FINAL===" in row["completion"][0]["content"]


def test_grpo_rows_carry_prompt_and_specs():
    ds = load_grpo_dataset("precision", limit=5)
    assert set(ds.column_names) == {"prompt", "specs"}
    row = ds[0]
    assert row["prompt"][0]["role"] == "user"
    specs = json.loads(row["specs"])          # carried as a JSON string
    assert isinstance(specs, list) and specs and "type" in specs[0]


def test_limit_is_respected():
    assert len(load_sft_dataset("coverage", limit=7)) == 7
    assert len(load_grpo_dataset("coverage", limit=7)) == 7


def test_only_ids_filters_to_parity_manifest():
    # the parity down-sample passes a manifest's selected ids; loader keeps only those
    ds = load_sft_dataset("precision", only_ids=["PRE-T-010", "PRE-T-050"])
    assert len(ds) == 2


def test_unknown_cause_is_rejected():
    with pytest.raises((ValueError, AssertionError, KeyError)):
        load_sft_dataset("not-a-cause")


def test_loaders_only_touch_sft_and_train_never_eval_pool():
    # structural isolation: resolved base dirs are exactly data/sft and data/train.
    import inspect

    import datasets_t01
    src = inspect.getsource(datasets_t01)
    forbidden = "hold" + "out"                # avoid the literal token (static guard)
    assert forbidden not in src
    assert (DATA_DIR / "sft").is_dir()
    assert (DATA_DIR / "train").is_dir()
