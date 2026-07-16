"""PREREG §5: training code must never read data/holdout/. Enforced as a static
guard — any .py under training/ that mentions the holdout path fails this test.
Passes trivially while training/ is empty (T1.3) and keeps guarding as it fills.
"""
import pathlib

_TRAINING = pathlib.Path(__file__).resolve().parents[1] / "training"


def test_training_code_never_references_holdout():
    offenders = []
    for py in _TRAINING.rglob("*.py"):
        text = py.read_text(encoding="utf-8").lower()
        if "holdout" in text or "data/holdout" in text:
            offenders.append(py.relative_to(_TRAINING).as_posix())
    assert not offenders, (
        f"training/ code references the holdout — leakage risk: {offenders}")
