"""Loop-detector and forensic-row unit tests for the E07 degenerate probe."""
import random

from scripts.probe_degenerate import detect_repetition_loop, forensic_row


def test_healthy_varied_prose_not_looping():
    rng = random.Random(7)
    text = " ".join(f"w{rng.randrange(1000)}" for _ in range(3000))
    assert detect_repetition_loop(text)["looping"] is False


def test_pure_repetition_loops_with_period():
    text = "The same sentence keeps repeating here. " * 500  # unit length 40
    r = detect_repetition_loop(text)
    assert r["looping"] is True
    assert r["period"] == 40
    assert r["onset"] == 0


def test_single_char_runaway():
    assert detect_repetition_loop("a" * 5000) == {"looping": True, "period": 1, "onset": 0}


def test_loop_after_healthy_prefix_reports_onset():
    prefix = " ".join(f"w{i}" for i in range(600))
    unit = "and then it loops forever with this exact phrase. "  # length 51
    text = prefix + unit * 300
    r = detect_repetition_loop(text)
    assert r["looping"] is True
    assert r["period"] == len(unit)
    assert len(prefix) <= r["onset"] < len(prefix) + len(unit)


def test_short_text_not_looping():
    assert detect_repetition_loop("abc " * 10)["looping"] is False


def test_forensic_row_shape():
    rec = {"id": "CIF-012", "response": "x" * 10, "finish_reason": "stop"}
    row = forensic_row("E07-reasoning-full75", "qwen-9b", rec)
    assert row["source"] == "E07-reasoning-full75"
    assert row["model"] == "qwen-9b"
    assert row["id"] == "CIF-012"
    assert row["finish_reason"] == "stop"
    assert row["content_chars"] == 10
    assert row["reasoning_chars"] == 0
    assert row["content_loop"]["looping"] is False
    assert row["reasoning_loop"]["looping"] is False
