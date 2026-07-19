"""T1.3 callback tests (test-first). CSVLogger must survive a hard kill (JSONL is
streamed live), TimeBudget must trip should_training_stop, and GenerateMeter must
count only newly-generated tokens. No GPU / no trainer needed."""
import csv
import json
import types

from callbacks import CSVLogger, GenerateMeter, TimeBudget


class _State:
    def __init__(self, step): self.global_step = step


def _args(): return types.SimpleNamespace()


def test_csv_logger_streams_jsonl_and_rebuilds_csv(tmp_path):
    csv_path = tmp_path / "run.csv"
    lg = CSVLogger(str(csv_path))
    ctrl = types.SimpleNamespace()
    lg.on_train_begin(_args(), _State(0), ctrl)
    lg.on_log(_args(), _State(1), ctrl, logs={"loss": 2.0, "learning_rate": 1e-4})
    lg.on_log(_args(), _State(2), ctrl, logs={"loss": 1.5, "reward": 0.3})
    # JSONL streamed live (2 lines) — present even if on_train_end never fires
    lines = (tmp_path / "run.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["loss"] == 2.0
    rows = list(csv.DictReader(csv_path.open()))
    assert [r["step"] for r in rows] == ["1", "2"]
    assert rows[1]["reward"] == "0.3"           # union header picks up late keys


def test_time_budget_trips_stop_when_exceeded():
    cb = TimeBudget(budget_s=-1)                 # already over budget
    ctrl = types.SimpleNamespace(should_training_stop=False)
    cb.on_train_begin(_args(), _State(0), ctrl)
    cb.on_step_end(_args(), _State(1), ctrl)
    assert ctrl.should_training_stop is True and cb.hit is True


def test_time_budget_does_not_trip_within_budget():
    cb = TimeBudget(budget_s=10_000)
    ctrl = types.SimpleNamespace(should_training_stop=False)
    cb.on_train_begin(_args(), _State(0), ctrl)
    cb.on_step_end(_args(), _State(1), ctrl)
    assert ctrl.should_training_stop is False and cb.hit is False


def test_generate_meter_counts_new_tokens():
    # fake a model whose .generate returns a [batch, in_len+new] tensor-like object
    class FakeSeq:
        def __init__(self, b, t): self.shape = (b, t); self.ndim = 2

    class FakeModel:
        def generate(self, input_ids=None, **kw):
            return FakeSeq(input_ids.shape[0], input_ids.shape[1] + 10)

    class FakeIn:
        def __init__(self, b, t): self.shape = (b, t); self.ndim = 2

    m = FakeModel()
    meter = GenerateMeter().attach(m)
    m.generate(input_ids=FakeIn(4, 20))          # 4 seqs x 10 new tokens = 40
    assert meter.tokens == 40
    assert meter.calls == 1
    assert meter.tok_per_s >= 0
