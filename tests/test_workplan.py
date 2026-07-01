from pipeline.monitor import Stage, WorkPlan


def test_stage_pct_and_remaining():
    s = Stage(name="generate", total=225, done=45)
    assert s.pct == 20.0
    assert s.remaining == 180


def test_stage_pct_zero_total():
    assert Stage(name="validate", total=0).pct == 0.0


def test_stage_rate_and_eta_from_durations():
    s = Stage(name="grade", total=10, done=2)
    assert s.rate_s is None and s.eta_s() is None
    for d in (2.0, 4.0):        # mean 3.0s/item, 8 remaining
        s.record_duration(d)
    assert s.rate_s == 3.0
    assert s.eta_s() == 24.0


def test_stage_duration_window_caps_at_20():
    s = Stage(name="generate", total=100)
    for i in range(30):
        s.record_duration(float(i))
    assert len(s._durations) == 20
    assert s._durations[0] == 10.0     # oldest 10 dropped


def test_workplan_for_all_full_run():
    plan = WorkPlan.for_step("all", limit=75, n_candidates=3)
    totals = {s.name: s.total for s in plan.stages}
    assert totals == {
        "connectivity": 4, "load": 75, "generate": 225, "grade": 225,
        "classify": 75, "validate": 1, "aggregate": 1,
    }
    assert sum(totals.values()) == 606
    assert [s.name for s in plan.stages][0] == "connectivity"


def test_workplan_single_step():
    plan = WorkPlan.for_step("grade", limit=10, n_candidates=3)
    assert [s.name for s in plan.stages] == ["grade"]
    assert plan.get("grade").total == 30
