import math

from spoofwatch import holdover
from spoofwatch.holdover import Oscillator


def test_known_classes_present():
    for name in ("TCXO", "OCXO", "Rubidium", "Cesium"):
        assert name in holdover.OSCILLATORS


def test_time_error_zero_at_start():
    assert holdover.time_error_s(0, "OCXO") == 0.0


def test_time_error_grows():
    early = holdover.time_error_s(10, "OCXO")
    late = holdover.time_error_s(100, "OCXO")
    assert late > early > 0


def test_time_error_offset_term():
    # with zero drift, TE = y0 * t exactly
    osc = Oscillator("test", y0=1e-9, drift_per_s=0.0)
    assert abs(holdover.time_error_s(100, osc) - 1e-7) < 1e-18


def test_time_error_drift_term():
    # pure drift: TE = 0.5 * D * t^2
    osc = Oscillator("test", y0=0.0, drift_per_s=2e-12)
    assert abs(holdover.time_error_s(100, osc) - 0.5 * 2e-12 * 100 * 100) < 1e-20


def test_time_error_with_te0():
    osc = Oscillator("test", y0=0.0, drift_per_s=0.0)
    assert holdover.time_error_s(50, osc, te0_s=5e-7) == 5e-7


def test_negative_time_clamped():
    assert holdover.time_error_s(-10, "OCXO") == 0.0


def test_worse_oscillator_more_error():
    t = 100
    tcxo = holdover.time_error_s(t, "TCXO")
    ocxo = holdover.time_error_s(t, "OCXO")
    cesium = holdover.time_error_s(t, "Cesium")
    assert tcxo > ocxo > cesium


def test_unknown_oscillator_raises():
    try:
        holdover.time_error_s(10, "QUARTZWATCH")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_budget_positive():
    b = holdover.holdover_budget_s(holdover.TELECOM_MASK_S, "OCXO")
    assert b > 0
    assert math.isfinite(b)


def test_budget_offset_only():
    osc = Oscillator("test", y0=1e-9, drift_per_s=0.0)
    # threshold 1e-6 / offset 1e-9 = 1000 s
    assert abs(holdover.holdover_budget_s(1e-6, osc) - 1000.0) < 1e-6


def test_budget_ideal_infinite():
    osc = Oscillator("ideal", y0=0.0, drift_per_s=0.0)
    assert holdover.holdover_budget_s(1e-6, osc) == float("inf")


def test_budget_already_breached():
    osc = Oscillator("test", y0=1e-9, drift_per_s=0.0)
    assert holdover.holdover_budget_s(1e-6, osc, te0_s=2e-6) == 0.0


def test_budget_consistent_with_time_error():
    # at exactly the budget time, time error should equal the threshold
    thr = holdover.TELECOM_MASK_S
    b = holdover.holdover_budget_s(thr, "TCXO")
    te = holdover.time_error_s(b, "TCXO")
    assert abs(te - thr) < thr * 1e-3


def test_worse_oscillator_shorter_budget():
    thr = holdover.TELECOM_MASK_S
    assert holdover.holdover_budget_s(thr, "TCXO") < holdover.holdover_budget_s(thr, "OCXO")


def test_check_within_early():
    r = holdover.check_holdover(1.0, holdover.TELECOM_MASK_S, "OCXO")
    assert r["within"] is True
    assert r["breach"] is False
    assert r["fraction_used"] < 1.0


def test_check_breach_late():
    b = holdover.holdover_budget_s(holdover.TELECOM_MASK_S, "TCXO")
    r = holdover.check_holdover(b * 2, holdover.TELECOM_MASK_S, "TCXO")
    assert r["breach"] is True
    assert r["within"] is False


def test_check_reports_ns():
    r = holdover.check_holdover(100.0, holdover.TELECOM_MASK_S, "OCXO")
    assert r["time_error_ns"] > 0
    assert abs(r["time_error_ns"] - r["time_error_s"] * 1e9) < 1e-3


def test_check_remaining_shrinks():
    r1 = holdover.check_holdover(10.0, holdover.TELECOM_MASK_S, "TCXO")
    r2 = holdover.check_holdover(100.0, holdover.TELECOM_MASK_S, "TCXO")
    assert r2["remaining_s"] < r1["remaining_s"]


def test_check_oscillator_name():
    r = holdover.check_holdover(10.0, holdover.TELECOM_MASK_S, "Rubidium")
    assert r["oscillator"] == "Rubidium"


def test_check_custom_oscillator():
    osc = Oscillator("mine", y0=5e-10, drift_per_s=1e-13)
    r = holdover.check_holdover(50.0, 1e-6, osc)
    assert r["oscillator"] == "mine"


def test_project_length_and_monotonic():
    rows = holdover.project([0, 10, 20, 30], "OCXO")
    assert len(rows) == 4
    errs = [r["time_error_ns"] for r in rows]
    assert errs == sorted(errs)


def test_project_starts_at_te0():
    rows = holdover.project([0, 5], Oscillator("t", y0=0.0, drift_per_s=0.0), te0_s=1e-7)
    assert abs(rows[0]["time_error_ns"] - 100.0) < 1e-6


def test_utc_traceable_budget_shorter_than_telecom():
    # tighter mask -> shorter budget
    assert holdover.holdover_budget_s(holdover.UTC_TRACEABLE_S, "OCXO") < \
           holdover.holdover_budget_s(holdover.TELECOM_MASK_S, "OCXO")
