import math

from spoofwatch import inertial


def test_coast_radius_grows():
    r0 = inertial.coast_radius_km(0)
    r10 = inertial.coast_radius_km(10)
    assert r10 > r0
    assert math.isclose(r0, inertial.BASE_UNCERTAINTY_KM)


def test_coast_budget():
    b = inertial.coast_budget_s(drift_km_s=0.02, base_km=0.5, tolerance_km=5.0)
    assert math.isclose(b, (5.0 - 0.5) / 0.02)


def test_coast_budget_infinite_no_drift():
    assert inertial.coast_budget_s(drift_km_s=0.0) == float("inf")


def test_stationary_matching_fix_inside():
    last = inertial.Fix(0, 55.0, 20.0, speed_mps=0, heading_deg=0)
    gnss = inertial.Fix(5, 55.0, 20.0)
    r = inertial.gate(last, gnss)
    assert r["outside"] is False
    assert r["confidence"] == 0.0


def test_teleport_flagged():
    last = inertial.Fix(0, 55.0, 20.0, speed_mps=200, heading_deg=90)
    gnss = inertial.Fix(10, 55.0, 21.0)   # far east jump
    r = inertial.gate(last, gnss)
    assert r["outside"] is True
    assert r["excess_km"] > 0
    assert r["confidence"] > 0


def test_plausible_dead_reckoned_fix_inside():
    # moving east at 200 m/s for 10s -> ~2 km east; expected point matches
    last = inertial.Fix(0, 55.0, 20.0, speed_mps=200, heading_deg=90)
    dt = 10
    exp_lat, exp_lon = inertial._project(last, dt)
    gnss = inertial.Fix(dt, exp_lat, exp_lon)
    r = inertial.gate(last, gnss)
    assert r["outside"] is False


def test_expected_point_reported():
    last = inertial.Fix(0, 55.0, 20.0, speed_mps=100, heading_deg=0)  # due north
    gnss = inertial.Fix(10, 55.0, 20.0)
    r = inertial.gate(last, gnss)
    assert r["expected_lat"] > 55.0     # moved north
    assert math.isclose(r["expected_lon"], 20.0, abs_tol=1e-3)


def test_project_north():
    # 30.83 m/s for 3600 s -> ~111 km -> ~1 deg latitude
    f = inertial.Fix(0, 0.0, 0.0, speed_mps=111000 / 3600, heading_deg=0)
    lat, lon = inertial._project(f, 3600)
    assert 0.9 < lat < 1.1


def test_bigger_drift_shorter_budget():
    b_slow = inertial.coast_budget_s(drift_km_s=0.01)
    b_fast = inertial.coast_budget_s(drift_km_s=0.1)
    assert b_fast < b_slow


def test_gate_reports_budget():
    last = inertial.Fix(0, 55.0, 20.0)
    gnss = inertial.Fix(5, 55.0, 20.0)
    r = inertial.gate(last, gnss, drift_km_s=0.02, tolerance_km=5.0)
    assert r["coast_budget_s"] > 0


def test_wider_envelope_with_more_drift():
    last = inertial.Fix(0, 55.0, 20.0, speed_mps=100, heading_deg=90)
    gnss = inertial.Fix(60, 55.0, 20.2)
    tight = inertial.gate(last, gnss, drift_km_s=0.001)
    loose = inertial.gate(last, gnss, drift_km_s=1.0)
    assert loose["envelope_radius_km"] > tight["envelope_radius_km"]
