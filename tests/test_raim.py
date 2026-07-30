import math

import pytest

from spoofwatch import ecef, raim


def _clean_measurements(lat=55.0, lon=20.0, alt=100.0, clock_m=50.0,
                        azels=None, constellation="GPS", ids_prefix="G"):
    azels = azels or [(30, 60), (120, 45), (210, 50), (300, 40), (0, 70), (160, 30)]
    rx = ecef.geodetic_to_ecef(lat, lon, alt)
    ms = []
    for i, (az, el) in enumerate(azels):
        s = ecef.sat_ecef_from_azel(lat, lon, az, el)
        rng = math.dist(s, rx)
        ms.append(raim.Measurement(f"{ids_prefix}{i}", s[0], s[1], s[2],
                                   rng + clock_m, constellation))
    return ms, rx


def test_solve_recovers_position():
    ms, rx = _clean_measurements()
    sol = raim.solve_position(ms, x0=rx)
    assert sol.converged
    assert math.dist([sol.x, sol.y, sol.z], rx) < 1e-3
    assert sol.rms < 1e-3


def test_solve_recovers_clock():
    ms, rx = _clean_measurements(clock_m=123.0)
    sol = raim.solve_position(ms, x0=rx)
    assert math.isclose(sol.clocks["GPS"], 123.0, abs_tol=1e-2)


def test_solve_lla_matches():
    ms, rx = _clean_measurements(lat=48.85, lon=2.35, alt=35.0)
    sol = raim.solve_position(ms, x0=rx)
    lat, lon, alt = sol.lla
    assert math.isclose(lat, 48.85, abs_tol=1e-4)
    assert math.isclose(lon, 2.35, abs_tol=1e-4)


def test_underdetermined_raises():
    ms, _ = _clean_measurements(azels=[(30, 60), (120, 45), (210, 50)])
    with pytest.raises(ValueError):
        raim.solve_position(ms[:3])   # 3 meas, 4 unknowns


def test_raim_clean_no_fault():
    ms, rx = _clean_measurements()
    r = raim.raim_check(ms, x0=rx)
    assert r["raim_available"] is True
    assert r["fault"] is False
    assert r["confidence"] == 0.0


def test_raim_detects_fault():
    ms, rx = _clean_measurements()
    ms[2].pseudorange += 400.0
    r = raim.raim_check(ms, x0=rx)
    assert r["fault"] is True
    assert r["rms_m"] > raim.INTEGRITY_THRESH_M
    assert r["confidence"] > 0.0


def test_araim_excludes_correct_sat():
    ms, rx = _clean_measurements()
    ms[4].pseudorange -= 500.0
    r = raim.araim_check(ms, x0=rx)
    assert r["fault"] is True
    assert r["spoof_hypothesis"] is True
    assert "G4" in r["excluded"]
    assert r["cleaned_rms_m"] < raim.INTEGRITY_THRESH_M


def test_araim_clean_returns_no_exclusion():
    ms, rx = _clean_measurements()
    r = raim.araim_check(ms, x0=rx)
    assert r["fault"] is False
    assert r["excluded"] == []


def test_multi_constellation_solution():
    # two constellations, independent clock biases, all consistent
    az_g = [(30, 60), (120, 45), (210, 50)]
    az_e = [(80, 55), (250, 40), (330, 65)]
    ms_g, rx = _clean_measurements(azels=az_g, clock_m=40.0,
                                   constellation="GPS", ids_prefix="G")
    ms_e, _ = _clean_measurements(azels=az_e, clock_m=90.0,
                                  constellation="GAL", ids_prefix="E")
    ms = ms_g + ms_e
    sol = raim.solve_position(ms, x0=rx)
    assert math.isclose(sol.clocks["GPS"], 40.0, abs_tol=1e-1)
    assert math.isclose(sol.clocks["GAL"], 90.0, abs_tol=1e-1)
    assert sol.rms < 1e-2


def test_araim_constellation_exclusion():
    # bias an entire constellation coherently -> whole-constellation exclusion
    az_g = [(30, 60), (120, 45), (210, 50), (300, 40)]
    az_e = [(80, 55), (250, 40), (330, 65), (150, 35)]
    ms_g, rx = _clean_measurements(azels=az_g, clock_m=40.0,
                                   constellation="GPS", ids_prefix="G")
    ms_e, _ = _clean_measurements(azels=az_e, clock_m=90.0,
                                  constellation="GAL", ids_prefix="E")
    # shift every Galileo sat's position (a coherent spoof of that constellation)
    for m in ms_e:
        m.pseudorange += 600.0 if m.sat_id == "E0" else 0.0
        m.pseudorange += 250.0
    ms = ms_g + ms_e
    r = raim.araim_check(ms, x0=rx)
    assert r["fault"] is True
    assert r["spoof_hypothesis"] is True


def test_solution_dof():
    ms, rx = _clean_measurements()  # 6 meas, 4 unknowns
    sol = raim.solve_position(ms, x0=rx)
    assert sol.dof == 2


def test_weight_downweights_outlier():
    ms, rx = _clean_measurements()
    ms[1].pseudorange += 300.0
    ms[1].weight = 1e-6      # trust it almost not at all
    sol = raim.solve_position(ms, x0=rx)
    # position should stay close despite the outlier being present
    assert math.dist([sol.x, sol.y, sol.z], rx) < 50.0
