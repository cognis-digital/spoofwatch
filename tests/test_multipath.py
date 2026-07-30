import math

import pytest

from spoofwatch import multipath as mp
from spoofwatch.multipath import Epoch


# --------------------------------------------------------------------------
# deterministic synthetic channels
# --------------------------------------------------------------------------
def _epochs(sat, els, cn0s, resid):
    return [Epoch(float(i), sat, cn0s[i], els[i], code_m=resid[i], phase_m=0.0)
            for i in range(len(els))]


def clean(sat="G01", n=12):
    """Monotone-rising C/N0, tiny elevation-independent residual: not multipath."""
    els = [20.0 + i * 4.0 for i in range(n)]
    cn0 = [30.0 + 0.3 * e + (0.03 if i % 2 else -0.03) for i, e in enumerate(els)]
    resid = [(0.01 if i % 2 else -0.01) for i in range(n)]
    return _epochs(sat, els, cn0, resid)


def fading(sat="G02", n=12, amp=6.0):
    """Oscillatory C/N0 fading, flat residual: fading fingerprint only."""
    els = [20.0 + i * 4.0 for i in range(n)]
    cn0 = [30.0 + 0.3 * e + amp * math.sin(2.0 * i) for i, e in enumerate(els)]
    resid = [(0.01 if i % 2 else -0.01) for i in range(n)]
    return _epochs(sat, els, cn0, resid)


def elev_dep(sat="G03", amp=6.0):
    """Smooth C/N0, residual large at low elevation: elevation fingerprint only."""
    els = [10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0, 66.0]
    cn0 = [30.0 + 0.3 * e for e in els]            # perfectly smooth, no fading
    resid = [amp * math.exp(-e / 18.0) * math.sin(1.7 * i)
             for i, e in enumerate(els)]
    return _epochs(sat, els, cn0, resid)


def both(sat="G04", amp_cn0=6.0, amp_res=6.0):
    """Both fingerprints present — the strongest multipath case."""
    els = [10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0, 66.0]
    cn0 = [30.0 + 0.3 * e + amp_cn0 * math.sin(2.0 * i) for i, e in enumerate(els)]
    resid = [amp_res * math.exp(-e / 18.0) * math.sin(1.7 * i)
             for i, e in enumerate(els)]
    return _epochs(sat, els, cn0, resid)


# --------------------------------------------------------------------------
# primitives: statistics
# --------------------------------------------------------------------------
def test_mean_empty():
    assert mp._mean([]) == 0.0


def test_mean_basic():
    assert mp._mean([1.0, 2.0, 3.0]) == 2.0


def test_stdev_single():
    assert mp._stdev([5.0]) == 0.0


def test_stdev_known():
    assert abs(mp._stdev([1.0, 3.0]) - 1.0) < 1e-12


@pytest.mark.parametrize("slope,intercept", [(2.0, 1.0), (-3.0, 5.0), (0.5, -2.0)])
def test_linfit_recovers_line(slope, intercept):
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [slope * x + intercept for x in xs]
    s, b = mp.linfit(xs, ys)
    assert abs(s - slope) < 1e-9
    assert abs(b - intercept) < 1e-9


def test_linfit_zero_variance_x():
    s, b = mp.linfit([4.0, 4.0, 4.0], [2.0, 4.0, 6.0])
    assert s == 0.0
    assert abs(b - 4.0) < 1e-9


def test_linfit_empty():
    assert mp.linfit([], []) == (0.0, 0.0)


# --------------------------------------------------------------------------
# zero_crossings
# --------------------------------------------------------------------------
def test_zero_crossings_none():
    assert mp.zero_crossings([1.0, 2.0, 3.0]) == 0


def test_zero_crossings_monotone_step():
    assert mp.zero_crossings([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]) == 1


def test_zero_crossings_alternating():
    assert mp.zero_crossings([1.0, -1.0, 1.0, -1.0, 1.0]) == 4


def test_zero_crossings_skips_exact_zero():
    # zeros are ignored, so this is a single -> + transition
    assert mp.zero_crossings([-1.0, 0.0, 0.0, 1.0]) == 1


@pytest.mark.parametrize("k", [1.0, 1.7, 2.0, 3.0])
def test_zero_crossings_sine_is_oscillatory(k):
    vals = [math.sin(k * i) for i in range(12)]
    assert mp.zero_crossings(vals) >= 2


# --------------------------------------------------------------------------
# residual_series: carrier-ambiguity removal
# --------------------------------------------------------------------------
def test_residual_series_starts_at_zero():
    tr = _epochs("G01", [40.0, 45.0, 50.0], [45.0, 46.0, 47.0], [100.0, 101.0, 102.5])
    ser = mp.residual_series(tr)
    assert ser[0][1] == 0.0
    assert abs(ser[1][1] - 1.0) < 1e-9
    assert abs(ser[2][1] - 2.5) < 1e-9


def test_residual_series_ambiguity_cancels():
    # a constant integer-cycle offset on both epochs' code cancels via first-ref
    a = _epochs("G01", [40.0, 50.0], [45.0, 46.0], [0.0, 1.0])
    b = _epochs("G01", [40.0, 50.0], [45.0, 46.0], [1000.0, 1001.0])
    assert mp.residual_series(a) == mp.residual_series(b)


def test_residual_series_carries_elevation():
    tr = _epochs("G01", [12.0, 34.0], [45.0, 46.0], [0.0, 0.0])
    ser = mp.residual_series(tr)
    assert ser[0][0] == 12.0 and ser[1][0] == 34.0


def test_residual_series_empty():
    assert mp.residual_series([]) == []


def test_residual_series_sorts_by_ts():
    e0 = Epoch(5.0, "G1", 45.0, 40.0, code_m=5.0)
    e1 = Epoch(1.0, "G1", 45.0, 30.0, code_m=1.0)
    ser = mp.residual_series([e0, e1])
    # first by time is the ts=1 sample -> referenced to it
    assert ser[0][0] == 30.0 and ser[0][1] == 0.0


# --------------------------------------------------------------------------
# detrend_cn0
# --------------------------------------------------------------------------
def test_detrend_removes_linear_elevation_trend():
    tr = _epochs("G1", [10.0, 20.0, 30.0, 40.0], [31.0, 33.0, 35.0, 37.0], [0, 0, 0, 0])
    res = mp.detrend_cn0(tr)
    assert all(abs(r) < 1e-9 for r in res)


def test_detrend_leaves_oscillation():
    tr = fading(n=12, amp=5.0)
    res = mp.detrend_cn0(tr)
    assert mp._stdev(res) > 2.0
    assert mp.zero_crossings(res) >= 2


def test_detrend_empty():
    assert mp.detrend_cn0([]) == []


# --------------------------------------------------------------------------
# elevation_bands / residual_ratio / residual_slope
# --------------------------------------------------------------------------
def test_elevation_bands_split():
    ser = [(10.0, 1.0), (25.0, 2.0), (30.0, 3.0), (60.0, 4.0)]
    low, high = mp.elevation_bands(ser, low_el=30.0)
    assert low == [1.0, 2.0]
    assert high == [3.0, 4.0]


def test_residual_ratio_none_when_band_sparse():
    ser = [(10.0, 5.0), (12.0, -5.0), (60.0, 0.1)]   # high band has 1 sample
    assert mp.residual_ratio(ser, min_per_band=3) is None


def test_residual_ratio_none_when_high_flat():
    ser = [(10.0, 5.0), (12.0, -5.0), (14.0, 5.0),
           (60.0, 0.0), (62.0, 0.0), (64.0, 0.0)]
    assert mp.residual_ratio(ser) is None


def test_residual_ratio_large_for_low_heavy():
    ser = [(10.0, 6.0), (12.0, -6.0), (14.0, 6.0),
           (60.0, 0.2), (62.0, -0.2), (64.0, 0.2)]
    r = mp.residual_ratio(ser)
    assert r is not None and r > 2.0


def test_residual_slope_negative_for_decaying():
    ser = [(10.0, 6.0), (20.0, 4.0), (30.0, 2.0), (40.0, 1.0), (50.0, 0.2)]
    assert mp.residual_slope(ser) < 0.0


def test_residual_slope_empty():
    assert mp.residual_slope([]) == 0.0


# --------------------------------------------------------------------------
# by_sat grouping
# --------------------------------------------------------------------------
def test_by_sat_groups_and_sorts():
    eps = [Epoch(2.0, "A", 45, 30), Epoch(1.0, "A", 45, 20), Epoch(1.0, "B", 45, 40)]
    tracks = mp.by_sat(eps)
    assert set(tracks) == {"A", "B"}
    assert [e.ts for e in tracks["A"]] == [1.0, 2.0]


# --------------------------------------------------------------------------
# analyze_sat — clean channel
# --------------------------------------------------------------------------
def test_clean_not_multipath():
    r = mp.analyze_sat("G01", clean())
    assert r.multipath is False
    assert r.fading is False
    assert r.elevation_dependent is False
    assert r.confidence == 0.0


@pytest.mark.parametrize("n", [6, 8, 10, 12, 16, 20])
def test_clean_never_flagged_over_lengths(n):
    r = mp.analyze_sat("G01", clean(n=n))
    assert r.multipath is False


def test_clean_low_fluctuation():
    r = mp.analyze_sat("G01", clean())
    assert r.fluct_std_dbhz < mp.FLUCT_STD_DBHZ


# --------------------------------------------------------------------------
# analyze_sat — fading fingerprint
# --------------------------------------------------------------------------
def test_fading_flagged():
    r = mp.analyze_sat("G02", fading())
    assert r.multipath is True
    assert r.fading is True


def test_fading_reason_mentions_cn0():
    r = mp.analyze_sat("G02", fading())
    assert "C/N0 fading" in r.reason


def test_fading_high_std():
    r = mp.analyze_sat("G02", fading(amp=6.0))
    assert r.fluct_std_dbhz > mp.FLUCT_STD_DBHZ


def test_fading_has_crossings():
    r = mp.analyze_sat("G02", fading())
    assert r.n_crossings >= 2


@pytest.mark.parametrize("amp", [3.0, 4.0, 5.0, 6.0, 8.0])
def test_fading_amplitude_sweep_flags(amp):
    r = mp.analyze_sat("G02", fading(amp=amp))
    assert r.fading is True and r.multipath is True


@pytest.mark.parametrize("amp", [0.1, 0.3, 0.6])
def test_small_amplitude_not_fading(amp):
    r = mp.analyze_sat("G02", fading(amp=amp))
    assert r.fading is False


# --------------------------------------------------------------------------
# analyze_sat — elevation-dependence fingerprint
# --------------------------------------------------------------------------
def test_elev_dep_flagged():
    r = mp.analyze_sat("G03", elev_dep())
    assert r.multipath is True
    assert r.elevation_dependent is True


def test_elev_dep_not_fading():
    r = mp.analyze_sat("G03", elev_dep())
    assert r.fading is False


def test_elev_dep_ratio_reported():
    r = mp.analyze_sat("G03", elev_dep())
    assert r.residual_ratio is not None and r.residual_ratio > mp.RESIDUAL_RATIO


def test_elev_dep_slope_negative():
    r = mp.analyze_sat("G03", elev_dep())
    assert r.residual_slope_m_per_deg < 0.0


def test_elev_dep_low_exceeds_high():
    r = mp.analyze_sat("G03", elev_dep())
    assert r.low_residual_std_m > r.high_residual_std_m


def test_elev_dep_reason_mentions_residuals():
    r = mp.analyze_sat("G03", elev_dep())
    assert "elevation-dependent" in r.reason


@pytest.mark.parametrize("amp", [6.0, 8.0, 10.0, 12.0])
def test_elev_dep_amplitude_sweep(amp):
    r = mp.analyze_sat("G03", elev_dep(amp=amp))
    assert r.elevation_dependent is True


# --------------------------------------------------------------------------
# analyze_sat — both fingerprints
# --------------------------------------------------------------------------
def test_both_flagged():
    r = mp.analyze_sat("G04", both())
    assert r.fading is True
    assert r.elevation_dependent is True
    assert r.multipath is True


def test_both_high_confidence():
    r = mp.analyze_sat("G04", both(amp_cn0=6.0, amp_res=8.0))
    assert r.confidence > 0.5


def test_both_reason_lists_both():
    r = mp.analyze_sat("G04", both())
    assert "C/N0 fading" in r.reason and "elevation-dependent" in r.reason


# --------------------------------------------------------------------------
# analyze_sat — guards / invariants
# --------------------------------------------------------------------------
def test_too_few_epochs():
    r = mp.analyze_sat("G01", clean(n=4))
    assert r.multipath is False
    assert "need >=" in r.reason


def test_epoch_count_reported():
    r = mp.analyze_sat("G02", fading(n=14))
    assert r.n_epochs == 14


@pytest.mark.parametrize("gen", [clean, fading, elev_dep, both])
def test_confidence_bounded(gen):
    r = mp.analyze_sat("GX", gen())
    assert 0.0 <= r.confidence <= 1.0


@pytest.mark.parametrize("gen,flag", [(clean, False), (fading, True),
                                      (elev_dep, True), (both, True)])
def test_multipath_flag_matches_generator(gen, flag):
    r = mp.analyze_sat("GX", gen())
    assert r.multipath is flag


def test_analyze_deterministic():
    a = mp.analyze_sat("G04", both())
    b = mp.analyze_sat("G04", both())
    assert a == b


def test_clean_confidence_zero_specifically():
    assert mp.analyze_sat("G01", clean()).confidence == 0.0


# --------------------------------------------------------------------------
# check — multi-satellite aggregation
# --------------------------------------------------------------------------
def test_check_flags_only_dirty():
    eps = clean("G01") + fading("G02") + elev_dep("G03")
    res = mp.check(eps)
    assert res["multipath"] is True
    assert "G01" not in res["flagged"]
    assert "G02" in res["flagged"] and "G03" in res["flagged"]


def test_check_all_clean():
    eps = clean("G01") + clean("G05")
    res = mp.check(eps)
    assert res["multipath"] is False
    assert res["flagged"] == []
    assert res["confidence"] == 0.0


def test_check_counts():
    eps = clean("G01") + fading("G02") + elev_dep("G03") + both("G04")
    res = mp.check(eps)
    assert res["n_satellites"] == 4
    assert res["n_flagged"] == 3


def test_check_flagged_sorted():
    eps = both("G09") + fading("G02") + elev_dep("G05")
    res = mp.check(eps)
    assert res["flagged"] == sorted(res["flagged"])


def test_check_confidence_is_max_channel():
    eps = fading("G02") + both("G04", amp_cn0=6.0, amp_res=8.0)
    res = mp.check(eps)
    per = {d["sat_id"]: d["confidence"] for d in res["per_satellite"]}
    assert res["confidence"] == max(per.values())


def test_check_per_satellite_keys():
    res = mp.check(fading("G02"))
    d = res["per_satellite"][0]
    for key in ("sat_id", "n_epochs", "fluct_std_dbhz", "n_crossings",
                "low_residual_std_m", "high_residual_std_m", "residual_ratio",
                "residual_slope_m_per_deg", "fading", "elevation_dependent",
                "multipath", "confidence", "reason"):
        assert key in d


def test_check_empty():
    res = mp.check([])
    assert res["n_satellites"] == 0
    assert res["flagged"] == []
    assert res["multipath"] is False


def test_check_confidence_bounded():
    eps = clean("G01") + fading("G02") + elev_dep("G03") + both("G04")
    res = mp.check(eps)
    assert 0.0 <= res["confidence"] <= 1.0


def test_module_exported():
    import spoofwatch
    assert "multipath" in spoofwatch.__all__
