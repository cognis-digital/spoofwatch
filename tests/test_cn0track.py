import math

from spoofwatch import cn0track
from spoofwatch.cn0track import SatObs


def _genuine():
    # C/N0 rises with elevation, natural scatter
    return [
        SatObs("G01", 33.0, 8.0),
        SatObs("G02", 38.0, 22.0),
        SatObs("G03", 43.0, 40.0),
        SatObs("G04", 47.0, 62.0),
        SatObs("G05", 50.0, 84.0),
    ]


def _spoofed():
    # all channels ~equal high power regardless of elevation
    return [
        SatObs("G01", 49.5, 8.0),
        SatObs("G02", 50.0, 25.0),
        SatObs("G03", 49.8, 45.0),
        SatObs("G04", 50.2, 66.0),
        SatObs("G05", 49.9, 85.0),
    ]


def test_linfit_recovers_line():
    slope, intercept = cn0track.linfit([0, 1, 2, 3], [1, 3, 5, 7])
    assert abs(slope - 2.0) < 1e-9
    assert abs(intercept - 1.0) < 1e-9


def test_linfit_zero_variance_x():
    slope, intercept = cn0track.linfit([5, 5, 5], [2, 4, 6])
    assert slope == 0.0
    assert abs(intercept - 4.0) < 1e-9


def test_linfit_empty():
    assert cn0track.linfit([], []) == (0.0, 0.0)


def test_predicted_cn0():
    assert cn0track.predicted_cn0(0.2, 30.0, 50.0) == 40.0


def test_genuine_not_suspect():
    v = cn0track.check(_genuine())
    assert v.suspect is False
    assert v.slope_dbhz_per_deg > cn0track.FLAT_SLOPE_DBHZ_PER_DEG


def test_genuine_positive_slope():
    v = cn0track.check(_genuine())
    assert v.confidence == 0.0


def test_spoofed_is_suspect():
    v = cn0track.check(_spoofed())
    assert v.suspect is True
    assert v.flat is True
    assert v.uniform is True
    assert v.confidence > 0.5


def test_spoofed_reason_set():
    v = cn0track.check(_spoofed())
    assert "single-source" in v.reason


def test_too_few_sats():
    v = cn0track.check([SatObs("G01", 45.0, 30.0), SatObs("G02", 46.0, 40.0)])
    assert v.suspect is False
    assert "need >=" in v.reason


def test_confidence_bounded():
    for obs in (_genuine(), _spoofed()):
        v = cn0track.check(obs)
        assert 0.0 <= v.confidence <= 1.0


def test_n_sats_reported():
    v = cn0track.check(_spoofed())
    assert v.n_sats == 5


def test_spread_reported():
    v = cn0track.check(_spoofed())
    assert v.spread_dbhz < cn0track.UNIFORM_SPREAD_DBHZ


def test_outlier_detection():
    obs = _genuine()
    obs.append(SatObs("G06", 62.0, 10.0))   # very bright at low elevation
    ol = cn0track.outliers(obs)
    assert any(o["sat_id"] == "G06" for o in ol)


def test_outliers_sorted_by_excess():
    obs = _genuine()
    obs.append(SatObs("G06", 60.0, 10.0))
    obs.append(SatObs("G07", 70.0, 10.0))
    ol = cn0track.outliers(obs)
    assert ol[0]["excess_dbhz"] >= ol[-1]["excess_dbhz"]


def test_no_outliers_on_genuine():
    assert cn0track.outliers(_genuine()) == []


def test_outliers_too_few_sats():
    assert cn0track.outliers([SatObs("G01", 60.0, 10.0)]) == []


def test_check_reports_outliers():
    obs = _genuine()
    obs.append(SatObs("G06", 65.0, 10.0))
    v = cn0track.check(obs)
    assert len(v.outliers) >= 1


def test_flat_but_not_uniform_not_suspect():
    # flat slope but scattered power -> not the uniform-overpower signature
    obs = [
        SatObs("G01", 30.0, 8.0),
        SatObs("G02", 55.0, 25.0),
        SatObs("G03", 35.0, 45.0),
        SatObs("G04", 52.0, 66.0),
        SatObs("G05", 33.0, 85.0),
    ]
    v = cn0track.check(obs)
    assert v.uniform is False
    assert v.suspect is False


def test_flatline_detects_frozen():
    series = {
        "G01": [45.0, 45.0, 45.0, 45.0, 45.0],     # frozen
        "G02": [40.0, 42.0, 39.0, 41.0, 43.0],     # lively
    }
    r = cn0track.flatline(series)
    assert r["suspect"] is True
    assert any(f["sat_id"] == "G01" for f in r["frozen"])
    assert "G02" not in [f["sat_id"] for f in r["frozen"]]


def test_flatline_confidence_bounded():
    series = {"G01": [45.0] * 6, "G02": [45.0] * 6}
    r = cn0track.flatline(series)
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["n_frozen"] == 2


def test_flatline_skips_short_series():
    series = {"G01": [45.0, 45.0]}
    r = cn0track.flatline(series)
    assert r["n_checked"] == 0
    assert r["suspect"] is False


def test_flatline_no_frozen():
    series = {"G01": [40.0, 42.0, 38.0, 44.0, 41.0]}
    r = cn0track.flatline(series)
    assert r["suspect"] is False
    assert r["confidence"] == 0.0


def test_flatline_sorted_by_stdev():
    series = {
        "A": [45.0, 45.1, 45.0, 45.0],
        "B": [45.0, 45.0, 45.0, 45.0],
    }
    r = cn0track.flatline(series)
    assert r["frozen"][0]["stdev_dbhz"] <= r["frozen"][-1]["stdev_dbhz"]
