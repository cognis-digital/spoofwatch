import math

from spoofwatch import skygeom


def _sat(sid, az, el, const="GPS"):
    return skygeom.SatGeom(sid, az, el, const)


def _good_sky():
    # well-spread constellation: four quadrants + a high sat
    return [
        _sat("G1", 45, 20),
        _sat("G2", 135, 35),
        _sat("G3", 225, 25),
        _sat("G4", 315, 40),
        _sat("G5", 0, 75),
        _sat("G6", 180, 60),
    ]


def _clustered_sky():
    # single-source spoofer: everything from one narrow az/el band
    return [
        _sat("S1", 90, 45),
        _sat("S2", 91, 46),
        _sat("S3", 89, 44),
        _sat("S4", 90.5, 45.5),
        _sat("S5", 89.5, 44.5),
    ]


# --------------------------------------------------------------------------- #
# enu_los
# --------------------------------------------------------------------------- #
def test_los_zenith_is_up():
    e, n, u = skygeom.enu_los(0, 90)
    assert abs(u - 1.0) < 1e-9
    assert abs(e) < 1e-9 and abs(n) < 1e-9


def test_los_north_horizon():
    e, n, u = skygeom.enu_los(0, 0)
    assert abs(n - 1.0) < 1e-9
    assert abs(e) < 1e-9 and abs(u) < 1e-9


def test_los_east_horizon():
    e, n, u = skygeom.enu_los(90, 0)
    assert abs(e - 1.0) < 1e-9


def test_los_is_unit():
    e, n, u = skygeom.enu_los(123, 37)
    assert abs(math.sqrt(e * e + n * n + u * u) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# geometry_matrix
# --------------------------------------------------------------------------- #
def test_geometry_matrix_shape():
    G = skygeom.geometry_matrix(_good_sky())
    assert len(G) == 6
    assert all(len(row) == 4 for row in G)


def test_geometry_matrix_clock_column_is_one():
    G = skygeom.geometry_matrix(_good_sky())
    assert all(row[3] == 1.0 for row in G)


# --------------------------------------------------------------------------- #
# dop
# --------------------------------------------------------------------------- #
def test_dop_too_few_sats():
    d = skygeom.dop([_sat("G1", 0, 45), _sat("G2", 90, 45)])
    assert d["available"] is False


def test_dop_good_geometry_low():
    d = skygeom.dop(_good_sky())
    assert d["available"] is True
    assert d["degenerate"] is False
    assert d["gdop"] < skygeom.GDOP_SUSPECT


def test_dop_has_all_components():
    d = skygeom.dop(_good_sky())
    for k in ("gdop", "pdop", "hdop", "vdop", "tdop"):
        assert k in d and d[k] >= 0.0


def test_gdop_ge_pdop():
    d = skygeom.dop(_good_sky())
    assert d["gdop"] >= d["pdop"] - 1e-9


def test_pdop_relates_to_hv():
    d = skygeom.dop(_good_sky())
    assert abs(d["pdop"] ** 2 - (d["hdop"] ** 2 + d["vdop"] ** 2)) < 1e-2


def test_dop_clustered_geometry_high():
    d = skygeom.dop(_clustered_sky())
    # near-degenerate: DOP is large (or flagged degenerate/inf)
    assert d.get("degenerate") or d["gdop"] > skygeom.GDOP_SUSPECT


def test_dop_identical_sats_degenerate():
    sats = [_sat(f"X{i}", 90, 45) for i in range(5)]
    d = skygeom.dop(sats)
    assert d["degenerate"] is True
    assert math.isinf(d["gdop"])


# --------------------------------------------------------------------------- #
# az spread helper
# --------------------------------------------------------------------------- #
def test_az_spread_wrap():
    # 350 and 10 degrees are 20 apart, not 340
    spread = skygeom._az_spread_deg([350, 10])
    assert spread < 40


def test_az_spread_opposite():
    spread = skygeom._az_spread_deg([0, 180])
    assert spread > 150


def test_az_spread_single():
    assert skygeom._az_spread_deg([90]) == 0.0


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def test_check_good_sky_not_suspect():
    c = skygeom.check(_good_sky())
    assert c["available"] is True
    assert c["suspect"] is False
    assert c["confidence"] == 0.0


def test_check_clustered_suspect():
    c = skygeom.check(_clustered_sky())
    assert c["suspect"] is True
    assert c["confidence"] > 0.0


def test_check_low_diversity_flag():
    c = skygeom.check(_clustered_sky())
    assert c["low_diversity"] is True


def test_check_reports_spreads():
    c = skygeom.check(_good_sky())
    assert c["el_spread_deg"] > skygeom.MIN_EL_SPREAD_DEG
    assert c["az_spread_deg"] > skygeom.MIN_AZ_SPREAD_DEG


def test_check_degenerate_confidence_one():
    sats = [_sat(f"X{i}", 90, 45) for i in range(6)]
    c = skygeom.check(sats)
    assert c["degenerate_geometry"] is True
    assert c["confidence"] == 1.0


def test_check_too_few_sats():
    c = skygeom.check([_sat("G1", 0, 45), _sat("G2", 90, 45)])
    assert c["available"] is False
    assert c["suspect"] is False


def test_check_confidence_bounded():
    c = skygeom.check(_clustered_sky())
    assert 0.0 <= c["confidence"] <= 1.0


def test_check_severe_gdop_degenerate():
    # a barely-non-degenerate but very high DOP geometry is flagged
    sats = [_sat("A", 88, 80), _sat("B", 90, 81), _sat("C", 92, 80),
            _sat("D", 90, 82), _sat("E", 89, 79)]
    c = skygeom.check(sats)
    assert c["suspect"] is True


def test_moderate_gdop_partial_confidence():
    # construct a middling geometry: three clustered + one spread
    sats = [_sat("A", 80, 40), _sat("B", 100, 42), _sat("C", 90, 55),
            _sat("D", 95, 38)]
    c = skygeom.check(sats)
    assert 0.0 <= c["confidence"] <= 1.0
