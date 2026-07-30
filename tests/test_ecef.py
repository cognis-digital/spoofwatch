import math

from spoofwatch import ecef


def test_roundtrip_equator():
    x, y, z = ecef.geodetic_to_ecef(0.0, 0.0, 0.0)
    assert math.isclose(x, ecef.WGS84_A, rel_tol=1e-9)
    assert math.isclose(y, 0.0, abs_tol=1e-6)
    assert math.isclose(z, 0.0, abs_tol=1e-6)


def test_roundtrip_various():
    for lat, lon, alt in [(55.0, 20.0, 100.0), (-33.9, 18.4, 50.0),
                          (48.85, 2.35, 35.0), (70.0, -8.0, 0.0)]:
        xyz = ecef.geodetic_to_ecef(lat, lon, alt)
        rlat, rlon, ralt = ecef.ecef_to_geodetic(*xyz)
        assert math.isclose(rlat, lat, abs_tol=1e-6)
        assert math.isclose(rlon, lon, abs_tol=1e-6)
        assert math.isclose(ralt, alt, abs_tol=1e-3)


def test_pole():
    x, y, z = ecef.geodetic_to_ecef(90.0, 0.0, 0.0)
    assert math.isclose(x, 0.0, abs_tol=1e-6)
    assert math.isclose(y, 0.0, abs_tol=1e-6)
    assert z > 6.3e6


def test_enu_up_is_radial():
    # "up" at equator/prime-meridian points along +X
    v = ecef.enu_to_ecef_vector(0, 0, 1, 0.0, 0.0)
    assert math.isclose(v[0], 1.0, abs_tol=1e-9)
    assert math.isclose(v[1], 0.0, abs_tol=1e-9)
    assert math.isclose(v[2], 0.0, abs_tol=1e-9)


def test_enu_east():
    # "east" at prime meridian points along +Y
    v = ecef.enu_to_ecef_vector(1, 0, 0, 0.0, 0.0)
    assert math.isclose(v[1], 1.0, abs_tol=1e-9)


def test_sat_azel_range():
    rx = ecef.geodetic_to_ecef(55.0, 20.0, 0.0)
    sat = ecef.sat_ecef_from_azel(55.0, 20.0, 45.0, 30.0)
    d = math.dist(sat, rx)
    assert math.isclose(d, ecef.GPS_RANGE_M, rel_tol=1e-6)


def test_sat_zenith_is_up():
    # elevation 90 -> satellite straight up, farther from Earth centre
    sat = ecef.sat_ecef_from_azel(55.0, 20.0, 0.0, 90.0)
    rx = ecef.geodetic_to_ecef(55.0, 20.0, 0.0)
    assert linalg_norm(sat) > linalg_norm(rx)


def test_sat_positions_distinct():
    a = ecef.sat_ecef_from_azel(55.0, 20.0, 10.0, 40.0)
    b = ecef.sat_ecef_from_azel(55.0, 20.0, 200.0, 40.0)
    assert math.dist(a, b) > 1000.0


def linalg_norm(v):
    return math.sqrt(sum(c * c for c in v))
