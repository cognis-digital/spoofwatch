import math

from spoofwatch import doppler
from spoofwatch.ecef import geodetic_to_ecef, sat_ecef_from_azel


def _obs(sat_id, sat_pos, sat_vel, measured, const="GPS"):
    return doppler.DopplerObs(sat_id, sat_pos, sat_vel, measured, const)


def _genuine_scene(rx_lat=59.0, rx_lon=18.0, rx_vel=(0.0, 0.0, 0.0)):
    """Build a genuine multi-satellite scene where measured == predicted."""
    rx = geodetic_to_ecef(rx_lat, rx_lon, 0.0)
    sats = []
    layout = [(45, 30, 1000.0), (135, 50, -800.0), (225, 20, 400.0),
              (315, 70, -1200.0), (90, 40, 200.0)]
    for i, (az, el, speed) in enumerate(layout):
        pos = sat_ecef_from_azel(rx_lat, rx_lon, az, el)
        # give the satellite a velocity along an arbitrary but fixed direction
        d = doppler._sub(pos, rx)
        n = doppler._norm(d)
        u = (d[0] / n, d[1] / n, d[2] / n)
        # velocity partly along LOS -> nonzero range rate
        vel = (u[1] * speed, -u[0] * speed, u[2] * speed * 0.3)
        pred = doppler.expected_doppler_hz(rx, rx_vel, pos, vel)
        sats.append(_obs(f"G{i}", pos, vel, pred))
    return rx, rx_vel, sats


# --------------------------------------------------------------------------- #
# helpers / physics
# --------------------------------------------------------------------------- #
def test_los_unit_is_normalized():
    u = doppler.los_unit((0, 0, 0), (100, 0, 0))
    assert u == (1.0, 0.0, 0.0)


def test_los_unit_zero_distance():
    assert doppler.los_unit((5, 5, 5), (5, 5, 5)) == (0.0, 0.0, 0.0)


def test_range_rate_receding_positive():
    # satellite on +x, moving further out -> positive range rate
    rr = doppler.range_rate((0, 0, 0), (0, 0, 0), (100, 0, 0), (10, 0, 0))
    assert rr > 0


def test_range_rate_approaching_negative():
    rr = doppler.range_rate((0, 0, 0), (0, 0, 0), (100, 0, 0), (-10, 0, 0))
    assert rr < 0


def test_receiver_velocity_affects_range_rate():
    rr_static = doppler.range_rate((0, 0, 0), (0, 0, 0), (100, 0, 0), (0, 0, 0))
    rr_moving = doppler.range_rate((0, 0, 0), (5, 0, 0), (100, 0, 0), (0, 0, 0))
    assert rr_static == 0.0
    assert rr_moving < 0    # receiver approaching sat


def test_approaching_gives_positive_doppler():
    fd = doppler.expected_doppler_hz((0, 0, 0), (0, 0, 0), (100, 0, 0), (-10, 0, 0))
    assert fd > 0


def test_receding_gives_negative_doppler():
    fd = doppler.expected_doppler_hz((0, 0, 0), (0, 0, 0), (100, 0, 0), (10, 0, 0))
    assert fd < 0


def test_doppler_magnitude_reasonable():
    # 800 m/s radial ~ 4.2 kHz at L1
    fd = doppler.expected_doppler_hz((0, 0, 0), (0, 0, 0), (100, 0, 0), (-800, 0, 0))
    assert 3500 < fd < 4500


def test_doppler_scales_with_frequency():
    fd1 = doppler.expected_doppler_hz((0, 0, 0), (0, 0, 0), (100, 0, 0), (-800, 0, 0),
                                      freq_hz=doppler.GPS_L1_HZ)
    fd2 = doppler.expected_doppler_hz((0, 0, 0), (0, 0, 0), (100, 0, 0), (-800, 0, 0),
                                      freq_hz=doppler.GPS_L1_HZ / 2)
    assert abs(fd1) > abs(fd2)


# --------------------------------------------------------------------------- #
# check()
# --------------------------------------------------------------------------- #
def test_too_few_sats_unavailable():
    rx = geodetic_to_ecef(59, 18)
    obs = [_obs("G0", (rx[0] + 1e7, rx[1], rx[2]), (0, 0, 0), 0.0)]
    r = doppler.check(obs, rx)
    assert r.available is False
    assert r.n_sats == 1


def test_genuine_scene_consistent():
    rx, rx_vel, sats = _genuine_scene()
    r = doppler.check(sats, rx, rx_vel)
    assert r.available is True
    assert r.inconsistent is False
    assert r.residual_rms_hz < doppler.RESIDUAL_THRESH_HZ
    assert r.confidence == 0.0


def test_genuine_scene_residuals_near_zero():
    rx, rx_vel, sats = _genuine_scene()
    r = doppler.check(sats, rx, rx_vel)
    assert r.max_residual_hz < 1.0


def test_per_sat_entries_present():
    rx, rx_vel, sats = _genuine_scene()
    r = doppler.check(sats, rx, rx_vel)
    assert len(r.per_sat) == len(sats)
    assert set(r.per_sat[0].keys()) == {"sat_id", "predicted_hz", "measured_hz",
                                        "residual_hz"}


def test_biased_doppler_is_inconsistent():
    rx, rx_vel, sats = _genuine_scene()
    # corrupt every measured Doppler by a big offset (spoofer with wrong dynamics)
    bad = [doppler.DopplerObs(o.sat_id, o.sat_pos, o.sat_vel, o.measured_hz + 300.0)
           for o in sats]
    r = doppler.check(bad, rx, rx_vel)
    assert r.inconsistent is True
    assert r.confidence > 0.0
    assert r.residual_rms_hz > doppler.RESIDUAL_THRESH_HZ


def test_static_spoofer_signature():
    rx, rx_vel, sats = _genuine_scene()
    # a fixed-antenna spoofer: all measured Dopplers collapse to ~0
    spoofed = [doppler.DopplerObs(o.sat_id, o.sat_pos, o.sat_vel, 0.0) for o in sats]
    r = doppler.check(spoofed, rx, rx_vel)
    assert r.static_spoofer is True
    assert r.confidence >= 0.6


def test_static_spoofer_needs_predicted_spread():
    # if genuine predicted Dopplers are all near zero (sats not moving), no static flag
    rx = geodetic_to_ecef(59, 18)
    sats = []
    for i, az in enumerate([0, 90, 180, 270]):
        pos = sat_ecef_from_azel(59, 18, az, 45)
        sats.append(_obs(f"G{i}", pos, (0.0, 0.0, 0.0), 0.0))
    r = doppler.check(sats, rx)
    assert r.static_spoofer is False


def test_confidence_bounded():
    rx, rx_vel, sats = _genuine_scene()
    bad = [doppler.DopplerObs(o.sat_id, o.sat_pos, o.sat_vel, o.measured_hz + 5000.0)
           for o in sats]
    r = doppler.check(bad, rx, rx_vel)
    assert 0.0 <= r.confidence <= 1.0


def test_measured_spread_reported():
    rx, rx_vel, sats = _genuine_scene()
    r = doppler.check(sats, rx, rx_vel)
    assert r.measured_spread_hz >= 0.0
    assert r.predicted_spread_hz >= 0.0


def test_custom_threshold_tightens_detection():
    rx, rx_vel, sats = _genuine_scene()
    slightly = [doppler.DopplerObs(o.sat_id, o.sat_pos, o.sat_vel, o.measured_hz + 10.0)
                for o in sats]
    loose = doppler.check(slightly, rx, rx_vel, thresh_hz=25.0)
    tight = doppler.check(slightly, rx, rx_vel, thresh_hz=1.0)
    assert loose.inconsistent is False
    assert tight.inconsistent is True


def test_min_sats_override():
    rx = geodetic_to_ecef(59, 18)
    obs = [_obs("G0", (rx[0] + 1e7, rx[1], rx[2]), (10, 0, 0),
                doppler.expected_doppler_hz(rx, (0, 0, 0),
                                            (rx[0] + 1e7, rx[1], rx[2]), (10, 0, 0)))]
    r = doppler.check(obs, rx, min_sats=1)
    assert r.available is True


def test_result_is_dataclass():
    rx, rx_vel, sats = _genuine_scene()
    r = doppler.check(sats, rx, rx_vel)
    assert isinstance(r, doppler.DopplerResult)


def test_speed_of_light_constant():
    assert abs(doppler.SPEED_OF_LIGHT - 299792458.0) < 1e-3


def test_receiver_motion_consistency():
    # a moving receiver with correctly-modelled velocity stays consistent
    rx_lat, rx_lon = 59.0, 18.0
    rx = geodetic_to_ecef(rx_lat, rx_lon)
    rx_vel = (120.0, -40.0, 0.0)
    sats = []
    for i, (az, el) in enumerate([(30, 25), (120, 60), (210, 35), (300, 45)]):
        pos = sat_ecef_from_azel(rx_lat, rx_lon, az, el)
        vel = (300.0, -200.0, 100.0)
        pred = doppler.expected_doppler_hz(rx, rx_vel, pos, vel)
        sats.append(_obs(f"G{i}", pos, vel, pred))
    r = doppler.check(sats, rx, rx_vel)
    assert r.inconsistent is False


def test_ignoring_receiver_motion_creates_residual():
    # measured with true rx motion, but checked assuming a static receiver -> residual
    rx_lat, rx_lon = 59.0, 18.0
    rx = geodetic_to_ecef(rx_lat, rx_lon)
    rx_vel = (300.0, -300.0, 0.0)
    sats = []
    for i, (az, el) in enumerate([(30, 25), (120, 60), (210, 35), (300, 45)]):
        pos = sat_ecef_from_azel(rx_lat, rx_lon, az, el)
        vel = (300.0, -200.0, 100.0)
        pred_true = doppler.expected_doppler_hz(rx, rx_vel, pos, vel)
        sats.append(_obs(f"G{i}", pos, vel, pred_true))
    r = doppler.check(sats, rx, rx_vel=(0.0, 0.0, 0.0))
    assert r.residual_rms_hz > 0.0


def test_empty_returns_unavailable():
    r = doppler.check([], geodetic_to_ecef(59, 18))
    assert r.available is False
