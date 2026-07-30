from spoofwatch import aoa


def _sky():
    # a well-spread constellation across the sky
    return [aoa.DoAObservation("a", 10, 60), aoa.DoAObservation("b", 120, 40),
            aoa.DoAObservation("c", 250, 55), aoa.DoAObservation("d", 300, 25),
            aoa.DoAObservation("e", 180, 70)]


def _single_source():
    # every "satellite" from nearly one direction (a ground spoofer)
    return [aoa.DoAObservation(f"s{i}", 45 + i * 0.4, 30 + i * 0.2) for i in range(6)]


def test_sky_spread_large():
    assert aoa.angular_spread_deg(_sky()) > aoa.MIN_ANGULAR_SPREAD_DEG


def test_single_source_spread_small():
    assert aoa.angular_spread_deg(_single_source()) < aoa.MIN_ANGULAR_SPREAD_DEG


def test_sky_not_single_source():
    r = aoa.check(_sky())
    assert r["available"] is True
    assert r["single_source"] is False
    assert r["confidence"] == 0.0


def test_single_source_flagged():
    r = aoa.check(_single_source())
    assert r["available"] is True
    assert r["single_source"] is True
    assert r["confidence"] > 0.0


def test_too_few_sats_unavailable():
    r = aoa.check([aoa.DoAObservation("a", 10, 60), aoa.DoAObservation("b", 120, 40)])
    assert r["available"] is False
    assert r["single_source"] is False


def test_confidence_bounded():
    r = aoa.check(_single_source())
    assert 0.0 <= r["confidence"] <= 1.0


def test_identical_directions_zero_spread():
    obs = [aoa.DoAObservation(f"s{i}", 45.0, 30.0) for i in range(5)]
    assert aoa.angular_spread_deg(obs) < 1e-6


def test_empty_spread_zero():
    assert aoa.angular_spread_deg([]) == 0.0


def test_n_sats_reported():
    r = aoa.check(_sky())
    assert r["n_sats"] == 5


def test_threshold_configurable():
    obs = _single_source()
    # with a very small threshold even a single source won't trip
    r = aoa.check(obs, min_spread_deg=0.001)
    assert r["single_source"] is False
