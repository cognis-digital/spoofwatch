from spoofwatch import constellation as C


def _sub(name, lat, lon, clock=None):
    return C.SubSolution(name, lat, lon, clock)


def test_agreeing_constellations_no_divergence():
    subs = [_sub("GPS", 55.0, 20.0), _sub("GAL", 55.001, 20.001),
            _sub("GLO", 54.999, 19.999)]
    r = C.cross_check(subs)
    assert r["divergence"] is False
    assert r["flagged"] == []


def test_single_constellation_no_check():
    r = C.cross_check([_sub("GPS", 55.0, 20.0)])
    assert r["divergence"] is False
    assert "note" in r


def test_one_diverging_constellation_flagged():
    subs = [_sub("GPS", 55.0, 20.0), _sub("GAL", 55.001, 20.0),
            _sub("GLO", 55.5, 20.5)]      # GLO way off
    r = C.cross_check(subs)
    assert r["divergence"] is True
    assert "GLO" in r["flagged"]


def test_robust_reference_not_dragged():
    # 3 agree, 1 spoofed far away; the spoofed one must be the flagged outlier
    subs = [_sub("GPS", 55.0, 20.0), _sub("GAL", 55.0, 20.0),
            _sub("GLO", 55.0, 20.0), _sub("BDS", 56.0, 21.0)]
    r = C.cross_check(subs)
    assert r["flagged"] == ["BDS"]


def test_time_divergence_flagged():
    subs = [_sub("GPS", 55.0, 20.0, clock=10.0),
            _sub("GAL", 55.0, 20.0, clock=12.0),
            _sub("GLO", 55.0, 20.0, clock=800.0)]
    r = C.cross_check(subs)
    assert "GLO" in r["time_flagged"]
    assert r["divergence"] is True


def test_two_constellations_agree():
    subs = [_sub("GPS", 55.0, 20.0), _sub("GAL", 55.0005, 20.0005)]
    r = C.cross_check(subs)
    assert r["divergence"] is False


def test_two_constellations_disagree():
    subs = [_sub("GPS", 55.0, 20.0), _sub("GAL", 55.3, 20.3)]
    r = C.cross_check(subs)
    assert r["divergence"] is True


def test_confidence_bounded():
    subs = [_sub("GPS", 55.0, 20.0), _sub("GAL", 55.0, 20.0),
            _sub("GLO", 60.0, 25.0)]
    r = C.cross_check(subs)
    assert 0.0 <= r["confidence"] <= 1.0


def test_max_divergence_reported():
    subs = [_sub("GPS", 55.0, 20.0), _sub("GAL", 55.0, 20.0),
            _sub("GLO", 55.1, 20.0)]
    r = C.cross_check(subs)
    assert r["max_divergence_km"] > 0


def test_per_constellation_entries():
    subs = [_sub("GPS", 55.0, 20.0), _sub("GAL", 55.0, 20.0)]
    r = C.cross_check(subs)
    names = {e["constellation"] for e in r["per_constellation"]}
    assert names == {"GPS", "GAL"}
