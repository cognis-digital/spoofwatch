from spoofwatch import zones


def _z(lat, lon, r=30.0, deg=10):
    return {"center_lat": lat, "center_lon": lon, "radius_km": r,
            "degraded_reports": deg}


def test_first_zone_appears():
    tr = zones.ZoneTracker()
    alerts = tr.update(0, [_z(55.0, 20.0)])
    assert len(tr.tracks) == 1
    assert alerts[0]["event"] == "appeared"


def test_moving_zone_continues_track():
    tr = zones.ZoneTracker()
    tr.update(0, [_z(55.0, 20.0)])
    tr.update(60, [_z(55.1, 20.1)])   # nearby -> same track
    assert len(tr.tracks) == 1
    assert len(tr.tracks[0].history) == 2


def test_far_zone_new_track():
    tr = zones.ZoneTracker()
    tr.update(0, [_z(55.0, 20.0)])
    tr.update(60, [_z(10.0, 10.0)])   # far -> new track
    assert len(tr.tracks) == 2


def test_centroid_drift_tracked():
    tr = zones.ZoneTracker()
    tr.update(0, [_z(55.0, 20.0)])
    tr.update(60, [_z(55.2, 20.0)])
    assert tr.tracks[0].centroid_drift_km > 10.0


def test_radius_growth_tracked():
    tr = zones.ZoneTracker()
    tr.update(0, [_z(55.0, 20.0, r=20.0)])
    tr.update(60, [_z(55.0, 20.0, r=40.0)])
    assert tr.tracks[0].radius_growth_km == 20.0


def test_dissipation_alert():
    tr = zones.ZoneTracker(stale_epochs=1)
    tr.update(0, [_z(55.0, 20.0)])
    tr.update(60, [])        # epoch 2: age 1 (still live)
    a = tr.update(120, [])   # epoch 3: age 2 == stale+1 -> dissipated
    assert any(x["event"] == "dissipated" for x in a)


def test_active_tracks():
    tr = zones.ZoneTracker(stale_epochs=1)
    tr.update(0, [_z(55.0, 20.0)])
    assert len(tr.active_tracks()) == 1
    tr.update(60, [])
    tr.update(120, [])
    assert len(tr.active_tracks()) == 0


def test_lifetime():
    tr = zones.ZoneTracker()
    tr.update(0, [_z(55.0, 20.0)])
    tr.update(120, [_z(55.05, 20.05)])
    assert tr.tracks[0].lifetime_s == 120


def test_geojson_structure():
    tr = zones.ZoneTracker()
    tr.update(0, [_z(55.0, 20.0)])
    tr.update(60, [_z(55.05, 20.05)])
    gj = tr.to_geojson()
    assert gj["type"] == "FeatureCollection"
    f = gj["features"][0]
    assert f["properties"]["kind"] == "interference_track"
    assert len(f["properties"]["trajectory"]) == 2
    assert f["properties"]["epochs_seen"] == 2


def test_total_reports_accumulate():
    tr = zones.ZoneTracker()
    tr.update(0, [_z(55.0, 20.0, deg=10)])
    tr.update(60, [_z(55.05, 20.05, deg=15)])
    assert tr.tracks[0].total_reports == 25


def test_two_simultaneous_zones():
    tr = zones.ZoneTracker()
    a = tr.update(0, [_z(55.0, 20.0), _z(10.0, 10.0)])
    assert len(tr.tracks) == 2
    assert sum(1 for x in a if x["event"] == "appeared") == 2
