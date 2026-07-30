from spoofwatch import fusion


def _d(sid, lat, lon, r=20.0, conf=0.5, kind="jamming"):
    return fusion.SensorDetection(sid, lat, lon, r, conf, kind)


def test_empty():
    assert fusion.fuse([]) == []


def test_single_sensor_not_confirmed():
    z = fusion.fuse([_d("s1", 55.0, 20.0)])
    assert len(z) == 1
    assert z[0]["confirmed"] is False
    assert z[0]["sensor_count"] == 1


def test_two_sensors_confirmed():
    z = fusion.fuse([_d("s1", 55.0, 20.0, conf=0.6), _d("s2", 55.1, 20.1, conf=0.7)])
    assert len(z) == 1
    assert z[0]["confirmed"] is True
    assert z[0]["sensor_count"] == 2


def test_far_detections_separate():
    z = fusion.fuse([_d("s1", 55.0, 20.0), _d("s2", 10.0, 10.0)])
    assert len(z) == 2


def test_confirmed_ranks_first():
    dets = [_d("s1", 10.0, 10.0, conf=0.95),         # single, high conf, isolated
            _d("s2", 55.0, 20.0, conf=0.5), _d("s3", 55.1, 20.1, conf=0.5)]  # corroborated
    z = fusion.fuse(dets)
    assert z[0]["confirmed"] is True
    assert z[0]["sensor_count"] == 2


def test_combined_confidence_exceeds_individual():
    z = fusion.fuse([_d("s1", 55.0, 20.0, conf=0.5), _d("s2", 55.05, 20.05, conf=0.5)])
    assert z[0]["confidence"] > 0.5    # independent-evidence OR


def test_confidence_bounded():
    z = fusion.fuse([_d(f"s{i}", 55.0, 20.0, conf=0.9) for i in range(5)])
    assert all(0.0 <= x["confidence"] <= 1.0 for x in z)


def test_same_sensor_not_double_counted():
    z = fusion.fuse([_d("s1", 55.0, 20.0), _d("s1", 55.05, 20.05)])
    assert z[0]["sensor_count"] == 1
    assert z[0]["confirmed"] is False


def test_weighted_centroid():
    # high-confidence detection pulls centroid toward it
    z = fusion.fuse([_d("s1", 55.0, 20.0, conf=0.9), _d("s2", 55.2, 20.0, conf=0.1)])
    assert z[0]["center_lat"] < 55.1


def test_radius_is_max():
    z = fusion.fuse([_d("s1", 55.0, 20.0, r=20.0), _d("s2", 55.05, 20.05, r=45.0)])
    assert z[0]["radius_km"] == 45.0


def test_sensors_listed():
    z = fusion.fuse([_d("alpha", 55.0, 20.0), _d("bravo", 55.05, 20.05)])
    assert z[0]["sensors"] == ["alpha", "bravo"]
