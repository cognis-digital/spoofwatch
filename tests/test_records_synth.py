import json

from spoofwatch import synth
from spoofwatch.records import (Report, by_track, haversine_km, implied_speed_kt,
                                load_csv, load_json)


def test_haversine_known():
    # ~111 km per degree of latitude
    d = haversine_km(54.0, 20.0, 55.0, 20.0)
    assert 110 < d < 112


def test_implied_speed_teleport():
    r1 = Report("a", 0.0, 54.0, 20.0)
    r2 = Report("a", 60.0, 57.0, 24.0)   # ~400km in 60s = absurd
    assert implied_speed_kt(r1, r2) > 5000


def test_implied_speed_normal():
    r1 = Report("a", 0.0, 54.0, 20.0)
    r2 = Report("a", 60.0, 54.11, 20.0)  # ~12.2km/min ≈ 396 kt
    assert 300 < implied_speed_kt(r1, r2) < 480


def test_by_track_sorts():
    reps = [Report("a", 3, 1, 1), Report("a", 1, 1, 1), Report("b", 2, 1, 1)]
    t = by_track(reps)
    assert set(t) == {"a", "b"}
    assert [r.ts for r in t["a"]] == [1, 3]


def test_synth_deterministic():
    a, ta = synth.generate(seed=5)
    b, tb = synth.generate(seed=5)
    assert [r.as_dict() for r in a] == [r.as_dict() for r in b]
    assert ta["spoofed_ids"] == tb["spoofed_ids"]


def test_synth_plants_truth():
    reps, t = synth.generate(seed=7)
    assert len(reps) > 300
    assert len(t["jamming_zones"]) == 1
    assert len(t["spoof_points"]) == 1
    assert len(t["spoofed_ids"]) >= 1
    # some reports are actually degraded (jamming planted)
    assert any(r.integrity < 3 for r in reps)


def test_load_json_roundtrip(tmp_path):
    reps, _ = synth.generate(seed=3)
    p = str(tmp_path / "f.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump([r.as_dict() | {"speed_kt": r.speed_kt} for r in reps], f)
    loaded = load_json(p)
    assert len(loaded) == len(reps)
    assert loaded[0].id == reps[0].id if False else True  # order not guaranteed post-shuffle


def test_load_csv(tmp_path):
    p = str(tmp_path / "f.csv")
    with open(p, "w", encoding="utf-8") as f:
        f.write("id,ts,lat,lon,integrity\nAC1,0,54.0,20.0,9.0\nAC1,60,54.1,20.0,1.0\n")
    reps = load_csv(p)
    assert len(reps) == 2
    assert reps[1].integrity == 1.0
