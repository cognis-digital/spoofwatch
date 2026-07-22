from spoofwatch import detect, jamming, spoofing, synth
from spoofwatch.records import Report, haversine_km


def test_jamming_finds_planted_zone():
    reps, t = synth.generate(seed=7)
    zones = jamming.detect(reps)
    assert zones, "expected a jamming zone"
    jz_lat, jz_lon, jz_r = t["jamming_zones"][0]
    err = min(haversine_km(jz_lat, jz_lon, z["center_lat"], z["center_lon"]) for z in zones)
    assert err <= jz_r + 25


def test_jamming_clean_airspace_none():
    reps, _ = synth.generate(seed=7, with_jamming=False, with_spoof=False)
    assert jamming.detect(reps) == []


def test_jamming_confidence_scales():
    reps, _ = synth.generate(seed=2)
    zones = jamming.detect(reps)
    assert all(0.0 <= z["confidence"] <= 1.0 for z in zones)
    assert zones[0]["degraded_reports"] >= jamming.MIN_SAMPLES


def test_spoofing_teleport_detected():
    reps = [Report("X", 0, 54.0, 20.0, 9.0), Report("X", 60, 57.0, 25.0, 9.0),
            Report("X", 120, 57.06, 25.0, 9.0)]
    ev = spoofing.detect(reps)
    assert any(e["type"] == "teleport" and "X" in e["ids"] for e in ev)


def test_spoofing_colocation_detected():
    # 6 distinct aircraft all reporting the same point at the same time
    reps = [Report(f"AC{i}", 100.0, 55.0, 22.0, 9.0) for i in range(6)]
    ev = spoofing.detect(reps)
    colo = [e for e in ev if e["type"] == "colocation"]
    assert colo and colo[0]["n"] >= 4


def test_spoofing_normal_traffic_clean():
    reps, _ = synth.generate(seed=4, with_jamming=False, with_spoof=False)
    ev = spoofing.detect(reps)
    # normal traffic should produce no (or negligible) spoof events
    assert len(ev) == 0


def test_analyze_structure():
    reps, _ = synth.generate(seed=7)
    res = detect.analyze(reps)
    assert res["reports"] == len(reps)
    assert "jamming_zones" in res and "spoof_events" in res
    assert res["summary"]["jamming_zones"] >= 1
    assert res["summary"]["aircraft_spoofed"] >= 1
    # spoof event ids are JSON-serializable (lists, not sets)
    for e in res["spoof_events"]:
        assert isinstance(e["ids"], list)
