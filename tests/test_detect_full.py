from spoofwatch import detect, meaconing, osnma, synth
from spoofwatch.records import Report


def test_analyze_full_backward_compatible():
    reps, _ = synth.generate(seed=7)
    res = detect.analyze_full(reps)
    # still has everything analyze() has
    assert res["reports"] == len(reps)
    assert "jamming_zones" in res and "spoof_events" in res
    assert res["summary"]["jamming_zones"] >= 1


def test_analyze_full_matches_analyze_base_fields():
    reps, _ = synth.generate(seed=3)
    base = detect.analyze(reps)
    full = detect.analyze_full(reps)
    assert full["reports"] == base["reports"]
    assert full["summary"]["jamming_zones"] == base["summary"]["jamming_zones"]


def test_analyze_full_osnma_elevation():
    # a teleport track whose id also fails OSNMA -> elevated
    reps = [Report("X", 0, 54.0, 20.0, 9.0), Report("X", 60, 57.0, 25.0, 9.0),
            Report("X", 120, 57.06, 25.0, 9.0)]
    osn = [osnma.OsnmaReport("X", 60, "auth-failed")]
    res = detect.analyze_full(reps, osnma_reports=osn)
    tele = [e for e in res["spoof_events"] if e["type"] == "teleport"]
    assert tele and tele[0]["confidence"] >= osnma.CORROBORATED_CONFIDENCE
    assert res["summary"]["osnma_auth_fails"] >= 1


def test_analyze_full_meaconing():
    reps, _ = synth.generate(seed=7)
    navs = [meaconing.NavObservation(f"AC{i}", 100, 55.0, 22.0, time_offset_s=2.0 + i * 0.05)
            for i in range(6)]
    res = detect.analyze_full(reps, nav_observations=navs)
    assert res["summary"]["meaconing_events"] >= 1
    assert any(e["type"] == "meaconing" for e in res["meaconing_events"])


def test_analyze_full_json_serializable_ids():
    reps, _ = synth.generate(seed=7)
    osn = [osnma.OsnmaReport("AC000", 1, "auth-failed")]
    res = detect.analyze_full(reps, osnma_reports=osn)
    for e in res["spoof_events"]:
        assert isinstance(e["ids"], list)
    for e in res["osnma_events"]:
        assert isinstance(e["ids"], list)


def test_analyze_unchanged_no_new_keys_leak():
    # the original analyze() must NOT have the new keys (additive, not mutated)
    reps, _ = synth.generate(seed=7)
    res = detect.analyze(reps)
    assert "osnma_events" not in res
    assert "meaconing_events" not in res
