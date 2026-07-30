from spoofwatch import meaconing as M


def _obs(_id, ts, lat, lon, off=None, msg=None):
    return M.NavObservation(_id, ts, lat, lon, off, msg)


def test_consistent_offset_detected():
    obs = [_obs(f"AC{i}", 100, 55.0, 22.0, off=2.0 + i * 0.05) for i in range(6)]
    ev = M.detect_consistent_offset(obs)
    assert ev and ev[0]["type"] == "meaconing"
    assert ev[0]["n"] >= M.MIN_AIRCRAFT
    assert abs(ev[0]["common_offset_s"] - 2.0) < 1.0


def test_scattered_offsets_not_meaconing():
    obs = [_obs(f"AC{i}", 100, 55.0, 22.0, off=i * 50.0) for i in range(6)]
    ev = M.detect_consistent_offset(obs)
    assert ev == []


def test_too_few_aircraft():
    obs = [_obs(f"AC{i}", 100, 55.0, 22.0, off=2.0) for i in range(2)]
    assert M.detect_consistent_offset(obs) == []


def test_no_offsets_present():
    obs = [_obs(f"AC{i}", 100, 55.0, 22.0) for i in range(6)]
    assert M.detect_consistent_offset(obs) == []


def test_content_replay_detected():
    obs = [_obs("AC0", 100, 55, 22, msg="FRAME_A"),
           _obs("AC1", 160, 55, 22, msg="FRAME_A")]
    ev = M.detect_content_replay(obs)
    assert ev and ev[0]["type"] == "content_replay"
    assert ev[0]["replay_delay_s"] == 60


def test_content_replay_below_min_delay_ignored():
    obs = [_obs("AC0", 100, 55, 22, msg="FRAME_A"),
           _obs("AC1", 102, 55, 22, msg="FRAME_A")]
    assert M.detect_content_replay(obs, min_delay_s=10) == []


def test_content_replay_single_occurrence_ignored():
    obs = [_obs("AC0", 100, 55, 22, msg="FRAME_A")]
    assert M.detect_content_replay(obs) == []


def test_distinct_messages_not_replay():
    obs = [_obs("AC0", 100, 55, 22, msg="A"), _obs("AC1", 200, 55, 22, msg="B")]
    assert M.detect_content_replay(obs) == []


def test_detect_combines_both():
    obs = [_obs(f"AC{i}", 100, 55.0, 22.0, off=2.0 + i * 0.05, msg="F") for i in range(6)]
    obs.append(_obs("AC0", 300, 55.0, 22.0, off=2.0, msg="F"))
    ev = M.detect(obs)
    kinds = {e["type"] for e in ev}
    assert "meaconing" in kinds
    assert "content_replay" in kinds


def test_confidence_bounded():
    obs = [_obs(f"AC{i}", 100, 55.0, 22.0, off=2.0) for i in range(8)]
    ev = M.detect_consistent_offset(obs)
    assert all(0.0 <= e["confidence"] <= 1.0 for e in ev)
