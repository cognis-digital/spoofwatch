from spoofwatch import noisefloor
from spoofwatch.noisefloor import NoiseSample


def _noise(seq):
    # seq: list of (ts, noise_dbm)
    return [NoiseSample(ts, noise_dbm=v) for ts, v in seq]


def test_level_from_noise():
    assert NoiseSample(0, noise_dbm=-100).level() == -100


def test_level_from_agc_inverts():
    # higher AGC gain -> lower interference level
    assert NoiseSample(0, agc_db=40).level() == -40


def test_level_requires_a_field():
    try:
        NoiseSample(0).level()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_jn_ratio():
    assert noisefloor.jn_ratio(-90, -100) == 10


def test_empty_series():
    r = noisefloor.monitor([])
    assert r["states"] == []
    assert r["events"] == []
    assert r["jamming_now"] is False


def test_quiet_series_no_events():
    r = noisefloor.monitor(_noise([(t, -100 + (t % 2)) for t in range(10)]))
    assert r["events"] == []
    assert r["jamming_now"] is False


def test_jamming_burst_detected():
    seq = [(t, -100) for t in range(5)] + [(t, -88) for t in range(5, 10)] + \
          [(t, -100) for t in range(10, 15)]
    r = noisefloor.monitor(_noise(seq))
    assert len(r["events"]) == 1
    ev = r["events"][0]
    assert ev.start_ts == 5
    assert ev.peak_jn_db >= 6.0
    assert ev.ongoing is False


def test_event_duration():
    seq = [(t, -100) for t in range(5)] + [(t, -85) for t in range(5, 10)] + \
          [(t, -100) for t in range(10, 13)]
    r = noisefloor.monitor(_noise(seq))
    ev = r["events"][0]
    assert ev.duration_s > 0


def test_hysteresis_no_chatter():
    # after opening, level hovers between off and on thresholds -> stays one event
    seq = [(t, -100) for t in range(5)]                          # clean baseline seed
    seq += [(5, -92), (6, -96), (7, -92), (8, -96), (9, -92)]    # 4-8 dB J/N, never < off
    seq += [(t, -100) for t in range(10, 13)]
    r = noisefloor.monitor(_noise(seq), on_db=6.0, off_db=3.0)
    assert len(r["events"]) == 1


def test_ongoing_event_at_end():
    seq = [(t, -100) for t in range(5)] + [(t, -85) for t in range(5, 10)]
    r = noisefloor.monitor(_noise(seq))
    assert r["jamming_now"] is True
    assert r["events"][-1].ongoing is True
    assert r["events"][-1].end_ts is None


def test_states_length_matches():
    seq = [(t, -100) for t in range(8)]
    r = noisefloor.monitor(_noise(seq))
    assert len(r["states"]) == 8


def test_baseline_adapts_to_slow_drift():
    # slow ramp of the quiet floor should be tracked, not flagged
    seq = [(t, -100 + t * 0.2) for t in range(30)]
    r = noisefloor.monitor(_noise(seq))
    assert r["events"] == []


def test_agc_drop_detected():
    # AGC gain drops sharply -> interference level rises
    seq = [NoiseSample(t, agc_db=40) for t in range(5)]
    seq += [NoiseSample(t, agc_db=28) for t in range(5, 10)]
    seq += [NoiseSample(t, agc_db=40) for t in range(10, 14)]
    r = noisefloor.monitor(seq)
    assert len(r["events"]) == 1


def test_unsorted_input_sorted():
    seq = _noise([(2, -100), (0, -100), (1, -100)])
    r = noisefloor.monitor(seq)
    assert [s.ts for s in r["states"]] == [0, 1, 2]


def test_summarize_counts():
    seq = [(t, -100) for t in range(5)] + [(t, -85) for t in range(5, 10)] + \
          [(t, -100) for t in range(10, 15)]
    s = noisefloor.summarize(_noise(seq))
    assert s["jam_events"] == 1
    assert s["peak_jn_db"] >= 6.0
    assert 0.0 <= s["confidence"] <= 1.0


def test_summarize_quiet():
    s = noisefloor.summarize(_noise([(t, -100) for t in range(10)]))
    assert s["jam_events"] == 0
    assert s["confidence"] == 0.0
    assert s["jamming_now"] is False


def test_peak_jn_reflects_worst():
    seq = [(t, -100) for t in range(5)] + [(5, -80)] + [(t, -100) for t in range(6, 9)]
    r = noisefloor.monitor(_noise(seq))
    assert r["events"][0].peak_jn_db >= 18.0


def test_two_separate_events():
    seq = ([(t, -100) for t in range(5)] + [(5, -85), (6, -85)] +
           [(t, -100) for t in range(7, 11)] + [(11, -85), (12, -85)] +
           [(t, -100) for t in range(13, 16)])
    r = noisefloor.monitor(_noise(seq))
    assert len(r["events"]) == 2


def test_confidence_scales_with_severity():
    mild = noisefloor.summarize(
        _noise([(t, -100) for t in range(5)] + [(5, -92), (6, -92)] +
               [(t, -100) for t in range(7, 10)]))
    severe = noisefloor.summarize(
        _noise([(t, -100) for t in range(5)] + [(5, -75), (6, -75)] +
               [(t, -100) for t in range(7, 10)]))
    assert severe["confidence"] >= mild["confidence"]


def test_states_carry_jamming_flag():
    seq = [(t, -100) for t in range(5)] + [(5, -85), (6, -85)] + \
          [(t, -100) for t in range(7, 10)]
    r = noisefloor.monitor(_noise(seq))
    assert any(st.jamming for st in r["states"])
    assert any(not st.jamming for st in r["states"])
