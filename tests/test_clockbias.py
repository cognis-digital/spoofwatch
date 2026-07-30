from spoofwatch import clockbias


def _s(ts, bias, drift=None):
    return clockbias.ClockSample(ts, bias, drift)


def test_stable_clock_no_events():
    samples = [_s(t, t * 2.0, 2.0) for t in range(10)]   # steady 2 ns/s drift
    assert clockbias.detect(samples) == []


def test_step_jump_detected():
    samples = [_s(0, 0), _s(1, 2), _s(2, 500)]
    ev = clockbias.detect(samples)
    assert any(e["type"] == "time_step" for e in ev)
    step = [e for e in ev if e["type"] == "time_step"][0]
    assert step["jump_ns"] > clockbias.STEP_JUMP_NS


def test_negative_jump_detected():
    samples = [_s(0, 0), _s(1, 0), _s(2, -400)]
    ev = clockbias.detect(samples)
    assert any(e["type"] == "time_step" for e in ev)


def test_drift_anomaly_detected():
    # implied drift 60 ns/s (< step threshold over 1s? 60<100 so no step) but > drift bound
    samples = [_s(0, 0), _s(1, 60)]
    ev = clockbias.detect(samples, step_jump_ns=100.0, max_drift_ns_s=50.0)
    assert any(e["type"] == "drift_anomaly" for e in ev)


def test_drift_predicted_step_ignored():
    # bias moves by exactly the declared (physical) drift over dt -> no event
    samples = [_s(0, 0, drift=40.0), _s(1, 40), _s(2, 80)]
    assert clockbias.detect(samples) == []


def test_confidence_bounded():
    samples = [_s(0, 0), _s(1, 100000)]
    ev = clockbias.detect(samples)
    assert all(0.0 <= e["confidence"] <= 1.0 for e in ev)


def test_zero_dt_skipped():
    samples = [_s(1, 0), _s(1, 500)]
    assert clockbias.detect(samples) == []


def test_unsorted_input_sorted():
    samples = [_s(2, 500), _s(0, 0), _s(1, 2)]
    ev = clockbias.detect(samples)
    assert ev and ev[0]["ts_from"] == 1 and ev[0]["ts_to"] == 2


def test_summarize():
    samples = [_s(0, 0), _s(1, 2), _s(2, 500), _s(3, 502)]
    s = clockbias.summarize(samples)
    assert s["samples"] == 4
    assert s["time_steps"] >= 1
    assert s["max_jump_ns"] > 0


def test_summarize_clean():
    samples = [_s(t, t * 2.0, 2.0) for t in range(5)]
    s = clockbias.summarize(samples)
    assert s["time_steps"] == 0
    assert s["drift_anomalies"] == 0
    assert s["max_jump_ns"] == 0.0
