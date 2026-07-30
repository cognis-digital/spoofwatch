from spoofwatch import codecarrier
from spoofwatch.codecarrier import RangeSample


def _clean_track(sat="G01", n=10, base=2.0e7, drift=0.0):
    # code and phase both advance together; divergence stays flat
    out = []
    for t in range(n):
        rng = base + 100.0 * t
        out.append(RangeSample(t, sat, code_m=rng, phase_m=rng - 1234.5 + drift * t))
    return out


def test_by_sat_groups_and_sorts():
    samples = [RangeSample(2, "A", 1, 0), RangeSample(0, "A", 1, 0),
               RangeSample(1, "B", 1, 0)]
    tracks = codecarrier.by_sat(samples)
    assert set(tracks) == {"A", "B"}
    assert [s.ts for s in tracks["A"]] == [0, 2]


def test_divergence_starts_at_zero():
    series = codecarrier.divergence_series(_clean_track())
    assert series[0][1] == 0.0


def test_divergence_flat_when_locked():
    series = codecarrier.divergence_series(_clean_track())
    assert all(abs(d) < 1e-6 for _, d in series)


def test_divergence_empty():
    assert codecarrier.divergence_series([]) == []


def test_divergence_ambiguity_removed():
    # a large constant phase offset (integer ambiguity) must cancel
    t1 = _clean_track()
    for s in t1:
        s.phase_m += 1.0e6
    series = codecarrier.divergence_series(t1)
    assert all(abs(d) < 1e-6 for _, d in series)


def test_clean_not_flagged():
    r = codecarrier.check(_clean_track())
    assert r["diverges"] is False
    assert r["flagged"] == []
    assert r["confidence"] == 0.0


def test_diverging_sat_flagged():
    # code walks away from carrier at 3 m/s -> exceeds 1 m/s bound
    track = []
    for t in range(10):
        rng = 2.0e7 + 100.0 * t
        track.append(RangeSample(t, "G02", code_m=rng + 3.0 * t, phase_m=rng))
    r = codecarrier.check(track)
    assert "G02" in r["flagged"]
    assert r["diverges"] is True
    assert r["confidence"] > 0.0


def test_mixed_flags_only_bad():
    good = _clean_track("G01")
    bad = []
    for t in range(10):
        rng = 2.0e7 + 100.0 * t
        bad.append(RangeSample(t, "G09", code_m=rng + 5.0 * t, phase_m=rng))
    r = codecarrier.check(good + bad)
    assert r["flagged"] == ["G09"]


def test_short_track_skipped():
    track = [RangeSample(0, "G01", 1, 0), RangeSample(1, "G01", 2, 1)]
    r = codecarrier.check(track)
    assert r["per_satellite"] == []


def test_confidence_bounded():
    track = []
    for t in range(10):
        rng = 2.0e7 + 100.0 * t
        track.append(RangeSample(t, "G02", code_m=rng + 50.0 * t, phase_m=rng))
    r = codecarrier.check(track)
    assert 0.0 <= r["confidence"] <= 1.0


def test_per_satellite_fields():
    r = codecarrier.check(_clean_track())
    row = r["per_satellite"][0]
    for key in ("sat_id", "epochs", "max_rate_m_s", "total_divergence_m", "diverges"):
        assert key in row


def test_max_rate_reported():
    track = []
    for t in range(6):
        rng = 2.0e7
        # jump the code at one epoch
        code = rng + (200.0 if t == 3 else 0.0)
        track.append(RangeSample(t, "G05", code_m=code, phase_m=rng))
    r = codecarrier.check(track)
    assert r["max_rate_m_s"] >= 200.0


def test_threshold_configurable():
    track = []
    for t in range(10):
        rng = 2.0e7
        track.append(RangeSample(t, "G02", code_m=rng + 0.5 * t, phase_m=rng))
    lax = codecarrier.check(track, rate_thresh=1.0)
    strict = codecarrier.check(track, rate_thresh=0.2)
    assert lax["diverges"] is False
    assert strict["diverges"] is True


def test_zero_dt_ignored():
    track = [RangeSample(0, "G01", 2.0e7, 2.0e7),
             RangeSample(0, "G01", 2.0e7 + 500, 2.0e7),   # duplicate ts
             RangeSample(1, "G01", 2.0e7, 2.0e7)]
    r = codecarrier.check(track)
    # must not raise; sat present
    assert any(row["sat_id"] == "G01" for row in r["per_satellite"])


def test_n_flagged_count():
    tracks = []
    for sat in ("A", "B"):
        for t in range(8):
            rng = 2.0e7
            tracks.append(RangeSample(t, sat, code_m=rng + 4.0 * t, phase_m=rng))
    r = codecarrier.check(tracks)
    assert r["n_flagged"] == 2


def test_hatch_first_equals_code():
    track = _clean_track()
    sm = codecarrier.hatch_smooth(track)
    assert sm[0][1] == track[0].code_m


def test_hatch_length():
    track = _clean_track(n=12)
    sm = codecarrier.hatch_smooth(track)
    assert len(sm) == 12


def test_hatch_empty():
    assert codecarrier.hatch_smooth([]) == []


def test_hatch_suppresses_code_noise():
    # phase is clean-linear; code carries alternating multipath noise
    track = []
    true_rng = 2.0e7
    for t in range(60):
        noise = 20.0 if t % 2 == 0 else -20.0
        track.append(RangeSample(t, "G01", code_m=true_rng + noise, phase_m=true_rng))
    sm = codecarrier.hatch_smooth(track, window=100)
    # late smoothed values should be far closer to the truth than the raw code swing
    late_err = abs(sm[-1][1] - true_rng)
    assert late_err < 5.0


def test_hatch_tracks_phase_motion():
    # constant velocity: phase advances 100 m/epoch, code matches
    track = []
    for t in range(30):
        rng = 2.0e7 + 100.0 * t
        track.append(RangeSample(t, "G01", code_m=rng, phase_m=rng))
    sm = codecarrier.hatch_smooth(track)
    assert abs(sm[-1][1] - track[-1].code_m) < 1e-3


def test_hatch_window_ramp():
    # a single early code error should decay as the window ramps
    track = []
    for t in range(20):
        rng = 2.0e7
        code = rng + (100.0 if t == 1 else 0.0)
        track.append(RangeSample(t, "G01", code_m=code, phase_m=rng))
    sm = codecarrier.hatch_smooth(track, window=100)
    assert abs(sm[-1][1] - 2.0e7) < abs(sm[1][1] - 2.0e7)
