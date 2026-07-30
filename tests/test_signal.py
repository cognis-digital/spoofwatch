from spoofwatch import signal


def _epochs(seq):
    # seq: list of (ts, cn0_list, agc)
    return [signal.SignalEpoch(ts, {f"s{i}": v for i, v in enumerate(cn0)}, agc)
            for ts, cn0, agc in seq]


def test_first_epoch_clean():
    ev = signal.analyze_epochs(_epochs([(0, [40, 41, 39], 100)]))
    assert ev[0].kind == "clean"


def test_uniform_rise_is_spoof():
    ev = signal.analyze_epochs(_epochs([
        (0, [40, 41, 39], 100),
        (1, [41, 40, 40], 100),
        (2, [52, 53, 51], 100),   # +11 dB-Hz, tight spread
    ]))
    assert ev[-1].kind == "spoof"
    assert ev[-1].score > 0.5


def test_non_uniform_rise_not_spoof():
    ev = signal.analyze_epochs(_epochs([
        (0, [40, 41, 39], 100),
        (1, [60, 41, 30], 100),   # big spread -> not the uniform-overpower signature
    ]))
    assert ev[-1].kind != "spoof"


def test_agc_drop_is_jam():
    ev = signal.analyze_epochs(_epochs([
        (0, [40, 41], 100),
        (1, [41, 40], 70),        # 30% AGC drop
    ]))
    assert ev[-1].kind == "jam"
    assert "agc_drop_frac" in ev[-1].detail


def test_cn0_collapse_is_jam():
    ev = signal.analyze_epochs(_epochs([
        (0, [42, 43], 100),
        (1, [30, 31], 100),       # -12 dB-Hz collapse
    ]))
    assert ev[-1].kind == "jam"


def test_clean_stays_clean():
    ev = signal.analyze_epochs(_epochs([
        (0, [40, 41, 39], 100),
        (1, [41, 40, 40], 100),
        (2, [39, 40, 41], 100),
    ]))
    assert all(e.kind == "clean" for e in ev)


def test_scores_bounded():
    ev = signal.analyze_epochs(_epochs([
        (0, [40, 41], 100), (1, [80, 81], 10), (2, [20, 21], 100),
    ]))
    assert all(0.0 <= e.score <= 1.0 for e in ev)


def test_epochs_sorted_by_ts():
    ev = signal.analyze_epochs(_epochs([
        (2, [52, 53, 51], 100),
        (0, [40, 41, 39], 100),
        (1, [41, 40, 40], 100),
    ]))
    assert [e.ts for e in ev] == [0, 1, 2]
    assert ev[-1].kind == "spoof"


def test_summarize_counts():
    ev = signal.analyze_epochs(_epochs([
        (0, [40, 41], 100),
        (1, [52, 53], 100),    # spoof
        (2, [40, 41], 70),     # jam (agc drop)
    ]))
    s = signal.summarize(ev)
    assert s["epochs"] == 3
    assert s["spoof_epochs"] >= 1
    assert s["jam_epochs"] >= 1
    assert 0.0 <= s["peak_spoof_score"] <= 1.0


def test_missing_agc_tolerated():
    ev = signal.analyze_epochs([
        signal.SignalEpoch(0, {"a": 40, "b": 41}),
        signal.SignalEpoch(1, {"a": 52, "b": 53}),
    ])
    assert ev[-1].kind == "spoof"


def test_empty_epochs():
    assert signal.analyze_epochs([]) == []
