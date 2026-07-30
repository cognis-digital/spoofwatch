from spoofwatch import pnttrack
from spoofwatch.pnttrack import TrustSample


def _samples(vals, t0=0):
    return [TrustSample(t0 + i, v) for i, v in enumerate(vals)]


def test_empty():
    r = pnttrack.track([])
    assert r["points"] == []
    assert r["episodes"] == []
    assert r["summary"]["epochs"] == 0
    assert r["summary"]["in_denial_now"] is False


def test_all_high_locked():
    r = pnttrack.track(_samples([0.95] * 8))
    assert r["summary"]["final_state"] == "LOCKED"
    assert r["episodes"] == []
    assert r["summary"]["in_denial_now"] is False


def test_points_length():
    r = pnttrack.track(_samples([0.9] * 5))
    assert len(r["points"]) == 5
    assert r["summary"]["epochs"] == 5


def test_starts_denied_when_low():
    r = pnttrack.track(_samples([0.05] * 6))
    assert r["points"][0].state == "DENIED"
    assert r["summary"]["denial_episodes"] >= 1


def test_sustained_denial_episode():
    r = pnttrack.track(_samples([0.95] * 3 + [0.05] * 8))
    assert r["summary"]["denial_episodes"] >= 1
    assert r["summary"]["in_denial_now"] is True
    assert r["summary"]["final_state"] in ("DENIED", "SUSPECT")


def test_transient_single_dip_no_denial():
    # one low sample amid high trust must not open a denial episode
    r = pnttrack.track(_samples([0.9] * 3 + [0.1] + [0.9] * 6))
    assert r["summary"]["denial_episodes"] == 0


def test_recovery_closes_episode():
    r = pnttrack.track(_samples([0.05] * 6 + [0.95] * 8))
    assert r["summary"]["final_state"] == "LOCKED"
    assert r["summary"]["in_denial_now"] is False
    assert r["episodes"][0]["ongoing"] is False
    assert r["episodes"][0]["end_ts"] is not None


def test_ongoing_episode_at_end():
    r = pnttrack.track(_samples([0.9] * 2 + [0.05] * 8))
    assert r["episodes"][-1]["ongoing"] is True
    assert r["episodes"][-1]["end_ts"] is None


def test_min_trust_tracked():
    r = pnttrack.track(_samples([0.9, 0.9, 0.02, 0.9, 0.9]))
    assert r["summary"]["min_trust"] == 0.02


def test_smoothed_bounded():
    r = pnttrack.track(_samples([0.0, 1.0, 0.0, 1.0, 0.5, 0.3]))
    assert all(0.0 <= p.smoothed <= 1.0 for p in r["points"])


def test_raw_trust_preserved():
    r = pnttrack.track(_samples([0.9, 0.4, 0.1]))
    assert [round(p.trust, 2) for p in r["points"]] == [0.9, 0.4, 0.1]


def test_unsorted_input_sorted():
    s = [TrustSample(2, 0.9), TrustSample(0, 0.9), TrustSample(1, 0.9)]
    r = pnttrack.track(s)
    assert [p.ts for p in r["points"]] == [0, 1, 2]


def test_episode_fields_present():
    r = pnttrack.track(_samples([0.05] * 8))
    ep = r["episodes"][0]
    for key in ("start_ts", "end_ts", "duration_s", "min_trust",
                "worst_state", "epochs", "ongoing"):
        assert key in ep


def test_worst_state_denied():
    r = pnttrack.track(_samples([0.02] * 10))
    assert r["episodes"][0]["worst_state"] == "DENIED"


def test_final_state_valid():
    r = pnttrack.track(_samples([0.7] * 6))
    assert r["summary"]["final_state"] in pnttrack.STATE_ORDER


def test_state_order_ranking():
    assert pnttrack.STATE_ORDER == ["DENIED", "SUSPECT", "DEGRADED", "LOCKED"]


def test_total_denial_positive():
    r = pnttrack.track(_samples([0.05] * 8))
    assert r["summary"]["total_denial_s"] > 0


def test_hysteresis_delays_entry():
    # trust drops sharply but state does not commit to a worse band on the first epoch
    r = pnttrack.track(_samples([0.95, 0.95, 0.05]))
    # third epoch's committed state should still be the better band (enter_hold not met)
    assert r["points"][2].state == "LOCKED"


def test_degraded_band_not_denial():
    # trust parked in the DEGRADED band is not a denial episode
    r = pnttrack.track(_samples([0.65] * 10))
    assert r["summary"]["denial_episodes"] == 0


def test_two_denial_episodes():
    vals = [0.95] * 3 + [0.05] * 8 + [0.95] * 10 + [0.05] * 8
    r = pnttrack.track(vals and _samples(vals))
    assert r["summary"]["denial_episodes"] == 2


def test_custom_alpha_smoothing():
    slow = pnttrack.track(_samples([1.0, 0.0, 0.0]), alpha=0.1)
    fast = pnttrack.track(_samples([1.0, 0.0, 0.0]), alpha=0.9)
    # a faster alpha reacts harder to the drop
    assert fast["points"][-1].smoothed < slow["points"][-1].smoothed
