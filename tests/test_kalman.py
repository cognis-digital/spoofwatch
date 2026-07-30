import math

from spoofwatch import kalman


# --------------------------------------------------------------------------- #
# projection helpers
# --------------------------------------------------------------------------- #
def test_enu_roundtrip():
    lat0, lon0 = 59.0, 18.0
    e, n = kalman.enu_from_ll(59.05, 18.1, lat0, lon0)
    lat, lon = kalman.ll_from_enu(e, n, lat0, lon0)
    assert abs(lat - 59.05) < 1e-6
    assert abs(lon - 18.1) < 1e-6


def test_enu_origin_is_zero():
    e, n = kalman.enu_from_ll(59.0, 18.0, 59.0, 18.0)
    assert abs(e) < 1e-9 and abs(n) < 1e-9


def test_enu_east_positive():
    e, n = kalman.enu_from_ll(59.0, 18.1, 59.0, 18.0)
    assert e > 0 and abs(n) < 1.0


def test_enu_north_positive():
    e, n = kalman.enu_from_ll(59.1, 18.0, 59.0, 18.0)
    assert n > 0 and abs(e) < 1.0


def test_one_degree_north_about_111km():
    e, n = kalman.enu_from_ll(60.0, 18.0, 59.0, 18.0)
    assert 110_000 < n < 112_000


# --------------------------------------------------------------------------- #
# init / predict
# --------------------------------------------------------------------------- #
def test_init_state():
    s = kalman.init_state(59.0, 18.0, ts=100.0)
    assert s.initialized
    assert s.pos_enu == (0.0, 0.0)
    assert s.lat0 == 59.0 and s.lon0 == 18.0
    assert s.ts == 100.0


def test_init_covariance_shape():
    s = kalman.init_state(59.0, 18.0)
    assert len(s.P) == 4 and all(len(row) == 4 for row in s.P)


def test_predict_moves_position_by_velocity():
    s = kalman.init_state(0.0, 0.0)
    s.x = [0.0, 0.0, 10.0, 5.0]     # ve=10, vn=5
    kalman.predict(s, 2.0)
    assert abs(s.x[0] - 20.0) < 1e-9
    assert abs(s.x[1] - 10.0) < 1e-9


def test_predict_advances_time():
    s = kalman.init_state(0.0, 0.0, ts=50.0)
    kalman.predict(s, 3.0)
    assert s.ts == 53.0


def test_predict_sets_coasting():
    s = kalman.init_state(0.0, 0.0)
    kalman.predict(s, 1.0)
    assert s.coasting is True


def test_predict_grows_covariance():
    s = kalman.init_state(0.0, 0.0)
    tr0 = s.P[0][0] + s.P[1][1]
    kalman.predict(s, 5.0)
    tr1 = s.P[0][0] + s.P[1][1]
    assert tr1 > tr0


def test_predict_negative_dt_raises():
    s = kalman.init_state(0.0, 0.0)
    try:
        kalman.predict(s, -1.0)
        assert False
    except ValueError:
        pass


def test_predict_velocity_unchanged():
    s = kalman.init_state(0.0, 0.0)
    s.x = [0.0, 0.0, 7.0, -3.0]
    kalman.predict(s, 4.0)
    assert abs(s.x[2] - 7.0) < 1e-9
    assert abs(s.x[3] + 3.0) < 1e-9


# --------------------------------------------------------------------------- #
# update / gating
# --------------------------------------------------------------------------- #
def test_update_accepts_nearby_fix():
    s = kalman.init_state(59.0, 18.0)
    r = kalman.update(s, 59.0001, 18.0001)
    assert r["accepted"] is True
    assert s.updates == 1


def test_update_reduces_uncertainty():
    s = kalman.init_state(59.0, 18.0)
    kalman.predict(s, 1.0)
    before = s.pos_uncertainty_m
    kalman.update(s, 59.0, 18.0)
    assert s.pos_uncertainty_m < before


def test_update_clears_coasting():
    s = kalman.init_state(59.0, 18.0)
    kalman.predict(s, 1.0)
    kalman.update(s, 59.0, 18.0)
    assert s.coasting is False


def test_teleport_is_rejected():
    s = kalman.init_state(59.0, 18.0)
    # tighten the state uncertainty with a couple of good fixes first
    kalman.update(s, 59.0, 18.0)
    kalman.predict(s, 1.0)
    kalman.update(s, 59.0, 18.0)
    # now a 50 km jump
    r = kalman.update(s, 59.5, 18.5)
    assert r["accepted"] is False
    assert s.rejects >= 1


def test_rejected_fix_leaves_state():
    s = kalman.init_state(59.0, 18.0)
    kalman.update(s, 59.0, 18.0)
    pos_before = tuple(s.x)
    kalman.update(s, 60.0, 19.0)   # huge jump, rejected
    assert tuple(s.x) == pos_before


def test_nis_reported():
    s = kalman.init_state(59.0, 18.0)
    r = kalman.update(s, 59.0, 18.0)
    assert "nis" in r and r["nis"] >= 0.0


def test_innovation_norm_reported():
    s = kalman.init_state(59.0, 18.0)
    r = kalman.update(s, 59.001, 18.0)
    assert r["innovation_norm_m"] > 0


def test_gate_override_accepts():
    s = kalman.init_state(59.0, 18.0)
    kalman.update(s, 59.0, 18.0)
    # with an enormous gate even a jump is accepted
    r = kalman.update(s, 59.2, 18.2, gate=1e12)
    assert r["accepted"] is True


# --------------------------------------------------------------------------- #
# run()
# --------------------------------------------------------------------------- #
def _straight_track(n=10, dt=1.0, dlat=0.0005, dlon=0.0003, lat0=59.0, lon0=18.0):
    return [(i * dt, lat0 + i * dlat, lon0 + i * dlon) for i in range(n)]


def test_run_straight_track():
    state, events = kalman.run(_straight_track())
    assert len(events) == 9
    assert all(e["accepted"] for e in events)
    assert state.updates == 9


def test_run_tracks_velocity():
    state, _ = kalman.run(_straight_track(n=15))
    ve, vn = state.vel_enu
    # north velocity should be positive (track heads north-east)
    assert vn > 0 and ve > 0


def test_run_estimate_near_last_fix():
    track = _straight_track(n=12)
    state, _ = kalman.run(track)
    lat, lon = state.pos_ll
    _, last_lat, last_lon = track[-1]
    assert abs(lat - last_lat) < 0.01
    assert abs(lon - last_lon) < 0.01


def test_run_rejects_injected_teleport():
    track = _straight_track(n=8)
    # inject a spoofed teleport in the middle
    track.insert(4, (3.5, 61.0, 20.0))
    state, events = kalman.run(track)
    assert any(not e["accepted"] for e in events)


def test_run_empty_raises():
    try:
        kalman.run([])
        assert False
    except ValueError:
        pass


def test_run_sorts_unordered_fixes():
    track = _straight_track(n=6)
    shuffled = [track[3], track[0], track[5], track[1], track[4], track[2]]
    state, events = kalman.run(shuffled)
    assert [e["ts"] for e in events] == sorted(e["ts"] for e in events)


def test_run_events_have_ts():
    _, events = kalman.run(_straight_track())
    assert all("ts" in e for e in events)


# --------------------------------------------------------------------------- #
# coast()
# --------------------------------------------------------------------------- #
def test_coast_extrapolates_forward():
    state, _ = kalman.run(_straight_track(n=10))
    lat_before, lon_before = state.pos_ll
    out = kalman.coast(state, 10.0)
    # position should advance along the established heading
    assert out["lat"] != lat_before or out["lon"] != lon_before
    assert out["coasting"] is True


def test_coast_grows_uncertainty():
    state, _ = kalman.run(_straight_track(n=10))
    u0 = state.pos_uncertainty_m
    out = kalman.coast(state, 30.0)
    assert out["uncertainty_m"] > u0


def test_coast_reports_dt():
    state, _ = kalman.run(_straight_track(n=6))
    out = kalman.coast(state, 12.0)
    assert out["dt_s"] == 12.0


def test_uncertainty_positive():
    s = kalman.init_state(59.0, 18.0)
    assert s.pos_uncertainty_m > 0


def test_long_outage_then_reacquire():
    # coast through a gap, then a fix consistent with the coasted position is kept
    state, _ = kalman.run(_straight_track(n=8, dt=1.0))
    kalman.predict(state, 5.0)             # 5 s outage
    lat, lon = state.pos_ll
    r = kalman.update(state, lat, lon)     # reacquire at the coasted position
    assert r["accepted"] is True
