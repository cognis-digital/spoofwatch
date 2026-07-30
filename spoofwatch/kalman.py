"""Constant-velocity Kalman filter for resilient PNT (pure stdlib).

Where :mod:`spoofwatch.inertial` gives a one-shot plausibility gate, this module
runs a proper recursive estimator over a *track* of position fixes and delivers
three things operators actually want under interference:

1. **Smoothing + coasting** — a 2-D constant-velocity filter (state ``[e, n, ve,
   vn]`` in a local East/North tangent plane, metres and m/s). When GNSS drops
   out you keep *predicting* from the last good velocity — dead-reckoning through
   the outage — while the position covariance grows honestly so you know how much
   to trust the coasted fix.
2. **Innovation spoof-gating** — each new fix produces an *innovation* (measured
   minus predicted position). Normalised by the innovation covariance it becomes
   the **NIS** (normalised innovation squared), a chi-square statistic. A fix that
   teleports the track trips the gate and is rejected as an outlier/spoof instead
   of dragging the estimate onto the false position.
3. **A live uncertainty** — the position-covariance trace, in metres, as a
   running protection-level-style bound.

Local ENU is centred on the filter's origin (first fix) via an equirectangular
projection — accurate to well under a metre over the tens-of-km a single track
spans, and dependency-free. The 4×4 / 2×2 linear algebra reuses
:mod:`spoofwatch.linalg`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import linalg

EARTH_R_M = 6_371_000.0
# default process noise (accel PSD, m^2/s^3-ish) and measurement noise (m)
DEFAULT_PROCESS_NOISE = 1.0
DEFAULT_MEAS_NOISE_M = 15.0
# NIS chi-square gate: 2 DOF, ~99% -> 9.21; above this a fix is rejected
NIS_GATE = 9.21


def enu_from_ll(lat, lon, lat0, lon0):
    """Equirectangular local East/North metres of (lat,lon) about (lat0,lon0)."""
    e = math.radians(lon - lon0) * EARTH_R_M * math.cos(math.radians(lat0))
    n = math.radians(lat - lat0) * EARTH_R_M
    return e, n


def ll_from_enu(e, n, lat0, lon0):
    """Inverse of :func:`enu_from_ll` -> (lat, lon) degrees."""
    lat = lat0 + math.degrees(n / EARTH_R_M)
    lon = lon0 + math.degrees(e / (EARTH_R_M * math.cos(math.radians(lat0))))
    return lat, lon


@dataclass
class KalmanState:
    x: list                          # [e, n, ve, vn]
    P: list                          # 4x4 covariance
    lat0: float                      # projection origin
    lon0: float
    ts: float = 0.0
    initialized: bool = False
    coasting: bool = False
    updates: int = 0
    rejects: int = 0
    history: list = field(default_factory=list)

    @property
    def pos_enu(self):
        return (self.x[0], self.x[1])

    @property
    def vel_enu(self):
        return (self.x[2], self.x[3])

    @property
    def pos_ll(self):
        return ll_from_enu(self.x[0], self.x[1], self.lat0, self.lon0)

    @property
    def pos_uncertainty_m(self):
        # 1-sigma radial position uncertainty from the covariance trace
        return math.sqrt(max(0.0, self.P[0][0] + self.P[1][1]))


def _F(dt):
    return [
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _Q(dt, q):
    # discrete white-noise-acceleration process covariance
    dt2 = dt * dt
    dt3 = dt2 * dt / 2.0
    dt4 = dt2 * dt2 / 4.0
    return [
        [dt4 * q, 0.0, dt3 * q, 0.0],
        [0.0, dt4 * q, 0.0, dt3 * q],
        [dt3 * q, 0.0, dt2 * q, 0.0],
        [0.0, dt3 * q, 0.0, dt2 * q],
    ]


def _add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _inv2(M):
    a, b = M[0][0], M[0][1]
    c, d = M[1][0], M[1][1]
    det = a * d - b * c
    if abs(det) < 1e-15:
        raise ValueError("singular 2x2")
    return [[d / det, -b / det], [-c / det, a / det]]


def init_state(lat, lon, ts=0.0, pos_var=None, vel_var=1e4):
    """Create a fresh filter seeded at (lat,lon), used as the projection origin."""
    pv = pos_var if pos_var is not None else (DEFAULT_MEAS_NOISE_M ** 2)
    P = [[pv if i == j and i < 2 else (vel_var if i == j else 0.0)
          for j in range(4)] for i in range(4)]
    return KalmanState(x=[0.0, 0.0, 0.0, 0.0], P=P, lat0=lat, lon0=lon,
                       ts=ts, initialized=True)


def predict(state, dt, q=DEFAULT_PROCESS_NOISE):
    """Advance the state ``dt`` seconds (dead-reckoning; no measurement)."""
    if dt < 0:
        raise ValueError("dt must be non-negative")
    F = _F(dt)
    state.x = linalg.matvec(F, state.x)
    state.P = _add(linalg.matmul(linalg.matmul(F, state.P), linalg.transpose(F)),
                   _Q(dt, q))
    state.ts += dt
    state.coasting = True
    return state


def update(state, lat, lon, meas_noise_m=DEFAULT_MEAS_NOISE_M, gate=NIS_GATE):
    """Fuse a GNSS position fix, gating out teleports via the NIS chi-square test.

    Returns a dict with the innovation, its NIS statistic, and whether the fix was
    ``accepted``. A rejected fix leaves the predicted state untouched (the track
    coasts) and increments the reject counter.
    """
    e, n = enu_from_ll(lat, lon, state.lat0, state.lon0)
    # innovation y = z - H x  (H picks position)
    yv = [e - state.x[0], n - state.x[1]]
    r = meas_noise_m ** 2
    # innovation covariance S = H P Hᵀ + R  (top-left 2x2 of P plus R)
    S = [[state.P[0][0] + r, state.P[0][1]],
         [state.P[1][0], state.P[1][1] + r]]
    Sinv = _inv2(S)
    nis = (yv[0] * (Sinv[0][0] * yv[0] + Sinv[0][1] * yv[1])
           + yv[1] * (Sinv[1][0] * yv[0] + Sinv[1][1] * yv[1]))

    accepted = nis <= gate
    result = {
        "accepted": bool(accepted),
        "nis": round(nis, 4),
        "gate": gate,
        "innovation_e": round(yv[0], 3),
        "innovation_n": round(yv[1], 3),
        "innovation_norm_m": round(math.hypot(yv[0], yv[1]), 3),
    }
    if not accepted:
        state.rejects += 1
        state.history.append(result)
        return result

    # Kalman gain K = P Hᵀ S⁻¹  (Hᵀ selects the position columns)
    # P Hᵀ is the first two columns of P (4x2)
    PHt = [[state.P[i][0], state.P[i][1]] for i in range(4)]
    K = linalg.matmul(PHt, Sinv)            # 4x2
    # state update x += K y
    for i in range(4):
        state.x[i] += K[i][0] * yv[0] + K[i][1] * yv[1]
    # covariance update P = (I - K H) P ; K H is 4x4 with K in first two cols
    KH = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        KH[i][0] = K[i][0]
        KH[i][1] = K[i][1]
    ImKH = [[(1.0 if i == j else 0.0) - KH[i][j] for j in range(4)]
            for i in range(4)]
    state.P = linalg.matmul(ImKH, state.P)
    state.coasting = False
    state.updates += 1
    state.history.append(result)
    return result


def run(fixes, q=DEFAULT_PROCESS_NOISE, meas_noise_m=DEFAULT_MEAS_NOISE_M,
        gate=NIS_GATE):
    """Filter a chronological track of ``(ts, lat, lon)`` fixes end to end.

    The first fix seeds the origin. Each later fix is predicted-to and then
    gated: accepted fixes update the estimate, rejected (teleport/spoof) fixes are
    coasted through. Returns ``(state, events)`` where ``events`` is the per-fix
    update record list (empty for the seed).
    """
    fixes = sorted(fixes, key=lambda f: f[0])
    if not fixes:
        raise ValueError("no fixes")
    ts0, lat0, lon0 = fixes[0]
    state = init_state(lat0, lon0, ts=ts0)
    events = []
    for ts, lat, lon in fixes[1:]:
        dt = ts - state.ts
        if dt > 0:
            predict(state, dt, q=q)
        ev = update(state, lat, lon, meas_noise_m=meas_noise_m, gate=gate)
        ev["ts"] = ts
        events.append(ev)
    return state, events


def coast(state, dt, q=DEFAULT_PROCESS_NOISE):
    """Convenience: coast ``dt`` seconds and report the coasted fix + uncertainty."""
    predict(state, dt, q=q)
    lat, lon = state.pos_ll
    return {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "uncertainty_m": round(state.pos_uncertainty_m, 2),
        "coasting": True,
        "dt_s": dt,
    }
