"""Doppler / pseudorange-rate consistency check (pure stdlib).

Every genuine GNSS signal arrives with a Doppler shift set by the *relative
motion* between the satellite and the receiver along their line of sight. A
spoofer transmitting from a fixed ground antenna cannot reproduce that geometry
for every satellite at once: its counterfeit signals carry a Doppler that
matches the spoofer→receiver geometry (often near-static), not the real
satellite→receiver geometry. Compare the Doppler a receiver *measures* against
the Doppler its claimed satellite ephemeris *predicts* and the mismatch is one
of the hardest signatures for an attacker to fake.

Physics (single-frequency, non-relativistic):

    range_rate = (sat_vel - rx_vel) · los_unit          (m/s, +ve = receding)
    doppler_hz = -(carrier_freq / c) * range_rate

so an approaching satellite (negative range rate) yields a positive Doppler.
Feed per-satellite ECEF position + velocity and the receiver's own state; the
detector returns per-satellite Doppler residuals, an aggregate RMS, and two
labelled signatures — a broad inconsistency and the tell-tale *static-spoofer*
case where predicted Dopplers span a wide range but every measured value sits
near zero. No hardware assumptions beyond a receiver that reports Doppler (most
do). No NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SPEED_OF_LIGHT = 299_792_458.0          # m/s
GPS_L1_HZ = 1_575_420_000.0             # GPS/Galileo L1 carrier (Hz)

# a measured-vs-predicted Doppler residual (Hz) above this is inconsistent
RESIDUAL_THRESH_HZ = 25.0
# minimum satellites needed to judge consistency
MIN_SATS = 4
# static-spoofer test: predicted Doppler spread must exceed this to be meaningful
STATIC_PRED_SPREAD_HZ = 200.0
# ...while measured spread stays under this (signals are near-static / uniform)
STATIC_MEAS_SPREAD_HZ = 40.0


@dataclass
class DopplerObs:
    sat_id: str
    sat_pos: tuple           # satellite ECEF position (x, y, z) metres
    sat_vel: tuple           # satellite ECEF velocity (vx, vy, vz) m/s
    measured_hz: float       # Doppler the receiver reports (Hz)
    constellation: str = "GPS"


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def los_unit(rx_pos, sat_pos):
    """Unit line-of-sight vector pointing from the receiver to the satellite."""
    d = _sub(sat_pos, rx_pos)
    n = _norm(d)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    return (d[0] / n, d[1] / n, d[2] / n)


def range_rate(rx_pos, rx_vel, sat_pos, sat_vel):
    """Line-of-sight range rate (m/s); positive means the range is opening."""
    u = los_unit(rx_pos, sat_pos)
    return _dot(_sub(sat_vel, rx_vel), u)


def expected_doppler_hz(rx_pos, rx_vel, sat_pos, sat_vel, freq_hz=GPS_L1_HZ):
    """Predicted carrier Doppler (Hz) from the satellite/receiver geometry."""
    rr = range_rate(rx_pos, rx_vel, sat_pos, sat_vel)
    return -(freq_hz / SPEED_OF_LIGHT) * rr


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _spread(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


@dataclass
class DopplerResult:
    available: bool
    n_sats: int
    residual_rms_hz: float = 0.0
    max_residual_hz: float = 0.0
    predicted_spread_hz: float = 0.0
    measured_spread_hz: float = 0.0
    inconsistent: bool = False
    static_spoofer: bool = False
    confidence: float = 0.0
    per_sat: list = field(default_factory=list)
    reason: str = ""


def check(observations, rx_pos, rx_vel=(0.0, 0.0, 0.0), freq_hz=GPS_L1_HZ,
          thresh_hz=RESIDUAL_THRESH_HZ, min_sats=MIN_SATS):
    """Flag Doppler inconsistency between measured and geometry-predicted shifts.

    Returns a :class:`DopplerResult`. ``inconsistent`` trips when the residual
    RMS exceeds ``thresh_hz``; ``static_spoofer`` trips on the narrower signature
    of a fixed ground transmitter — a wide *predicted* Doppler spread across the
    sky collapsed to a near-uniform *measured* spread. Confidence scales with how
    far the residual RMS overshoots the threshold, reinforced when the static
    signature is present.
    """
    obs = list(observations)
    if len(obs) < min_sats:
        return DopplerResult(False, len(obs),
                             reason=f"need >= {min_sats} satellites")

    predicted = []
    measured = []
    residuals = []
    per_sat = []
    for o in obs:
        pred = expected_doppler_hz(rx_pos, rx_vel, o.sat_pos, o.sat_vel, freq_hz)
        resid = o.measured_hz - pred
        predicted.append(pred)
        measured.append(o.measured_hz)
        residuals.append(resid)
        per_sat.append({
            "sat_id": o.sat_id,
            "predicted_hz": round(pred, 2),
            "measured_hz": round(o.measured_hz, 2),
            "residual_hz": round(resid, 2),
        })

    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    max_res = max(abs(r) for r in residuals)
    pred_spread = _spread(predicted)
    meas_spread = _spread(measured)

    inconsistent = rms > thresh_hz
    static = (pred_spread >= STATIC_PRED_SPREAD_HZ
              and meas_spread <= STATIC_MEAS_SPREAD_HZ)

    conf = 0.0
    if inconsistent and thresh_hz > 0:
        conf = min(1.0, (rms - thresh_hz) / (thresh_hz * 2))
    if static:
        conf = max(conf, 0.6)
    conf = round(max(0.0, min(1.0, conf)), 3)

    return DopplerResult(
        available=True,
        n_sats=len(obs),
        residual_rms_hz=round(rms, 2),
        max_residual_hz=round(max_res, 2),
        predicted_spread_hz=round(pred_spread, 2),
        measured_spread_hz=round(meas_spread, 2),
        inconsistent=bool(inconsistent),
        static_spoofer=bool(static),
        confidence=conf,
        per_sat=per_sat,
    )
