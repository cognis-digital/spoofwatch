"""Code-carrier divergence monitor + carrier smoothing (pure stdlib).

Two independent range measurements come off every tracked satellite: the
**code** pseudorange (noisy, multipath-prone, ionosphere-delayed) and the
**carrier phase** (precise, smooth, but ambiguous by an unknown integer number of
wavelengths). Because both measure the same geometric range, their *difference*
should change only slowly and smoothly — driven by the ionosphere's gentle
code/carrier divergence. A sudden or sustained anomalous divergence **rate** is
the fingerprint of multipath, meaconing, or a spoofer dragging the code away from
the still-locked carrier.

This module tracks the divergence per satellite relative to its first epoch
(cancelling the unknown carrier ambiguity), measures its rate of change, and
flags satellites whose divergence rate exceeds a physical bound. It also provides
a **Hatch filter** — the standard carrier-smoothing of the code pseudorange,
which suppresses code noise and multipath using the low-noise carrier and is a
first-line resilient-PNT technique in its own right.

Ranges are metres; carrier phase is supplied already scaled to metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# a divergence rate above this magnitude (m/s) is physically implausible
DIVERGENCE_RATE_M_S = 1.0
# require at least this many epochs on a satellite to judge its rate
MIN_EPOCHS = 3
# default Hatch smoothing window (epochs)
HATCH_WINDOW = 100


@dataclass
class RangeSample:
    ts: float
    sat_id: str
    code_m: float                   # code pseudorange (m)
    phase_m: float                  # carrier phase scaled to metres (m)
    constellation: str = "GPS"


def by_sat(samples):
    """Group range samples by satellite, each track sorted by time."""
    tracks = {}
    for s in samples:
        tracks.setdefault(s.sat_id, []).append(s)
    for t in tracks.values():
        t.sort(key=lambda s: s.ts)
    return tracks


def divergence_series(track):
    """Per-epoch code-carrier divergence for one satellite track, ambiguity-removed.

    Returns a list of ``(ts, divergence_m)`` where divergence is
    ``(code - phase)`` referenced to the track's first epoch, so the unknown
    integer carrier ambiguity cancels and the series starts at 0.
    """
    track = sorted(track, key=lambda s: s.ts)
    if not track:
        return []
    ref = track[0].code_m - track[0].phase_m
    return [(s.ts, (s.code_m - s.phase_m) - ref) for s in track]


def _max_abs_rate(series):
    """Largest absolute epoch-to-epoch divergence rate (m/s) in a series."""
    worst = 0.0
    for (t0, d0), (t1, d1) in zip(series, series[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        worst = max(worst, abs(d1 - d0) / dt)
    return worst


def check(samples, rate_thresh=DIVERGENCE_RATE_M_S, min_epochs=MIN_EPOCHS):
    """Flag satellites whose code-carrier divergence rate is implausible.

    Returns a dict with per-satellite divergence stats, the list of ``flagged``
    satellites, and a 0..1 confidence scaled by how far the worst rate exceeds
    the threshold.
    """
    tracks = by_sat(samples)
    per = []
    flagged = []
    worst_overall = 0.0
    for sat_id, track in sorted(tracks.items()):
        if len(track) < min_epochs:
            continue
        series = divergence_series(track)
        rate = _max_abs_rate(series)
        total = series[-1][1] - series[0][1]
        worst_overall = max(worst_overall, rate)
        bad = rate > rate_thresh
        entry = {"sat_id": sat_id, "epochs": len(track),
                 "max_rate_m_s": round(rate, 4),
                 "total_divergence_m": round(total, 4),
                 "diverges": bool(bad)}
        per.append(entry)
        if bad:
            flagged.append(sat_id)
    conf = round(min(1.0, max(0.0, (worst_overall - rate_thresh) / rate_thresh)), 3) \
        if flagged else 0.0
    return {"per_satellite": per, "flagged": sorted(flagged),
            "n_flagged": len(flagged), "max_rate_m_s": round(worst_overall, 4),
            "diverges": bool(flagged), "confidence": conf}


def hatch_smooth(track, window=HATCH_WINDOW):
    """Carrier-smooth a single satellite's code pseudorange (Hatch filter).

    The recursive Hatch filter blends the incoming code with the previous
    smoothed value propagated by the carrier-phase change::

        s_k = (1/N) * code_k + ((N-1)/N) * (s_{k-1} + (phase_k - phase_{k-1}))

    with the weight ``N`` ramping from 1 up to ``window`` so early epochs trust
    the code and later epochs trust the accumulated carrier. Returns a list of
    ``(ts, smoothed_code_m)``. Strongly suppresses code multipath/noise — a
    resilient-PNT staple.
    """
    track = sorted(track, key=lambda s: s.ts)
    if not track:
        return []
    out = [(track[0].ts, track[0].code_m)]
    smoothed = track[0].code_m
    for k in range(1, len(track)):
        n = min(k + 1, window)
        dphase = track[k].phase_m - track[k - 1].phase_m
        smoothed = (1.0 / n) * track[k].code_m + \
                   ((n - 1.0) / n) * (smoothed + dphase)
        out.append((track[k].ts, smoothed))
    return out
