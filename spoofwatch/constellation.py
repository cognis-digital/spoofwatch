"""Multi-constellation cross-check (pure stdlib).

A spoofer running a single signal generator can usually only coherently forge
*one* constellation at a time; GPS, Galileo, GLONASS and BeiDou use independent
signals, codes and timekeeping. So when a receiver can emit an independent
position/time sub-solution *per constellation*, disagreement between them beyond
what geometry alone would explain is a robust spoof tell — even when each
individual fix looks internally consistent.

Feed a set of per-constellation :class:`SubSolution` objects for one epoch; this
computes the multi-constellation centroid, each constellation's divergence from
it, and flags the outliers (position and/or clock).
"""

from __future__ import annotations

from dataclasses import dataclass

from .records import haversine_km

# constellations diverging more than this from the consensus are inconsistent
POS_DIVERGENCE_KM = 2.0
# clock/time sub-solution divergence threshold (nanoseconds)
TIME_DIVERGENCE_NS = 300.0


@dataclass
class SubSolution:
    constellation: str
    lat: float
    lon: float
    clock_ns: float | None = None   # optional per-constellation time solution


def _centroid(subs):
    return (sum(s.lat for s in subs) / len(subs),
            sum(s.lon for s in subs) / len(subs))


def cross_check(subs, pos_thresh_km=POS_DIVERGENCE_KM, time_thresh_ns=TIME_DIVERGENCE_NS):
    """Cross-check per-constellation sub-solutions for one epoch.

    Uses a robust consensus: the centroid of all *but* the single
    farthest-out constellation, so one spoofed constellation cannot drag the
    reference onto itself. Returns a dict with per-constellation divergences and
    the list of flagged (diverging) constellations.
    """
    subs = list(subs)
    if len(subs) < 2:
        return {"constellations": [s.constellation for s in subs],
                "divergence": bool(False), "flagged": [], "max_divergence_km": 0.0,
                "note": "need >= 2 constellations to cross-check"}

    # robust reference: leave-one-out centroid minimising spread
    full_c = _centroid(subs)
    if len(subs) >= 3:
        # drop the single point farthest from the full centroid, recompute
        far = max(range(len(subs)),
                  key=lambda i: haversine_km(full_c[0], full_c[1], subs[i].lat, subs[i].lon))
        ref_subs = subs[:far] + subs[far + 1:]
    else:
        ref_subs = subs
    ref = _centroid(ref_subs)

    per = []
    flagged = []
    for s in subs:
        d_km = haversine_km(ref[0], ref[1], s.lat, s.lon)
        entry = {"constellation": s.constellation, "divergence_km": round(d_km, 3)}
        pos_bad = d_km > pos_thresh_km
        entry["position_diverges"] = bool(pos_bad)
        if pos_bad:
            flagged.append(s.constellation)
        per.append(entry)

    # clock cross-check (optional)
    clocks = [(s.constellation, s.clock_ns) for s in subs if s.clock_ns is not None]
    time_flagged = []
    if len(clocks) >= 2:
        med = sorted(c[1] for c in clocks)[len(clocks) // 2]
        for name, val in clocks:
            if abs(val - med) > time_thresh_ns:
                time_flagged.append(name)

    max_div = max((p["divergence_km"] for p in per), default=0.0)
    diverged = bool(flagged) or bool(time_flagged)
    return {
        "constellations": [s.constellation for s in subs],
        "reference_lat": round(ref[0], 5),
        "reference_lon": round(ref[1], 5),
        "per_constellation": per,
        "flagged": sorted(set(flagged)),
        "time_flagged": sorted(set(time_flagged)),
        "max_divergence_km": round(max_div, 3),
        "divergence": diverged,
        "confidence": round(min(1.0, max_div / (pos_thresh_km * 3)), 3) if diverged else 0.0,
    }
