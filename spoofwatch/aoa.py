"""Single-antenna / angle-of-arrival spoofing check (pure stdlib, hardware-optional).

A ground spoofer transmits every counterfeit "satellite" from *one* place, so
all signals arrive from a single direction — whereas a genuine constellation is
spread across the sky. Receivers with a controlled-reception-pattern antenna, a
rotating single antenna, or interferometric phase data can expose a
direction-of-arrival (azimuth/elevation) per satellite. Given those, this
detector flags the spoofing signature: the angular spread of arrivals collapses
toward a single point.

Documented as an *advanced, hardware-optional* signal — most feeds won't carry
DoA, but where they do it is one of the most decisive tells available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# genuine sky coverage has wide angular spread; below this (deg) is suspicious
MIN_ANGULAR_SPREAD_DEG = 15.0
# minimum satellites needed to judge the spread
MIN_SATS = 4


@dataclass
class DoAObservation:
    sat_id: str
    az_deg: float           # azimuth, degrees clockwise from North
    el_deg: float           # elevation, degrees above horizon


def _unit(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return (math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el))


def angular_spread_deg(observations):
    """Mean angular separation (deg) of arrivals from their resultant direction.

    Sums the unit direction vectors; a tight single-source cluster yields a long
    resultant (small mean angle), a sky-spread constellation a short one (large
    mean angle).
    """
    vecs = [_unit(o.az_deg, o.el_deg) for o in observations]
    if not vecs:
        return 0.0
    sx = sum(v[0] for v in vecs); sy = sum(v[1] for v in vecs); sz = sum(v[2] for v in vecs)
    mag = math.sqrt(sx * sx + sy * sy + sz * sz)
    if mag < 1e-9:
        return 90.0
    mean = (sx / mag, sy / mag, sz / mag)
    angs = []
    for v in vecs:
        dot = max(-1.0, min(1.0, v[0] * mean[0] + v[1] * mean[1] + v[2] * mean[2]))
        angs.append(math.degrees(math.acos(dot)))
    return sum(angs) / len(angs)


def check(observations, min_spread_deg=MIN_ANGULAR_SPREAD_DEG, min_sats=MIN_SATS):
    """Flag a single-source (single-antenna spoofer) direction-of-arrival signature.

    Returns a dict with the measured angular spread, a boolean ``single_source``
    flag, and a confidence that scales inversely with the spread. When too few
    satellites are present the check is reported as unavailable.
    """
    obs = list(observations)
    if len(obs) < min_sats:
        return {"available": False, "n_sats": len(obs),
                "reason": f"need >= {min_sats} satellites", "single_source": False}
    spread = angular_spread_deg(obs)
    single = spread < min_spread_deg
    conf = round(max(0.0, min(1.0, 1.0 - spread / min_spread_deg)), 3) if single else 0.0
    return {
        "available": True,
        "n_sats": len(obs),
        "angular_spread_deg": round(spread, 2),
        "threshold_deg": min_spread_deg,
        "single_source": bool(single),
        "confidence": conf,
    }
