"""Inertial-coast plausibility gate (pure stdlib).

The idea behind tightly-coupled GNSS/INS bridging, reduced to an offline
plausibility check: from the last *authenticated* fix, propagate where the
platform could plausibly be now — its expected position plus a growing
uncertainty radius driven by IMU/dead-reckoning drift — and reject any incoming
GNSS fix that lands outside that coast envelope. A spoofer that teleports you
will always fall outside the envelope; honest drift will not.

It also reports a **coast-time budget**: how long you can navigate on inertial
alone before the uncertainty radius exceeds an operator's tolerance — the answer
an operator in a jamming zone actually needs.

No Kalman filter, no NumPy — a configurable kinematic drift model. Distances use
the repo's haversine helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .records import haversine_km

# default INS grade: horizontal position drift growth (km per second of coast)
DEFAULT_DRIFT_KM_S = 0.02        # ~ tactical-grade IMU, unaided
# base uncertainty at the moment of the last good fix (km)
BASE_UNCERTAINTY_KM = 0.5
# operator tolerance: coast budget is the time until the radius hits this (km)
DEFAULT_TOLERANCE_KM = 5.0


@dataclass
class Fix:
    ts: float
    lat: float
    lon: float
    speed_mps: float = 0.0          # ground speed at the fix
    heading_deg: float = 0.0        # course over ground (deg from North)


def _project(fix: Fix, dt_s: float):
    """Dead-reckon a fix forward dt_s seconds along its heading at its speed."""
    dist_km = fix.speed_mps * dt_s / 1000.0
    dlat = (dist_km / 111.0) * math.cos(math.radians(fix.heading_deg))
    dlon = (dist_km / (111.0 * max(0.1, math.cos(math.radians(fix.lat))))) \
        * math.sin(math.radians(fix.heading_deg))
    return fix.lat + dlat, fix.lon + dlon


def coast_radius_km(dt_s, drift_km_s=DEFAULT_DRIFT_KM_S, base_km=BASE_UNCERTAINTY_KM):
    """Uncertainty radius of the coast envelope after dt_s seconds."""
    return base_km + drift_km_s * max(0.0, dt_s)


def coast_budget_s(drift_km_s=DEFAULT_DRIFT_KM_S, base_km=BASE_UNCERTAINTY_KM,
                   tolerance_km=DEFAULT_TOLERANCE_KM):
    """Seconds of unaided inertial coast before the radius exceeds tolerance."""
    if drift_km_s <= 0:
        return float("inf")
    budget = (tolerance_km - base_km) / drift_km_s
    return max(0.0, budget)


def gate(last_fix: Fix, gnss_fix: Fix, drift_km_s=DEFAULT_DRIFT_KM_S,
         base_km=BASE_UNCERTAINTY_KM, tolerance_km=DEFAULT_TOLERANCE_KM):
    """Test an incoming GNSS fix against the inertial coast envelope.

    Returns a dict: the propagated expected point, the envelope radius, the
    actual distance of the incoming fix from it, a boolean ``outside`` flag
    (spoof/implausible), and the remaining coast-time budget.
    """
    dt = gnss_fix.ts - last_fix.ts
    exp_lat, exp_lon = _project(last_fix, dt)
    radius = coast_radius_km(dt, drift_km_s, base_km)
    dist = haversine_km(exp_lat, exp_lon, gnss_fix.lat, gnss_fix.lon)
    outside = dist > radius
    excess = dist - radius
    conf = round(min(1.0, excess / max(radius, 1e-6)), 3) if outside else 0.0
    return {
        "dt_s": round(dt, 3),
        "expected_lat": round(exp_lat, 5),
        "expected_lon": round(exp_lon, 5),
        "envelope_radius_km": round(radius, 3),
        "fix_distance_km": round(dist, 3),
        "outside": bool(outside),
        "excess_km": round(excess, 3),
        "confidence": conf,
        "coast_budget_s": round(coast_budget_s(drift_km_s, base_km, tolerance_km), 1),
    }
