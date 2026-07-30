"""Alternative-PNT availability overlay (pure stdlib, reference layer).

When spoofwatch flags a jamming/spoofing zone, the operator's next question is
"so what do I navigate on *instead*?". This overlay answers it: given a location
(and the inertial grade in play), it estimates which fallback PNT modes remain
viable —

* **LEO signals-of-opportunity** (Starlink & other broadband Doppler positioning,
  no data decoded) — global, meter-to-few-metre class where terminals/receivers
  exist;
* **eLoran** — high-power low-frequency terrestrial PNT, sub-10 m and far harder
  to jam than GNSS, but only inside transmitter coverage; a small built-in
  station list (former LORAN-C / UK & EU trial sites) drives a coverage estimate;
* **inertial coast** — the seconds of unaided dead-reckoning budget from
  :mod:`spoofwatch.inertial`.

This is an analysis/reference layer, **not** a navigation solver — it tells you
what's available, not where you are.
"""

from __future__ import annotations

from .inertial import coast_budget_s
from .records import haversine_km

# representative eLoran / former-LORAN-C sites with nominal usable coverage (km).
# Illustrative for coverage estimation — not an operational almanac.
ELORAN_STATIONS = [
    {"name": "Anthorn (UK)", "lat": 54.911, "lon": -3.278, "coverage_km": 1200},
    {"name": "Lessay (FR)", "lat": 49.145, "lon": -1.505, "coverage_km": 1000},
    {"name": "Sylt (DE)", "lat": 54.808, "lon": 8.293, "coverage_km": 1000},
    {"name": "Ejde (Faroe)", "lat": 62.300, "lon": -7.070, "coverage_km": 1200},
    {"name": "Vaerlandet (NO)", "lat": 61.298, "lon": 4.688, "coverage_km": 1000},
    {"name": "Jan Mayen (NO)", "lat": 70.981, "lon": -8.720, "coverage_km": 1200},
]

# LEO SoO (Starlink-class Doppler) is treated as near-global between ~70S..70N
LEO_LAT_LIMIT = 70.0


def eloran_coverage(lat, lon, stations=None):
    """Best (nearest, in-coverage) eLoran station for a point, or None."""
    stations = stations if stations is not None else ELORAN_STATIONS
    best = None
    for s in stations:
        d = haversine_km(lat, lon, s["lat"], s["lon"])
        if d <= s["coverage_km"] and (best is None or d < best[1]):
            best = (s, d)
    if best is None:
        return None
    s, d = best
    # signal margin 1.0 at the transmitter, tapering to 0 at the coverage edge
    margin = round(max(0.0, 1.0 - d / s["coverage_km"]), 3)
    return {"station": s["name"], "distance_km": round(d, 1),
            "signal_margin": margin, "nominal_accuracy_m": 10.0}


def leo_soo_available(lat):
    """Whether LEO signals-of-opportunity (Starlink-class) are usable at a latitude."""
    return abs(lat) <= LEO_LAT_LIMIT


def availability(lat, lon, drift_km_s=None, tolerance_km=5.0, stations=None):
    """Resilience overlay for a location: which backup PNT modes remain viable.

    Returns a dict describing LEO SoO, eLoran, and inertial-coast availability,
    plus a coarse ``resilience`` rating (none/low/moderate/good) from how many
    independent fallbacks are in play.
    """
    from .inertial import DEFAULT_DRIFT_KM_S
    drift = DEFAULT_DRIFT_KM_S if drift_km_s is None else drift_km_s

    leo = leo_soo_available(lat)
    eloran = eloran_coverage(lat, lon, stations)
    coast = coast_budget_s(drift_km_s=drift, tolerance_km=tolerance_km)

    modes = 0
    modes += 1 if leo else 0
    modes += 1 if eloran else 0
    modes += 1 if coast >= 30 else 0
    rating = ["none", "low", "moderate", "good"][min(modes, 3)]

    return {
        "lat": lat, "lon": lon,
        "leo_soo": {"available": leo,
                    "nominal_accuracy_m_2d": 3.6 if leo else None},
        "eloran": eloran,
        "inertial_coast_s": round(coast, 1),
        "fallback_modes": modes,
        "resilience": rating,
    }
