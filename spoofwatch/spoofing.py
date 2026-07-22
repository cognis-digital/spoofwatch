"""Spoofing detection — kinematic implausibility + co-location signature.

Two robust, well-known GNSS-spoofing tells:
  1. **Teleport / impossible speed** — consecutive fixes on one track imply a
     ground speed no aircraft can fly.
  2. **Co-location** — many *distinct* aircraft reporting the same point at the
     same time (the classic "everyone snapped to the airport" spoof).
Pure stdlib.
"""

from __future__ import annotations

from .records import by_track, haversine_km, implied_speed_kt

MAX_PLAUSIBLE_KT = 1200.0   # above any conventional aircraft
COLOCATE_KM = 3.0           # reports this close count as the same point
COLOCATE_MIN_AIRCRAFT = 4   # this many distinct ids at one point = spoof
COLOCATE_WINDOW_S = 120.0


def _teleports(reports):
    events = []
    for acid, track in by_track(reports).items():
        for r1, r2 in zip(track, track[1:]):
            if implied_speed_kt(r1, r2) > MAX_PLAUSIBLE_KT:
                events.append({"type": "teleport", "ids": {acid},
                               "point": (round((r1.lat + r2.lat) / 2, 5),
                                         round((r1.lon + r2.lon) / 2, 5)),
                               "n": 1, "confidence": 0.9})
    return events


def _colocations(reports):
    # bucket reports by coarse time window, then group co-located distinct ids
    events = []
    buckets = {}
    for r in reports:
        buckets.setdefault(int(r.ts // COLOCATE_WINDOW_S), []).append(r)
    used = set()
    for _, rs in buckets.items():
        for i, anchor in enumerate(rs):
            key = (round(anchor.lat, 3), round(anchor.lon, 3))
            if key in used:
                continue
            near = [r for r in rs if haversine_km(anchor.lat, anchor.lon, r.lat, r.lon) <= COLOCATE_KM]
            ids = {r.id for r in near}
            if len(ids) >= COLOCATE_MIN_AIRCRAFT:
                used.add(key)
                clat = sum(r.lat for r in near) / len(near)
                clon = sum(r.lon for r in near) / len(near)
                events.append({"type": "colocation", "ids": ids,
                               "point": (round(clat, 5), round(clon, 5)),
                               "n": len(ids),
                               "confidence": round(min(1.0, len(ids) / (COLOCATE_MIN_AIRCRAFT * 2)), 2)})
    return events


def detect(reports):
    """Return spoof events (teleport + co-location), de-duplicated by point."""
    events = _teleports(reports) + _colocations(reports)
    # merge near-duplicate points, keep the higher-confidence/larger one
    merged = []
    for e in sorted(events, key=lambda x: -x["n"]):
        dup = False
        for m in merged:
            if haversine_km(e["point"][0], e["point"][1], m["point"][0], m["point"][1]) < COLOCATE_KM:
                m["ids"] |= e["ids"]; dup = True; break
        if not dup:
            merged.append(dict(e))
    return merged
