"""Position-report records + geo helpers (pure stdlib).

A Report is one position message from an aircraft (ADS-B) or vessel (AIS):
id, timestamp, lat/lon, and a navigation-integrity indicator (0..10, à la
ADS-B NIC/NACp — low means the transmitter itself is unsure of its fix, the
classic GPS-jamming signature). Loads from JSON or CSV.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass

EARTH_KM = 6371.0088


@dataclass
class Report:
    id: str            # aircraft/vessel identifier (icao24, mmsi, callsign)
    ts: float          # epoch seconds
    lat: float
    lon: float
    integrity: float = 8.0   # 0..10; low = degraded/lost GPS fix
    speed_kt: float | None = None   # reported ground speed if available

    def as_dict(self):
        return {"id": self.id, "ts": self.ts, "lat": round(self.lat, 6),
                "lon": round(self.lon, 6), "integrity": self.integrity}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def implied_speed_kt(r1: "Report", r2: "Report") -> float:
    """Ground speed implied by the jump between two consecutive reports (knots)."""
    dt_h = abs(r2.ts - r1.ts) / 3600.0
    if dt_h <= 0:
        return float("inf")
    km = haversine_km(r1.lat, r1.lon, r2.lat, r2.lon)
    return (km / 1.852) / dt_h   # km/h → knots


def load_json(path) -> list:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [Report(**{k: v for k, v in d.items() if k in Report.__annotations__}) for d in raw]


def load_csv(path) -> list:
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append(Report(id=row["id"], ts=float(row["ts"]),
                              lat=float(row["lat"]), lon=float(row["lon"]),
                              integrity=float(row.get("integrity", 8.0)),
                              speed_kt=float(row["speed_kt"]) if row.get("speed_kt") else None))
    return out


def by_track(reports) -> dict:
    """Group reports by id, sorted by time — a per-aircraft track."""
    tracks = {}
    for r in reports:
        tracks.setdefault(r.id, []).append(r)
    for t in tracks.values():
        t.sort(key=lambda r: r.ts)
    return tracks
