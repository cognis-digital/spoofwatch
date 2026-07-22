"""Deterministic synthetic airspace with planted GNSS interference + ground truth.

Normal traffic crosses a region emitting high-integrity position reports at
plausible speeds. Into it we plant:
  * a JAMMING zone — a disk where transiting aircraft report degraded integrity
    (the loss-of-fix signature), and
  * SPOOFING — (a) co-location: several aircraft snapped to one false point (the
    classic "everyone's at the airport" signature), and (b) a teleport jump
    (kinematically impossible).
Ground truth (zones, spoof points, tagged reports) lets the benchmark score
detection precision/recall and localization error. Synthetic — measures
algorithm correctness, not fielded accuracy.
"""

from __future__ import annotations

import math
import random

from .records import Report

# default region ~ Baltic (a real GNSS-interference hotspot)
LAT0, LAT1 = 54.0, 58.0
LON0, LON1 = 18.0, 26.0


def _move(lat, lon, hdg_deg, km):
    dlat = (km / 111.0) * math.cos(math.radians(hdg_deg))
    dlon = (km / (111.0 * math.cos(math.radians(lat)))) * math.sin(math.radians(hdg_deg))
    return lat + dlat, lon + dlon


def generate(seed: int = 7, n_aircraft: int = 75, fixes: int = 12, step_s: float = 60.0,
             with_jamming: bool = True, with_spoof: bool = True):
    rng = random.Random(seed)
    reports = []
    truth = {"jamming_zones": [], "spoof_points": [], "spoofed_ids": set(),
             "jammed_keys": set()}

    jz = None
    if with_jamming:
        jz = (rng.uniform(LAT0 + 1, LAT1 - 1), rng.uniform(LON0 + 1, LON1 - 1),
              rng.uniform(30, 55))   # lat, lon, radius km
        truth["jamming_zones"].append(jz)

    spoof_pt = None
    if with_spoof:
        spoof_pt = (rng.uniform(LAT0 + 1, LAT1 - 1), rng.uniform(LON0 + 1, LON1 - 1))
        truth["spoof_points"].append(spoof_pt)
    spoof_ids = set()

    t0 = 1_700_000_000.0
    for a in range(n_aircraft):
        acid = f"AC{a:03d}"
        lat = rng.uniform(LAT0, LAT1); lon = rng.uniform(LON0, LON1)
        hdg = rng.uniform(0, 360); speed_kt = rng.uniform(380, 520)
        km_per_step = speed_kt * 1.852 * (step_s / 3600.0)
        # a subset of aircraft near the spoof point get co-located
        co_spoof = with_spoof and rng.random() < 0.18
        if co_spoof:
            spoof_ids.add(acid); truth["spoofed_ids"].add(acid)
        for k in range(fixes):
            ts = t0 + k * step_s
            lat, lon = _move(lat, lon, hdg, km_per_step)
            integ = rng.uniform(8.5, 10.0)
            rlat, rlon = lat, lon
            # jamming: degrade integrity inside the zone
            if jz and _within(lat, lon, jz[0], jz[1], jz[2]):
                integ = rng.uniform(0.0, 2.5)
                truth["jammed_keys"].add((acid, round(ts, 1)))
            # co-location spoof: snap position to the false point
            if co_spoof and k >= fixes // 2:
                rlat = spoof_pt[0] + rng.uniform(-0.02, 0.02)
                rlon = spoof_pt[1] + rng.uniform(-0.02, 0.02)
            reports.append(Report(acid, ts, rlat, rlon, round(integ, 2), round(speed_kt, 1)))

    # one clean teleport-spoof track (kinematically impossible jump)
    if with_spoof:
        tid = "AC-TP1"; truth["spoofed_ids"].add(tid)
        lat = rng.uniform(LAT0, LAT1); lon = rng.uniform(LON0, LON1)
        for k in range(fixes):
            ts = t0 + k * step_s
            if k == fixes // 2:               # sudden teleport
                lat += rng.choice([-3, 3]); lon += rng.choice([-4, 4])
            else:
                lat, lon = _move(lat, lon, 90, 450 * 1.852 * step_s / 3600.0)
            reports.append(Report(tid, ts, lat, lon, 9.0, 450.0))

    rng.shuffle(reports)
    return reports, truth


def _within(lat, lon, clat, clon, r_km):
    from .records import haversine_km
    return haversine_km(lat, lon, clat, clon) <= r_km
