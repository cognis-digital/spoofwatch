"""Jamming detection — cluster degraded-integrity reports into interference zones.

GPS jamming shows up as position reports whose transmitters flag low navigation
integrity (loss/degradation of fix). In clean airspace those are rare and random;
under a jammer they concentrate over a geographic area. We take every degraded
report and spatially cluster them (single-link within `cluster_km`); any cluster
with enough members is an interference zone (centroid + radius + confidence).
Robust to sparse traffic; pure stdlib.
"""

from __future__ import annotations

from .records import haversine_km

INTEGRITY_DEGRADED = 3.0    # integrity below this = degraded / lost fix
CLUSTER_KM = 42.0           # degraded reports within this link into one zone
MIN_SAMPLES = 4             # a zone needs at least this many degraded reports


def detect(reports, integ_thresh=INTEGRITY_DEGRADED, cluster_km=CLUSTER_KM,
           min_samples=MIN_SAMPLES, **_ignored):
    deg = [(r.lat, r.lon) for r in reports if r.integrity < integ_thresh]
    if not deg:
        return []

    # single-link spatial clustering (O(n^2); fine for hundreds of reports)
    n = len(deg)
    cluster_of = [-1] * n
    cid = 0
    for i in range(n):
        if cluster_of[i] != -1:
            continue
        cluster_of[i] = cid
        stack = [i]
        while stack:
            j = stack.pop()
            for k in range(n):
                if cluster_of[k] == -1 and \
                        haversine_km(deg[j][0], deg[j][1], deg[k][0], deg[k][1]) <= cluster_km:
                    cluster_of[k] = cid; stack.append(k)
        cid += 1

    zones = []
    for c in range(cid):
        pts = [deg[i] for i in range(n) if cluster_of[i] == c]
        if len(pts) < min_samples:
            continue
        clat = sum(p[0] for p in pts) / len(pts)
        clon = sum(p[1] for p in pts) / len(pts)
        radius = max(haversine_km(clat, clon, p[0], p[1]) for p in pts)
        zones.append({"center_lat": round(clat, 5), "center_lon": round(clon, 5),
                      "radius_km": round(max(radius, 5.0), 1),
                      "degraded_reports": len(pts),
                      "confidence": round(min(1.0, len(pts) / (min_samples * 3)), 2)})
    zones.sort(key=lambda z: -z["degraded_reports"])
    return zones
