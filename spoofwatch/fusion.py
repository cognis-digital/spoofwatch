"""Crowdsourced / multi-sensor fusion aggregator (pure stdlib).

No single receiver is trustworthy on its own — one platform's "jamming zone"
could be a local antenna fault. Fusion merges interference detections from many
*independent* sensors into one confidence-weighted map, and applies **spatial
voting**: a zone corroborated by several independent sensors outranks a
single-sensor artifact. This supports the federated/crowdsourced monitoring
trend while staying local-first — you run the aggregator on your own box over
reports your fleet hands you; no vendor cloud.
"""

from __future__ import annotations

from dataclasses import dataclass

from .records import haversine_km

# detections within this distance fuse into one zone
CLUSTER_KM = 40.0
# a fused zone needs corroboration from at least this many distinct sensors to rank "confirmed"
MIN_SENSORS = 2


@dataclass
class SensorDetection:
    sensor_id: str
    lat: float
    lon: float
    radius_km: float = 20.0
    confidence: float = 0.5
    kind: str = "jamming"


def fuse(detections, cluster_km=CLUSTER_KM, min_sensors=MIN_SENSORS):
    """Fuse independent sensor detections into corroborated interference zones.

    Single-link clusters detections by proximity, then scores each cluster by the
    number of *distinct* sensors backing it (spatial voting) and their combined
    confidence. Returns zones sorted so multi-sensor corroboration ranks first.
    """
    dets = list(detections)
    n = len(dets)
    if n == 0:
        return []

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
                        haversine_km(dets[j].lat, dets[j].lon,
                                     dets[k].lat, dets[k].lon) <= cluster_km:
                    cluster_of[k] = cid
                    stack.append(k)
        cid += 1

    zones = []
    for c in range(cid):
        members = [dets[i] for i in range(n) if cluster_of[i] == c]
        sensors = {m.sensor_id for m in members}
        # confidence-weighted centroid
        wsum = sum(m.confidence for m in members) or 1e-9
        clat = sum(m.lat * m.confidence for m in members) / wsum
        clon = sum(m.lon * m.confidence for m in members) / wsum
        # combined confidence via independent-evidence OR, boosted by sensor count
        prod_miss = 1.0
        for m in members:
            prod_miss *= (1.0 - max(0.0, min(1.0, m.confidence)))
        combined = 1.0 - prod_miss
        radius = max(m.radius_km for m in members)
        zones.append({
            "center_lat": round(clat, 5),
            "center_lon": round(clon, 5),
            "radius_km": round(radius, 1),
            "sensor_count": len(sensors),
            "detections": len(members),
            "sensors": sorted(sensors),
            "confirmed": len(sensors) >= min_sensors,
            "confidence": round(combined, 3),
            "kind": members[0].kind,
        })
    # rank: confirmed first, then by sensor count, then confidence
    zones.sort(key=lambda z: (not z["confirmed"], -z["sensor_count"], -z["confidence"]))
    return zones
