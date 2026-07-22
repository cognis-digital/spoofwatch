"""Equirectangular grid + GeoJSON emit (pure stdlib). Fine at regional scale."""

from __future__ import annotations

import math


class Grid:
    def __init__(self, lat0, lon0, cell_km=20.0, mid_lat=None):
        self.lat0 = lat0
        self.lon0 = lon0
        self.cell_km = cell_km
        ml = mid_lat if mid_lat is not None else lat0
        self.dlat = cell_km / 111.0
        self.dlon = cell_km / (111.0 * max(0.1, math.cos(math.radians(ml))))

    def cell(self, lat, lon):
        return (int((lon - self.lon0) / self.dlon), int((lat - self.lat0) / self.dlat))

    def center(self, ix, iy):
        return (self.lat0 + (iy + 0.5) * self.dlat, self.lon0 + (ix + 0.5) * self.dlon)


def geojson(zones, spoofs):
    feats = []
    for z in zones:
        feats.append({"type": "Feature",
                      "properties": {"kind": "jamming_zone", "confidence": z["confidence"],
                                     "radius_km": z["radius_km"], "degraded_reports": z["degraded_reports"]},
                      "geometry": {"type": "Point", "coordinates": [round(z["center_lon"], 5),
                                                                    round(z["center_lat"], 5)]}})
    for s in spoofs:
        feats.append({"type": "Feature",
                      "properties": {"kind": f"spoof_{s['type']}", "confidence": s["confidence"],
                                     "aircraft": s["n"]},
                      "geometry": {"type": "Point", "coordinates": [round(s["point"][1], 5),
                                                                    round(s["point"][0], 5)]}})
    return {"type": "FeatureCollection", "features": feats}
