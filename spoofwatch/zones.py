"""Temporal interference-zone tracking + GeoJSON time-series (pure stdlib).

A single analysis pass gives you a snapshot of jamming/spoofing zones; operators
planning routes through contested navigation need to see interference *appear,
move, grow, and dissipate*. This module persists zones across successive
analysis epochs, associates each new detection with the nearest existing track
(nearest-neighbour within a gate), and maintains per-track history: centroid
drift, radius growth/shrink, first/last-seen, and lifetime.

It emits a timestamped GeoJSON ``FeatureCollection`` — each track a Point with
its trajectory and lifecycle in the properties — plus a simple alert stream of
newly-appeared and newly-dissipated zones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .records import haversine_km

# a new zone within this distance of a track's last centroid continues it
ASSOCIATION_KM = 60.0
# a track not updated for this many epochs is considered dissipated
STALE_EPOCHS = 2


@dataclass
class ZoneTrack:
    track_id: int
    first_ts: float
    last_ts: float
    lat: float
    lon: float
    radius_km: float
    history: list = field(default_factory=list)   # (ts, lat, lon, radius, reports)
    last_epoch: int = 0
    total_reports: int = 0

    @property
    def lifetime_s(self):
        return self.last_ts - self.first_ts

    @property
    def centroid_drift_km(self):
        if len(self.history) < 2:
            return 0.0
        f = self.history[0]
        l = self.history[-1]
        return haversine_km(f[1], f[2], l[1], l[2])

    @property
    def radius_growth_km(self):
        if len(self.history) < 2:
            return 0.0
        return self.history[-1][3] - self.history[0][3]


class ZoneTracker:
    """Associate per-epoch jamming zones into persistent tracks over time."""

    def __init__(self, association_km=ASSOCIATION_KM, stale_epochs=STALE_EPOCHS):
        self.association_km = association_km
        self.stale_epochs = stale_epochs
        self.tracks = []
        self._next_id = 0
        self._epoch = 0
        self.alerts = []

    def update(self, ts, zones):
        """Ingest one epoch's zones (as from :func:`jamming.detect`).

        Returns the list of alert dicts generated *this* epoch (appeared /
        dissipated). Zones are matched greedily to the nearest live track.
        """
        self._epoch += 1
        epoch_alerts = []
        live = [t for t in self.tracks
                if self._epoch - t.last_epoch <= self.stale_epochs]
        matched = set()
        for z in zones:
            zlat = z["center_lat"]; zlon = z["center_lon"]
            best = None
            best_d = self.association_km
            for t in live:
                if id(t) in matched:
                    continue
                d = haversine_km(t.lat, t.lon, zlat, zlon)
                if d <= best_d:
                    best_d = d
                    best = t
            if best is None:
                t = ZoneTrack(self._next_id, ts, ts, zlat, zlon,
                              z["radius_km"], last_epoch=self._epoch)
                self._next_id += 1
                self.tracks.append(t)
                best = t
                epoch_alerts.append({"event": "appeared", "track_id": t.track_id,
                                     "ts": ts, "lat": zlat, "lon": zlon})
            else:
                best.lat = zlat; best.lon = zlon
                best.radius_km = z["radius_km"]
                best.last_ts = ts
                best.last_epoch = self._epoch
            matched.add(id(best))
            best.history.append((ts, zlat, zlon, z["radius_km"],
                                 z.get("degraded_reports", 0)))
            best.total_reports += z.get("degraded_reports", 0)

        # dissipation alerts: tracks that just went stale this epoch
        for t in self.tracks:
            age = self._epoch - t.last_epoch
            if age == self.stale_epochs + 1:
                epoch_alerts.append({"event": "dissipated", "track_id": t.track_id,
                                     "ts": ts, "lat": t.lat, "lon": t.lon,
                                     "lifetime_s": t.lifetime_s})
        self.alerts.extend(epoch_alerts)
        return epoch_alerts

    def active_tracks(self):
        return [t for t in self.tracks
                if self._epoch - t.last_epoch <= self.stale_epochs]

    def to_geojson(self):
        feats = []
        for t in self.tracks:
            coords = [[round(h[2], 5), round(h[1], 5)] for h in t.history]
            feats.append({
                "type": "Feature",
                "properties": {
                    "kind": "interference_track",
                    "track_id": t.track_id,
                    "first_ts": t.first_ts,
                    "last_ts": t.last_ts,
                    "lifetime_s": round(t.lifetime_s, 1),
                    "radius_km": t.radius_km,
                    "radius_growth_km": round(t.radius_growth_km, 2),
                    "centroid_drift_km": round(t.centroid_drift_km, 2),
                    "epochs_seen": len(t.history),
                    "total_reports": t.total_reports,
                    "trajectory": coords,
                    "active": self._epoch - t.last_epoch <= self.stale_epochs,
                },
                "geometry": {"type": "Point",
                             "coordinates": [round(t.lon, 5), round(t.lat, 5)]},
            })
        return {"type": "FeatureCollection", "features": feats}
