"""Spoofwatch — detect & map GNSS (GPS) jamming and spoofing from open position feeds.

Feed it ADS-B / AIS position reports and it finds GPS **jamming** zones (spatial
clusters of degraded navigation-integrity reports) and **spoofing** events
(kinematically impossible teleports + the co-location "everyone at one point"
signature), then maps interference zones and spoof origins as GeoJSON.

Classical, deterministic, pure stdlib. Runs offline on hardware you own.
By Cognis Digital — situational awareness for a contested-navigation world.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["records", "synth", "jamming", "spoofing", "detect", "geo", "cli"]
