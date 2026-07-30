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
__all__ = [
    "records", "synth", "jamming", "spoofing", "detect", "geo", "cli",
    # additive detector + resilient-PNT modules (v0.2 build-out)
    "linalg", "ecef", "raim", "signal", "osnma", "clockbias", "inertial",
    "constellation", "meaconing", "zones", "classify", "fusion", "altpnt", "aoa",
    # additive PNT-resilience modules (v0.3 build-out)
    "doppler", "kalman", "skygeom", "confidence",
    # additive receiver-integrity + resilient-timing modules (v0.4 build-out)
    "cn0track", "noisefloor", "holdover", "codecarrier", "pnttrack",
    # additive GNSS-denied navigation module (v0.5 build-out)
    "terrainnav",
    # additive signal-quality / reflection-awareness module (v0.6 build-out)
    "multipath",
    # additive multi-antenna (CRPA) spatial spoofing detector (v0.7 build-out)
    "crpa",
    # additive GNSS dataset registry + C/N0 plausibility gate (v0.8 build-out)
    "datasets",
]
