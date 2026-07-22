# Changelog

Adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-07-22

Initial release.

### Added
- **Jamming detection** (`jamming.py`) — spatial single-link clustering of degraded
  navigation-integrity reports into interference zones (centroid, radius, confidence).
- **Spoofing detection** (`spoofing.py`) — kinematic teleport (impossible ground speed) +
  co-location signature (many distinct aircraft at one point).
- **Records + geo** (`records.py`, `geo.py`) — Report model, haversine, implied speed,
  JSON/CSV loaders, equirectangular grid, GeoJSON emit.
- **Orchestration + CLI** (`detect.py`, `cli.py`) — analyze, map (GeoJSON), synth, demo,
  selfcheck.
- **Verification harness** (`bench/run_all.py`) — 20 deterministic synthetic scenes with
  planted jamming + spoofing; regenerates `RESULTS.md`. Jamming detection 100%, loc err
  ~11 km; spoof recall 1.00, precision ~0.99, origin err ~0.3 km.
- 21 tests; CI across Python 3.9–3.13.
