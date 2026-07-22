<h1 align="center">Spoofwatch</h1>
<p align="center"><i>Detect &amp; map GPS/GNSS jamming and spoofing from open ADS-B / AIS position feeds — classical, deterministic, zero-dependency, offline.</i></p>
<p align="center">Part of the Cognis Neural Suite · <a href="https://cognis.digital">cognis.digital</a></p>

---

GPS interference is now a daily fact of life over the Baltic, Black Sea, and Middle East —
aircraft and ships lose their fix (**jamming**) or get shoved to a false position
(**spoofing**). Spoofwatch takes the position reports those platforms already broadcast and
turns them into a **map of where navigation is being attacked** — no special hardware, no
cloud, no dependencies.

## What it finds
- **Jamming zones** — GPS jamming makes transmitters flag low navigation integrity
  (loss of fix). In clean airspace those reports are rare and random; under a jammer they
  concentrate. Spoofwatch clusters degraded-integrity reports into zones with a centroid,
  radius, and confidence.
- **Spoofing** — two robust signatures:
  - **Teleport** — consecutive fixes imply a kinematically impossible ground speed.
  - **Co-location** — many *distinct* aircraft reporting the *same* point at the same time
    (the classic "everyone's suddenly at the airport" spoof).
- **A map** — jamming zones and spoof origins as **GeoJSON** you drop straight onto a map.

## Results (reproducible)
`python bench/run_all.py` — 20 deterministic synthetic scenes (Baltic-scale traffic) with
planted jamming zones + spoofing, scored against ground truth:

| Metric | Result |
|---|---|
| Jamming-zone detection rate | **100%** |
| Jamming-zone localization error (mean) | **~11 km** |
| Spoofed-aircraft recall | **1.00** |
| Spoofed-aircraft precision | **~0.99** |
| Spoof-origin localization error (mean) | **~0.3 km** |

## Quick start
```bash
python -m spoofwatch demo                       # synth a scene, detect, summarize
python -m spoofwatch synth --seed 7 --out feed.json
python -m spoofwatch analyze feed.json          # jamming zones + spoof events (JSON)
python -m spoofwatch map feed.json --out interference.geojson
python -m spoofwatch selfcheck                  # runs the benchmark, asserts metrics pass
```

**Feed format** — JSON or CSV of position reports: `id, ts, lat, lon, integrity` (0–10, low =
degraded fix, à la ADS-B NIC/NACp), optional `speed_kt`. Point it at your own ADS-B/AIS
export; the detection is source-agnostic.

## Use cases
Contested-navigation situational awareness · flight-safety and maritime-safety monitoring ·
force protection · OSINT interference mapping — **without** sending feeds to a vendor cloud.

## Install
```bash
git clone https://github.com/cognis-digital/spoofwatch && cd spoofwatch
python -m pytest -q          # 21 tests
python -m spoofwatch demo
```

## Honest scope
Synthetic benchmarks measure **algorithm correctness on known ground truth**, not fielded
accuracy on live feeds. Real ADS-B/AIS adds coverage gaps, receiver artifacts, and
legitimate low-integrity reports; jamming and spoofing also co-occur. See
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). It's a strong, transparent, self-hostable
baseline you can read and extend, not a black box.

---
<p align="center">© 2026 Cognis Digital LLC · <a href="https://cognis.digital">cognis.digital</a> · COCL-1.0</p>
