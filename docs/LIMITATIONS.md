# Limitations & honest scope

Spoofwatch is a transparent, self-hostable **baseline** — classical detection you can read
and extend — not a certified navigation-warfare system. Stated plainly:

- **Synthetic benchmark ≠ fielded accuracy.** `RESULTS.md` scores on *planted* ground
  truth. Live ADS-B/AIS is messier.
- **Depends on the integrity flag.** Jamming detection uses transmitter-reported navigation
  integrity (NIC/NACp-style). Feeds without it, or platforms that keep transmitting a stale
  fix at nominal integrity, weaken the jamming signal (spoofing detection does not need it).
- **Coverage gaps & receiver artifacts.** Ground-station gaps and multipath produce
  low-integrity or discontinuous reports that can mimic jamming; a real deployment needs a
  baseline model of normal coverage.
- **Legitimate low-integrity reports exist** (initialization, maneuver, terrain). The
  `min_samples` clustering floor suppresses isolated ones but a busy anomaly can still form
  a low-confidence zone — treat confidence, not presence, as the signal.
- **Jamming + spoofing co-occur.** An aircraft physically inside a jammer but spoofed to a
  false point yields a degraded report *at the false location* — a realistic but confounding
  case that can place a small spurious zone near a spoof origin.
- **Teleport threshold is aircraft-oriented** (>1200 kt). Very high-altitude or non-aircraft
  platforms need a different bound.
- **Equirectangular geo** is fine regionally; not for global-scale or polar work.

## Sensible extensions (kept out to stay zero-dependency)
- Baseline coverage/AGC modeling to separate gaps from jamming.
- Cross-platform corroboration (multiple receivers, RF sensors).
- Time-series zone tracking (onset, growth, movement of a jammer).
- Live ADS-B/AIS ingestion adapters.
