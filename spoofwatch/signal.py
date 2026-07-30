"""Receiver-level C/N0 + AGC anomaly detector (pure stdlib).

The position-feed detectors see the *consequences* of interference; this one
sees its *cause* in the front end. Feed it per-epoch carrier-to-noise-density
(C/N0, dB-Hz) for each tracked satellite plus the receiver's Automatic Gain
Control (AGC) level, and it flags the two textbook signatures documented on even
low-cost off-the-shelf receivers:

* **Spoofing** — a sudden, *uniform* rise in C/N0 across all channels as the
  spoofer overpowers the genuine constellation (mean jumps up, spread stays
  tight).
* **Jamming** — an abrupt AGC drop as the front end de-sensitises against a
  strong interferer, usually with a simultaneous C/N0 collapse.

Output is a per-epoch interference score in 0..1 with a labelled event type.
Purely relative/statistical — no calibration to a specific receiver required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# spoof: mean C/N0 rises by at least this many dB-Hz between epochs, uniformly
SPOOF_CN0_RISE_DBHZ = 6.0
SPOOF_SPREAD_MAX_DBHZ = 4.0     # "uniform" = low cross-channel spread
# jam: AGC falls by at least this fraction of its running level, or C/N0 collapses
JAM_AGC_DROP_FRAC = 0.20
JAM_CN0_DROP_DBHZ = 8.0


@dataclass
class SignalEpoch:
    ts: float
    cn0: dict                       # sat_id -> C/N0 (dB-Hz)
    agc: float | None = None        # receiver AGC level (higher = more gain)


@dataclass
class SignalEvent:
    ts: float
    kind: str                       # "spoof" | "jam" | "clean"
    score: float                    # 0..1 interference score
    mean_cn0: float
    cn0_spread: float
    detail: dict = field(default_factory=dict)


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _stdev(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def analyze_epochs(epochs, spoof_rise=SPOOF_CN0_RISE_DBHZ,
                   spread_max=SPOOF_SPREAD_MAX_DBHZ,
                   agc_drop_frac=JAM_AGC_DROP_FRAC,
                   cn0_drop=JAM_CN0_DROP_DBHZ):
    """Score a chronological list of :class:`SignalEpoch` for interference.

    Returns a list of :class:`SignalEvent`, one per epoch (the first epoch is
    always ``clean`` — no prior to difference against).
    """
    epochs = sorted(epochs, key=lambda e: e.ts)
    events = []
    prev_mean = None
    prev_agc = None
    for e in epochs:
        vals = list(e.cn0.values())
        m = _mean(vals)
        spread = _stdev(vals)
        kind = "clean"
        score = 0.0
        detail = {}

        if prev_mean is not None:
            d_cn0 = m - prev_mean
            # --- spoof: uniform rise ---
            if d_cn0 >= spoof_rise and spread <= spread_max:
                kind = "spoof"
                score = min(1.0, (d_cn0 / spoof_rise) * (1.0 - spread / (spread_max * 2)))
                detail["cn0_rise_dbhz"] = round(d_cn0, 2)
            # --- jam: AGC drop and/or C/N0 collapse ---
            agc_drop = False
            if e.agc is not None and prev_agc is not None and prev_agc > 0:
                frac = (prev_agc - e.agc) / prev_agc
                if frac >= agc_drop_frac:
                    agc_drop = True
                    detail["agc_drop_frac"] = round(frac, 3)
            cn0_collapse = d_cn0 <= -cn0_drop
            if agc_drop or cn0_collapse:
                kind = "jam"
                s_agc = detail.get("agc_drop_frac", 0.0) / max(agc_drop_frac, 1e-9)
                s_cn0 = (-d_cn0) / cn0_drop if cn0_collapse else 0.0
                score = min(1.0, max(s_agc, s_cn0))
                if cn0_collapse:
                    detail["cn0_drop_dbhz"] = round(-d_cn0, 2)

        events.append(SignalEvent(e.ts, kind, round(score, 3), round(m, 2),
                                  round(spread, 2), detail))
        prev_mean = m
        if e.agc is not None:
            prev_agc = e.agc
    return events


def summarize(events):
    """Roll a per-epoch event list into counts + a peak score per event type."""
    out = {"epochs": len(events), "spoof_epochs": 0, "jam_epochs": 0,
           "clean_epochs": 0, "peak_spoof_score": 0.0, "peak_jam_score": 0.0}
    for e in events:
        out[f"{e.kind}_epochs"] += 1
        if e.kind == "spoof":
            out["peak_spoof_score"] = max(out["peak_spoof_score"], e.score)
        elif e.kind == "jam":
            out["peak_jam_score"] = max(out["peak_jam_score"], e.score)
    return out
