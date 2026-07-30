"""Receiver clock-bias / time-push jump detector (pure stdlib).

Spoofing and meaconing don't only move you in *space* — a time-synchronisation
attack pushes the receiver *clock*, which is catastrophic for the timing-locked
infrastructure and logistics timestamping that depend on GNSS as a clock. This
detector tracks the receiver clock bias (and, if given, drift) epoch to epoch
and flags discontinuities that no real oscillator would produce:

* **Step jump** — the bias moves far more between two epochs than the drift and
  elapsed time can explain (a sudden time-push).
* **Unphysical drift** — the epoch-to-epoch drift exceeds a sane oscillator
  bound (a slow "walk-off" meaconing pull).

Biases are in nanoseconds. Everything is relative to the receiver's own
recent behaviour, so no per-oscillator calibration is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

# a bias step this many ns beyond the drift-predicted value is a jump
STEP_JUMP_NS = 100.0
# drift beyond this magnitude (ns per second) is judged unphysical for the OCXO/TCXO class
MAX_DRIFT_NS_PER_S = 50.0


@dataclass
class ClockSample:
    ts: float                       # epoch seconds
    bias_ns: float                  # receiver clock bias (ns)
    drift_ns_s: float | None = None  # optional measured drift (ns/s)


def detect(samples, step_jump_ns=STEP_JUMP_NS, max_drift_ns_s=MAX_DRIFT_NS_PER_S):
    """Flag clock-bias step jumps and unphysical drift across a sample series.

    Returns a list of event dicts, each with ``type`` (``time_step`` or
    ``drift_anomaly``), the timestamps bracketing it, the offending magnitude,
    and a 0..1 confidence.
    """
    samples = sorted(samples, key=lambda s: s.ts)
    events = []
    for a, b in zip(samples, samples[1:]):
        dt = b.ts - a.ts
        if dt <= 0:
            continue
        # predict b's bias from a's bias + a's drift over dt (drift optional)
        drift = a.drift_ns_s if a.drift_ns_s is not None else 0.0
        predicted = a.bias_ns + drift * dt
        residual = b.bias_ns - predicted
        implied_drift = (b.bias_ns - a.bias_ns) / dt

        if abs(residual) > step_jump_ns:
            events.append({
                "type": "time_step", "ts_from": a.ts, "ts_to": b.ts,
                "jump_ns": round(residual, 2),
                "implied_drift_ns_s": round(implied_drift, 3),
                "confidence": round(min(1.0, abs(residual) / (step_jump_ns * 3)), 3),
            })
        elif abs(implied_drift) > max_drift_ns_s:
            events.append({
                "type": "drift_anomaly", "ts_from": a.ts, "ts_to": b.ts,
                "implied_drift_ns_s": round(implied_drift, 3),
                "confidence": round(min(1.0, abs(implied_drift) / (max_drift_ns_s * 3)), 3),
            })
    return events


def summarize(samples, **kw):
    events = detect(samples, **kw)
    return {
        "samples": len(samples),
        "time_steps": sum(1 for e in events if e["type"] == "time_step"),
        "drift_anomalies": sum(1 for e in events if e["type"] == "drift_anomaly"),
        "max_jump_ns": max((abs(e["jump_ns"]) for e in events
                            if e["type"] == "time_step"), default=0.0),
        "events": events,
    }
