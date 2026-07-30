"""GNSS timing-holdover error growth + coast budget (pure stdlib).

GNSS is not only a positioning system; it is the world's clock. Telecom base
stations, power grids, data centres and logistics timestamping discipline a local
oscillator to GNSS time. When the sky is jammed or spoofed and the receiver stops
trusting its fix, that oscillator must **hold over** — free-run on its own,
accumulating time error until either GNSS returns or the error breaches the
holding budget of whatever it feeds.

This module answers the operator's holdover questions with a deterministic
oscillator model. An oscillator's time error grows as

    TE(t) = TE0 + y0 * t + 0.5 * D * t^2

where ``y0`` is the fractional-frequency offset locked in at the moment of the
outage (s/s) and ``D`` is the fractional-frequency drift / aging rate (per s).
The two together give the parabolic walk-off every holdover spec is written
against. Built-in :data:`OSCILLATORS` cover the common classes (TCXO → Cesium);
you can also pass your own. From this we get the **time error at any elapsed
time**, the **holdover budget** (seconds until a threshold such as the telecom
1.5 µs mask is breached), and a **projection** across a coast window — exactly
what an operator entering a jamming zone needs to plan around.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# common telecom / timing masks (seconds of time error)
TELECOM_MASK_S = 1.5e-6            # 1.5 µs — classic mobile-backhaul holdover mask
UTC_TRACEABLE_S = 1.0e-6          # 1 µs — UTC-traceable timestamping budget


@dataclass
class Oscillator:
    name: str
    y0: float           # fractional-frequency offset at outage (s/s), magnitude
    drift_per_s: float  # fractional-frequency drift / aging (per second), magnitude


# representative (magnitude) specs for common disciplined-oscillator classes.
# y0 is the residual offset just after loss of lock; drift is aging per second.
OSCILLATORS = {
    "TCXO": Oscillator("TCXO", y0=1.0e-8, drift_per_s=1.0e-10),
    "OCXO": Oscillator("OCXO", y0=1.0e-10, drift_per_s=1.0e-12),
    "Rubidium": Oscillator("Rubidium", y0=1.0e-11, drift_per_s=1.0e-14),
    "Cesium": Oscillator("Cesium", y0=1.0e-12, drift_per_s=1.0e-16),
}


def _osc(osc):
    if isinstance(osc, Oscillator):
        return osc
    if osc in OSCILLATORS:
        return OSCILLATORS[osc]
    raise ValueError(f"unknown oscillator class {osc!r}; "
                     f"known: {sorted(OSCILLATORS)}")


def time_error_s(t_s, osc="OCXO", te0_s=0.0):
    """Accumulated time error (seconds) after ``t_s`` seconds of holdover.

    Uses the worst-case magnitude model ``TE = te0 + y0*t + 0.5*D*t^2`` (offset
    and drift taken as same-sign so errors add rather than cancel).
    """
    o = _osc(osc)
    t = max(0.0, t_s)
    return te0_s + o.y0 * t + 0.5 * o.drift_per_s * t * t


def holdover_budget_s(threshold_s=TELECOM_MASK_S, osc="OCXO", te0_s=0.0):
    """Seconds of holdover until the time error reaches ``threshold_s``.

    Solves ``0.5*D*t^2 + y0*t + (te0 - threshold) = 0`` for the positive root.
    Returns ``inf`` for an ideal (zero-offset, zero-drift) oscillator, and ``0``
    if the threshold is already breached at ``t=0``.
    """
    o = _osc(osc)
    rem = threshold_s - te0_s
    if rem <= 0:
        return 0.0
    a = 0.5 * o.drift_per_s
    b = o.y0
    c = -rem
    if a <= 0 and b <= 0:
        return float("inf")
    if a <= 0:
        return rem / b
    disc = b * b - 4 * a * c
    if disc < 0:
        return float("inf")
    return (-b + math.sqrt(disc)) / (2 * a)


def check_holdover(elapsed_s, threshold_s=TELECOM_MASK_S, osc="OCXO", te0_s=0.0):
    """Status of an in-progress holdover at ``elapsed_s`` into an outage.

    Returns a dict: current time error (s and ns), whether it is still within the
    threshold, the fraction of budget consumed, and the remaining budget in
    seconds. ``breach`` is the resilient-PNT alarm — the clock has drifted past
    what its consumers tolerate.
    """
    o = _osc(osc)
    te = time_error_s(elapsed_s, o, te0_s=te0_s)
    budget = holdover_budget_s(threshold_s, o, te0_s=te0_s)
    within = te <= threshold_s
    frac = te / threshold_s if threshold_s > 0 else float("inf")
    remaining = max(0.0, budget - elapsed_s) if math.isfinite(budget) else float("inf")
    return {
        "oscillator": o.name,
        "elapsed_s": round(elapsed_s, 3),
        "time_error_s": te,
        "time_error_ns": round(te * 1e9, 3),
        "threshold_s": threshold_s,
        "within": bool(within),
        "breach": bool(not within),
        "fraction_used": round(frac, 4) if math.isfinite(frac) else float("inf"),
        "budget_s": round(budget, 3) if math.isfinite(budget) else float("inf"),
        "remaining_s": round(remaining, 3) if math.isfinite(remaining) else float("inf"),
    }


def project(times_s, osc="OCXO", te0_s=0.0):
    """Time-error projection across a list of elapsed times (seconds).

    Returns a list of ``{"t_s", "time_error_ns"}`` rows — a holdover curve to
    plot or table for an operator planning a jamming transit.
    """
    o = _osc(osc)
    out = []
    for t in times_s:
        out.append({"t_s": round(t, 3),
                    "time_error_ns": round(time_error_s(t, o, te0_s=te0_s) * 1e9, 3)})
    return out
