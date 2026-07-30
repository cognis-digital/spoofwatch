"""Temporal resilient-PNT trust tracker + denial state machine (pure stdlib).

:mod:`spoofwatch.confidence` fuses the detectors into a single-epoch trust score.
Operators, though, act on *episodes*, not instants — a one-frame dip is noise, a
sustained collapse is a navigation-denial event you log, alarm, and coast
through. This module runs the trust score forward in time through a hysteretic
state machine so that transient flicker does not thrash the alarm while a real
attack is declared promptly and cleared only once genuinely recovered.

States, worst → best: ``DENIED`` → ``SUSPECT`` → ``DEGRADED`` → ``LOCKED``.
Hysteresis is dwell-based: the track only *enters* a worse state after the
smoothed trust has sat below that state's floor for ``enter_hold`` consecutive
epochs, and only *recovers* upward after sitting above the higher floor for
``exit_hold`` epochs. Trust is smoothed with an exponential moving average first
so single-sample spikes cannot trip a transition on their own. The tracker emits
a per-epoch state series, a list of **denial episodes** (start/end/duration and
the worst trust reached), and a rollup an operator can glance at: total
time-in-denial, number of episodes, and the current state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# trust floors for each state (smoothed trust >= floor qualifies for that state)
STATE_FLOORS = [
    (0.80, "LOCKED"),
    (0.60, "DEGRADED"),
    (0.35, "SUSPECT"),
    (0.0, "DENIED"),
]
STATE_ORDER = ["DENIED", "SUSPECT", "DEGRADED", "LOCKED"]

# EMA smoothing applied to raw trust before the state machine (0..1)
TRUST_ALPHA = 0.4
# consecutive epochs required to enter a worse / recover to a better state
ENTER_HOLD = 2
EXIT_HOLD = 3
# states at or below this rank count as a navigation-denial episode
DENIAL_STATES = {"DENIED", "SUSPECT"}


@dataclass
class TrustSample:
    ts: float
    trust: float                    # 0..1 (e.g. PNTConfidence.trust)


@dataclass
class TrackPoint:
    ts: float
    trust: float                    # raw
    smoothed: float
    state: str


def _instant_state(trust):
    """The state a given (smoothed) trust value falls into, ignoring hysteresis."""
    for floor, name in STATE_FLOORS:
        if trust >= floor:
            return name
    return STATE_FLOORS[-1][1]


def _rank(state):
    return STATE_ORDER.index(state)


def track(samples, alpha=TRUST_ALPHA, enter_hold=ENTER_HOLD, exit_hold=EXIT_HOLD,
          denial_states=DENIAL_STATES):
    """Run the hysteretic trust state machine over a chronological trust series.

    ``samples`` is a list of :class:`TrustSample`. Returns a dict with the
    per-epoch ``points`` (:class:`TrackPoint`), the detected denial ``episodes``,
    and a ``summary`` rollup.
    """
    samples = sorted(samples, key=lambda s: s.ts)
    points = []
    if not samples:
        return {"points": [], "episodes": [], "summary": {
            "epochs": 0, "denial_episodes": 0, "total_denial_s": 0.0,
            "min_trust": None, "final_state": None, "in_denial_now": False}}

    smoothed = samples[0].trust
    state = _instant_state(smoothed)
    # candidate-target tracking for dwell-based hysteresis
    pending = None      # (target_state, count)

    for i, s in enumerate(samples):
        if i == 0:
            smoothed = s.trust
        else:
            smoothed = (1 - alpha) * smoothed + alpha * s.trust
        target = _instant_state(smoothed)

        if target != state:
            worse = _rank(target) < _rank(state)
            hold = enter_hold if worse else exit_hold
            if pending is not None and pending[0] == target:
                pending = (target, pending[1] + 1)
            else:
                pending = (target, 1)
            if pending[1] >= hold:
                state = target
                pending = None
        else:
            pending = None

        points.append(TrackPoint(s.ts, round(s.trust, 4),
                                 round(smoothed, 4), state))

    # extract denial episodes from the committed state series
    episodes = []
    cur = None
    for p in points:
        if p.state in denial_states:
            if cur is None:
                cur = {"start_ts": p.ts, "min_trust": p.trust,
                       "worst_state": p.state, "n": 1}
            else:
                cur["min_trust"] = min(cur["min_trust"], p.trust)
                if _rank(p.state) < _rank(cur["worst_state"]):
                    cur["worst_state"] = p.state
                cur["n"] += 1
            cur["last_ts"] = p.ts
        else:
            if cur is not None:
                episodes.append(_close_episode(cur, p.ts, ongoing=False))
                cur = None
    if cur is not None:
        episodes.append(_close_episode(cur, cur["last_ts"], ongoing=True))

    total_denial = sum(e["duration_s"] for e in episodes)
    min_trust = min(p.trust for p in points)
    summary = {
        "epochs": len(points),
        "denial_episodes": len(episodes),
        "total_denial_s": round(total_denial, 3),
        "min_trust": round(min_trust, 4),
        "final_state": points[-1].state,
        "in_denial_now": points[-1].state in denial_states,
    }
    return {"points": points, "episodes": episodes, "summary": summary}


def _close_episode(cur, end_ts, ongoing):
    return {
        "start_ts": cur["start_ts"],
        "end_ts": None if ongoing else end_ts,
        "duration_s": round(cur.get("last_ts", end_ts) - cur["start_ts"], 3),
        "min_trust": round(cur["min_trust"], 4),
        "worst_state": cur["worst_state"],
        "epochs": cur["n"],
        "ongoing": bool(ongoing),
    }
