"""Resilient-PNT confidence integrator (pure stdlib).

Each spoofwatch detector looks at one facet of the problem — RAIM residuals,
C/N0 and AGC, multi-constellation divergence, clock jumps, angle-of-arrival,
Doppler, sky geometry, Kalman innovation gating. This module fuses their
individual verdicts into a single **resilient-PNT trust score** in 0..1, the
number an autonomy stack or operator can actually act on: *how much should I
trust my position right now, and what is my protection level?*

Evidence is combined with a weighted **noisy-OR** — independent channels each
push the spoof probability up, a reliable channel more than a marginal one, and
no single channel can be overruled into silence. The trust score is
``1 − P(spoof)``; a categorical trust level and an inflated protection-level
bound come along for free. Everything is deterministic and dependency-free, so
the confidence you compute offline is the confidence you would compute anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# per-channel reliability weight in 0..1 (how much a full detection moves P(spoof))
DEFAULT_WEIGHTS = {
    "raim": 1.0,
    "signal": 0.9,
    "constellation": 1.0,
    "clockbias": 0.8,
    "aoa": 1.0,
    "doppler": 1.0,
    "skygeom": 0.7,
    "kalman": 0.8,
    "teleport": 0.9,
    "colocation": 0.9,
    "meaconing": 0.9,
    "inertial": 0.8,
}

# trust-score band edges -> categorical level
TRUST_BANDS = [
    (0.80, "trusted"),
    (0.60, "degraded"),
    (0.35, "suspect"),
    (0.0, "denied"),
]

# protection level: nominal bound (m) inflated as trust falls
NOMINAL_PL_M = 15.0
PL_INFLATION = 6.0          # metres of extra bound per unit spoof probability, scaled


def _clip(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


@dataclass
class PNTConfidence:
    trust: float                     # 0..1, higher = safer
    spoof_probability: float         # 0..1
    level: str                       # trusted | degraded | suspect | denied
    protection_level_m: float
    contributions: dict = field(default_factory=dict)
    channels: list = field(default_factory=list)
    top_channel: str = ""


def _level(trust):
    for edge, name in TRUST_BANDS:
        if trust >= edge:
            return name
    return TRUST_BANDS[-1][1]


def score(evidence, weights=None, nominal_pl_m=NOMINAL_PL_M):
    """Fuse per-channel spoof evidence into a resilient-PNT confidence.

    ``evidence`` maps a channel name to a 0..1 probability that *that* channel
    considers PNT under attack (e.g. a detector's ``confidence`` field). Unknown
    channels default to weight 1.0. Returns a :class:`PNTConfidence`.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    contributions = {}
    miss = 1.0                       # P(no channel fired), for noisy-OR
    for name, prob in evidence.items():
        p = _clip(float(prob))
        weight = _clip(float(w.get(name, 1.0)))
        eff = p * weight             # weighted evidence this channel contributes
        contributions[name] = round(eff, 4)
        miss *= (1.0 - eff)

    spoof_p = _clip(1.0 - miss)
    trust = round(1.0 - spoof_p, 4)
    spoof_p = round(spoof_p, 4)
    level = _level(trust)
    pl = round(nominal_pl_m * (1.0 + PL_INFLATION * spoof_p), 2)
    top = max(contributions.items(), key=lambda kv: kv[1])[0] if contributions else ""

    return PNTConfidence(
        trust=trust,
        spoof_probability=spoof_p,
        level=level,
        protection_level_m=pl,
        contributions=contributions,
        channels=sorted(evidence.keys()),
        top_channel=top,
    )


def _get(obj, key, default=0.0):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def from_detectors(raim=None, signal_summary=None, constellation=None,
                   clockbias=None, aoa=None, doppler=None, skygeom=None,
                   kalman_events=None, teleport_rate=None, colocation=None,
                   weights=None, nominal_pl_m=NOMINAL_PL_M):
    """Build the evidence map from raw detector outputs, then :func:`score` it.

    Every argument is optional — pass whatever channels you actually ran. Each is
    reduced to a 0..1 channel probability:

    * ``raim`` — a :func:`spoofwatch.raim.raim_check` / ``araim_check`` dict.
    * ``signal_summary`` — a :func:`spoofwatch.signal.summarize` dict.
    * ``constellation`` — a :func:`spoofwatch.constellation.cross_check` dict.
    * ``clockbias`` — a :func:`spoofwatch.clockbias.summarize`/``detect`` dict.
    * ``aoa`` — a :func:`spoofwatch.aoa.check` dict.
    * ``doppler`` — a :class:`spoofwatch.doppler.DopplerResult`.
    * ``skygeom`` — a :func:`spoofwatch.skygeom.check` dict.
    * ``kalman_events`` — the event list from :func:`spoofwatch.kalman.run`
      (fraction of gated/rejected fixes becomes the channel probability).
    * ``teleport_rate`` — a 0..1 fraction of kinematically-impossible steps.
    * ``colocation`` — a 0..1 co-location-spoof confidence.
    """
    ev = {}

    if raim is not None:
        ev["raim"] = _clip(_get(raim, "confidence")) if _get(raim, "fault") else 0.0

    if signal_summary is not None:
        ev["signal"] = _clip(max(_get(signal_summary, "peak_spoof_score"),
                                 _get(signal_summary, "peak_jam_score")))

    if constellation is not None:
        # spoofwatch.constellation.cross_check dict uses the "divergence" flag
        ev["constellation"] = _clip(_get(constellation, "confidence")) \
            if _get(constellation, "divergence") else 0.0

    if clockbias is not None:
        # accept a summarize() dict (counts + events) or a raw detect() list
        if isinstance(clockbias, list):
            events = clockbias
        else:
            events = _get(clockbias, "events", []) or []
        conf = max((_get(e, "confidence") for e in events), default=0.0)
        ev["clockbias"] = _clip(conf)

    if aoa is not None:
        ev["aoa"] = _clip(_get(aoa, "confidence")) if _get(aoa, "single_source") else 0.0

    if doppler is not None:
        fired = _get(doppler, "inconsistent") or _get(doppler, "static_spoofer")
        ev["doppler"] = _clip(_get(doppler, "confidence")) if fired else 0.0

    if skygeom is not None:
        ev["skygeom"] = _clip(_get(skygeom, "confidence")) if _get(skygeom, "suspect") else 0.0

    if kalman_events is not None:
        evs = list(kalman_events)
        if evs:
            rejected = sum(1 for e in evs if not _get(e, "accepted", True))
            ev["kalman"] = _clip(rejected / len(evs))

    if teleport_rate is not None:
        ev["teleport"] = _clip(float(teleport_rate))

    if colocation is not None:
        ev["colocation"] = _clip(float(colocation))

    return score(ev, weights=weights, nominal_pl_m=nominal_pl_m)
