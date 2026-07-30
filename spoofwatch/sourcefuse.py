"""Multi-source PNT arbiter — fuse independent navigation sources, safely (stdlib).

The 2026 answer to GPS denial is not one clever sensor but *many independent
methods, continuously cross-checked, degrading safely*: GNSS, inertial, visual
odometry, terrain-relative, celestial, and RF signals-of-opportunity each give a
position with its own uncertainty. This module fuses them by inverse-variance
weighting, but first runs **fault detection and exclusion** — any source whose
fix disagrees with the consensus beyond a statistical gate is thrown out (the same
logic RAIM applies to satellites, applied at the source level) — then reports a
fused fix, a trust level, and which sources it trusted or rejected.

Where ``confidence.py``/``fusion.py`` fuse *detector verdicts* into a spoof
probability, this fuses *navigation solutions* into a position you can fly on.

Deterministic, dependency-free. Positions are local ENU meters (x, y); extend to a
full 3-D/ECEF frame as needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple


class SourceClass(str, Enum):
    GNSS = "gnss"
    INERTIAL = "inertial"
    VISUAL = "visual"            # visual / VIO odometry
    TERRAIN = "terrain"          # terrain-relative navigation
    CELESTIAL = "celestial"
    RF_SOO = "rf_soo"            # RF signals of opportunity


@dataclass(frozen=True)
class NavFix:
    source: str
    x: float
    y: float
    sigma_m: float               # 1-sigma horizontal uncertainty
    source_class: SourceClass

    def __post_init__(self) -> None:
        if self.sigma_m <= 0:
            raise ValueError("sigma_m must be > 0")


class TrustLevel(str, Enum):
    TRUSTED = "trusted"          # >= 3 independent classes agree
    DEGRADED = "degraded"        # 2 classes agree
    UNSAFE = "unsafe"            # < 2 -> coast on dead reckoning


@dataclass(frozen=True)
class FusedFix:
    x: float
    y: float
    sigma_m: float
    trust: TrustLevel
    sources_used: Tuple[str, ...]
    sources_rejected: Tuple[str, ...]
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 4), "y": round(self.y, 4),
            "sigma_m": round(self.sigma_m, 4),
            "trust": self.trust.value,
            "sources_used": list(self.sources_used),
            "sources_rejected": list(self.sources_rejected),
            "reasoning": self.reasoning,
        }


def _weighted_consensus(fixes: Sequence[NavFix]) -> Tuple[float, float, float]:
    """Inverse-variance weighted mean position and its 1-sigma."""
    w = [1.0 / (f.sigma_m ** 2) for f in fixes]
    wsum = sum(w)
    x = sum(wi * f.x for wi, f in zip(w, fixes)) / wsum
    y = sum(wi * f.y for wi, f in zip(w, fixes)) / wsum
    sigma = math.sqrt(1.0 / wsum)
    return x, y, sigma


def _normalized_residual(f: NavFix, cx: float, cy: float, csigma: float) -> float:
    d = math.hypot(f.x - cx, f.y - cy)
    combined = math.sqrt(f.sigma_m ** 2 + csigma ** 2)
    return d / combined if combined > 0 else 0.0


def fuse_sources(fixes: Sequence[NavFix], gate: float = 3.0) -> FusedFix:
    """Fuse independent nav fixes with fault detection and exclusion.

    Iteratively computes the inverse-variance consensus and excludes the single
    worst source whose normalized residual exceeds ``gate`` (in sigmas), until all
    remaining sources are mutually consistent. Trust follows how many independent
    source *classes* survive.
    """
    if not fixes:
        return FusedFix(0.0, 0.0, float("inf"), TrustLevel.UNSAFE, (), (),
                        "no navigation sources available")

    accepted: List[NavFix] = list(fixes)
    rejected: List[NavFix] = []

    while len(accepted) > 1:
        cx, cy, csigma = _weighted_consensus(accepted)
        residuals = [(f, _normalized_residual(f, cx, cy, csigma)) for f in accepted]
        worst_f, worst_r = max(residuals, key=lambda t: t[1])
        if worst_r > gate:
            accepted.remove(worst_f)
            rejected.append(worst_f)
        else:
            break

    cx, cy, csigma = _weighted_consensus(accepted)
    classes = {f.source_class for f in accepted}
    if len(classes) >= 3:
        trust = TrustLevel.TRUSTED
    elif len(classes) == 2:
        trust = TrustLevel.DEGRADED
    else:
        trust = TrustLevel.UNSAFE

    if trust is TrustLevel.UNSAFE:
        reason = (f"only {len(classes)} independent source class(es) consistent — "
                  "coast on dead reckoning")
    elif rejected:
        reason = (f"{len(rejected)} source(s) excluded as faulted; fused "
                  f"{len(accepted)} across {len(classes)} classes")
    else:
        reason = f"all {len(accepted)} sources consistent across {len(classes)} classes"

    return FusedFix(cx, cy, csigma, trust,
                    tuple(f.source for f in accepted),
                    tuple(f.source for f in rejected), reason)


@dataclass
class PntArbiter:
    """Stateful arbiter: fuse per epoch and coast uncertainty when unsafe."""

    gate: float = 3.0
    coast_growth_m_per_epoch: float = 5.0
    _last_sigma: float = field(default=0.0, init=False)
    _epochs_coasting: int = field(default=0, init=False)

    def step(self, fixes: Sequence[NavFix]) -> FusedFix:
        fused = fuse_sources(fixes, self.gate)
        if fused.trust is TrustLevel.UNSAFE:
            self._epochs_coasting += 1
            grown = (self._last_sigma
                     + self.coast_growth_m_per_epoch * self._epochs_coasting)
            fused = FusedFix(fused.x, fused.y, max(fused.sigma_m, grown)
                             if math.isfinite(fused.sigma_m) else grown,
                             fused.trust, fused.sources_used,
                             fused.sources_rejected,
                             fused.reasoning + f" (coasting {self._epochs_coasting} epochs)")
        else:
            self._epochs_coasting = 0
            self._last_sigma = fused.sigma_m
        return fused

    @property
    def coasting_epochs(self) -> int:
        return self._epochs_coasting
