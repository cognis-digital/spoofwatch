"""RAIM / ARAIM pseudorange-residual integrity monitor (pure stdlib).

An optional, deeper feed than the position-only detectors: when a receiver can
hand spoofwatch its *raw per-satellite pseudoranges* plus each satellite's ECEF
position, we recompute the navigation solution ourselves and test its internal
consistency.

Pipeline
--------
1. **Gauss-Newton least-squares fix** — solve for receiver position and a clock
   bias per constellation from the pseudoranges (see :func:`solve_position`).
2. **Global integrity test** — the residual sum of squares, normalised by the
   redundancy (degrees of freedom), is the classic RAIM test statistic. A fault
   or spoof inflates it past an integrity threshold (:func:`raim_check`).
3. **Fault exclusion / solution separation** — ARAIM-style: drop each satellite
   (or constellation subset) in turn, re-solve, and see whose removal collapses
   the residual. That satellite is the inconsistent one (:func:`araim_check`).

Legacy single-fault RAIM (one constellation, exclude one satellite) is the
degenerate case and is used as a fallback when only one constellation is
present. No NumPy; the linear algebra lives in :mod:`spoofwatch.linalg`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import linalg
from .ecef import ecef_to_geodetic

# residual RMS (metres) above which the geometry is judged inconsistent
INTEGRITY_THRESH_M = 25.0
# minimum redundancy (measurements minus unknowns) to run an integrity test
MIN_REDUNDANCY = 1


@dataclass
class Measurement:
    sat_id: str
    x: float                 # satellite ECEF X (m)
    y: float                 # satellite ECEF Y (m)
    z: float                 # satellite ECEF Z (m)
    pseudorange: float       # measured pseudorange (m)
    constellation: str = "GPS"   # GPS / GAL / GLO / BDS
    weight: float = 1.0          # optional per-measurement weight


@dataclass
class Solution:
    x: float
    y: float
    z: float
    clocks: dict             # constellation -> clock bias (m)
    residuals: list          # one per measurement, same order as input
    sse: float               # residual sum of squares
    dof: int                 # degrees of freedom (n - unknowns)
    rms: float               # sqrt(sse / dof), the RAIM test statistic (m)
    iterations: int = 0
    converged: bool = True
    sat_ids: list = field(default_factory=list)

    @property
    def lla(self):
        return ecef_to_geodetic(self.x, self.y, self.z)


def _constellations(meas):
    # deterministic ordering of the distinct constellations present
    seen = []
    for m in meas:
        if m.constellation not in seen:
            seen.append(m.constellation)
    return seen


def solve_position(measurements, x0=None, iters=12, tol=1e-4):
    """Weighted least-squares receiver fix from pseudoranges.

    Unknowns are 3 position coordinates plus one clock bias per constellation.
    Returns a :class:`Solution`. Raises ``ValueError`` on under-determined or
    degenerate geometry.
    """
    meas = list(measurements)
    consts = _constellations(meas)
    n = len(meas)
    nun = 3 + len(consts)
    if n < nun:
        raise ValueError(f"under-determined: {n} measurements < {nun} unknowns")

    cidx = {c: i for i, c in enumerate(consts)}
    # initial guess: Earth-centre-ish start biased outward, zero clocks
    if x0 is None:
        px, py, pz = 0.0, 0.0, 0.0
    else:
        px, py, pz = x0
    clocks = [0.0] * len(consts)

    it = 0
    converged = False
    for it in range(1, iters + 1):
        H = []
        r = []
        w = []
        for m in meas:
            dx, dy, dz = m.x - px, m.y - py, m.z - pz
            rng = math.sqrt(dx * dx + dy * dy + dz * dz)
            if rng < 1e-6:
                rng = 1e-6
            los = (dx / rng, dy / rng, dz / rng)   # receiver -> satellite unit
            b = clocks[cidx[m.constellation]]
            predicted = rng + b
            # design row: d(predicted)/d[px,py,pz, clock_k]
            row = [-los[0], -los[1], -los[2]] + [0.0] * len(consts)
            row[3 + cidx[m.constellation]] = 1.0
            H.append(row)
            r.append(m.pseudorange - predicted)
            w.append(max(m.weight, 1e-9))
        # weighted normal equations
        Hw = [[H[i][j] * w[i] for j in range(nun)] for i in range(n)]
        At = linalg.transpose(Hw)
        AtA = linalg.matmul(At, H)
        Atb = linalg.matvec(At, r)
        delta = linalg.solve(AtA, Atb)
        px += delta[0]; py += delta[1]; pz += delta[2]
        for k in range(len(consts)):
            clocks[k] += delta[3 + k]
        if linalg.vnorm(delta[:3]) < tol:
            converged = True
            break

    # final residuals at the solution
    resid = []
    for m in meas:
        dx, dy, dz = m.x - px, m.y - py, m.z - pz
        rng = math.sqrt(dx * dx + dy * dy + dz * dz)
        b = clocks[cidx[m.constellation]]
        resid.append(m.pseudorange - (rng + b))
    sse = sum(e * e for e in resid)
    dof = n - nun
    rms = math.sqrt(sse / dof) if dof > 0 else 0.0
    return Solution(px, py, pz, {c: clocks[cidx[c]] for c in consts},
                    resid, sse, dof, rms, it, converged,
                    [m.sat_id for m in meas])


def raim_check(measurements, thresh_m=INTEGRITY_THRESH_M, x0=None):
    """Global RAIM integrity test.

    Returns a dict: solution RMS, degrees of freedom, a boolean ``fault`` flag,
    and a 0..1 ``confidence`` that scales with how far the statistic exceeds the
    threshold. Requires redundancy (more measurements than unknowns).
    """
    sol = solve_position(measurements, x0=x0)
    available = sol.dof >= MIN_REDUNDANCY
    fault = available and sol.rms > thresh_m
    conf = 0.0
    if available and thresh_m > 0:
        conf = round(min(1.0, max(0.0, (sol.rms - thresh_m) / (thresh_m * 2))), 3)
    return {
        "fault": bool(fault),
        "rms_m": round(sol.rms, 3),
        "sse": round(sol.sse, 3),
        "dof": sol.dof,
        "threshold_m": thresh_m,
        "raim_available": available,
        "confidence": conf if fault else 0.0,
        "n_measurements": len(measurements),
        "constellations": _constellations(measurements),
    }


def araim_check(measurements, thresh_m=INTEGRITY_THRESH_M, x0=None):
    """Fault detection & exclusion via solution separation (ARAIM-style).

    If the all-in-view solution trips the integrity test, drop each satellite in
    turn; the subset whose full-set residual RMS falls back under threshold (and
    is minimised) identifies the excluded satellite(s). Multi-constellation and
    multi-fault aware — it also tries excluding a whole constellation at once
    (a coherent single-constellation spoof). Falls back to legacy single-fault
    exclusion when only one constellation is present.
    """
    base = raim_check(measurements, thresh_m=thresh_m, x0=x0)
    result = dict(base)
    result["excluded"] = []
    result["method"] = None
    if not base["fault"]:
        return result

    meas = list(measurements)
    best = None  # (rms, excluded_ids, method, cleaned_solution)

    # 1) single-satellite exclusion (needs redundancy after removal)
    for i in range(len(meas)):
        subset = meas[:i] + meas[i + 1:]
        try:
            sol = solve_position(subset, x0=x0)
        except ValueError:
            continue
        if sol.dof < MIN_REDUNDANCY:
            continue
        cand = (sol.rms, [meas[i].sat_id], "single-sat", sol)
        if best is None or cand[0] < best[0]:
            best = cand

    # 2) whole-constellation exclusion (coherent single-constellation spoof)
    consts = _constellations(meas)
    if len(consts) > 1:
        for c in consts:
            subset = [m for m in meas if m.constellation != c]
            try:
                sol = solve_position(subset, x0=x0)
            except ValueError:
                continue
            if sol.dof < MIN_REDUNDANCY:
                continue
            excl = [m.sat_id for m in meas if m.constellation == c]
            cand = (sol.rms, excl, f"constellation:{c}", sol)
            if best is None or cand[0] < best[0]:
                best = cand

    if best is not None and best[0] <= thresh_m:
        result["excluded"] = best[1]
        result["method"] = best[2]
        result["cleaned_rms_m"] = round(best[0], 3)
        result["spoof_hypothesis"] = True
    else:
        # can't isolate a single culprit -> broad / all-constellation spoof
        result["method"] = "unresolved"
        result["spoof_hypothesis"] = True
        result["cleaned_rms_m"] = round(best[0], 3) if best else None
    return result
