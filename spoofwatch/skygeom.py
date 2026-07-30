"""Sky-geometry / Dilution-of-Precision monitor (pure stdlib).

The *arrangement* of satellites across the sky determines how position error
amplifies measurement error — the classic Dilution of Precision (DOP). A genuine
constellation is spread over the hemisphere and yields small DOP; a spoofer
transmitting every counterfeit "satellite" from one antenna forces their claimed
azimuth/elevation into a narrow band, collapsing the geometry matrix toward
rank-deficiency and blowing DOP up. This module computes the standard DOP family
and flags that degenerate-geometry signature — a complement to
:mod:`spoofwatch.aoa` (which needs true direction-of-arrival hardware): here we
only need the azimuth/elevation each satellite *claims*.

The geometry (design) matrix ``G`` has one row per satellite,
``[-e, -n, -u, 1]``, where ``(e, n, u)`` is the local East/North/Up line-of-sight
unit vector and the trailing ``1`` is the receiver-clock partial. With
``Q = (GᵀG)⁻¹``:

* **GDOP** = √tr(Q) — geometric (all four)   * **PDOP** = √(Q₀₀+Q₁₁+Q₂₂) — position
* **HDOP** = √(Q₀₀+Q₁₁) — horizontal         * **VDOP** = √Q₂₂ — vertical
* **TDOP** = √Q₃₃ — time/clock

Linear algebra via :mod:`spoofwatch.linalg`; no NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import linalg

# GDOP above this indicates poor / suspiciously degenerate geometry
GDOP_SUSPECT = 6.0
GDOP_SEVERE = 20.0
# minimum satellites for a 4-unknown DOP solution
MIN_SATS = 4
# elevation/azimuth diversity floors: a genuine sky spreads wider than these
MIN_EL_SPREAD_DEG = 10.0
MIN_AZ_SPREAD_DEG = 30.0


@dataclass
class SatGeom:
    sat_id: str
    az_deg: float           # azimuth, degrees clockwise from North
    el_deg: float           # elevation, degrees above horizon
    constellation: str = "GPS"


def enu_los(az_deg, el_deg):
    """Local East/North/Up unit line-of-sight vector for an az/el pair."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    ce = math.cos(el)
    return (ce * math.sin(az), ce * math.cos(az), math.sin(el))


def geometry_matrix(sats):
    """Build the ``n×4`` design matrix ``[-e, -n, -u, 1]`` for the satellite set."""
    G = []
    for s in sats:
        e, n, u = enu_los(s.az_deg, s.el_deg)
        G.append([-e, -n, -u, 1.0])
    return G


def _inv4(A):
    # invert a 4x4 by solving A X = I column by column (raises on singular)
    n = len(A)
    cols = []
    for j in range(n):
        b = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(linalg.solve(A, b))
    # cols[j] is the j-th column of the inverse -> transpose into rows
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def dop(sats, min_sats=MIN_SATS):
    """Compute the DOP family for a satellite geometry.

    Returns a dict of GDOP/PDOP/HDOP/VDOP/TDOP plus ``available`` and
    ``degenerate`` flags. Rank-deficient geometry (all signals from one
    direction) is reported as ``degenerate`` with infinite DOP rather than
    raising.
    """
    sats = list(sats)
    if len(sats) < min_sats:
        return {"available": False, "n_sats": len(sats),
                "reason": f"need >= {min_sats} satellites"}
    G = geometry_matrix(sats)
    GtG = linalg.matmul(linalg.transpose(G), G)
    try:
        Q = _inv4(GtG)
    except ValueError:
        return {"available": True, "n_sats": len(sats), "degenerate": True,
                "gdop": float("inf"), "pdop": float("inf"),
                "hdop": float("inf"), "vdop": float("inf"),
                "tdop": float("inf")}
    q = [Q[i][i] for i in range(4)]
    q = [max(0.0, v) for v in q]        # guard tiny negative round-off
    gdop = math.sqrt(sum(q))
    pdop = math.sqrt(q[0] + q[1] + q[2])
    hdop = math.sqrt(q[0] + q[1])
    vdop = math.sqrt(q[2])
    tdop = math.sqrt(q[3])
    return {
        "available": True,
        "n_sats": len(sats),
        "degenerate": False,
        "gdop": round(gdop, 3),
        "pdop": round(pdop, 3),
        "hdop": round(hdop, 3),
        "vdop": round(vdop, 3),
        "tdop": round(tdop, 3),
    }


def _spread_deg(vals):
    return (max(vals) - min(vals)) if vals else 0.0


def _az_spread_deg(azs):
    """Angular spread of azimuths accounting for the 0/360 wrap (0..180)."""
    if len(azs) < 2:
        return 0.0
    rad = [math.radians(a) for a in azs]
    sx = sum(math.cos(a) for a in rad)
    sy = sum(math.sin(a) for a in rad)
    mag = math.hypot(sx, sy) / len(rad)
    # mean resultant length -> circular spread; map to 0..180 degrees
    return math.degrees(math.acos(max(-1.0, min(1.0, mag)))) * 2.0


def check(sats, gdop_suspect=GDOP_SUSPECT, gdop_severe=GDOP_SEVERE,
          min_sats=MIN_SATS):
    """Flag degenerate / single-source sky geometry.

    Combines a high-DOP test with an azimuth/elevation-diversity test: a genuine
    constellation spreads across the sky, a single-source spoofer does not.
    Returns the DOP dict augmented with ``suspect`` / ``degenerate_geometry``
    flags and a 0..1 confidence.
    """
    sats = list(sats)
    d = dop(sats, min_sats=min_sats)
    if not d.get("available"):
        d["suspect"] = False
        return d

    els = [s.el_deg for s in sats]
    azs = [s.az_deg for s in sats]
    el_spread = _spread_deg(els)
    az_spread = _az_spread_deg(azs)
    low_diversity = el_spread < MIN_EL_SPREAD_DEG and az_spread < MIN_AZ_SPREAD_DEG

    gdop = d["gdop"]
    degenerate = d.get("degenerate", False) or gdop >= gdop_severe
    suspect = degenerate or gdop >= gdop_suspect or low_diversity

    conf = 0.0
    if math.isinf(gdop):
        conf = 1.0
    elif gdop >= gdop_suspect:
        conf = min(1.0, (gdop - gdop_suspect) / (gdop_severe - gdop_suspect))
    if low_diversity:
        conf = max(conf, 0.5)
    d.update({
        "el_spread_deg": round(el_spread, 2),
        "az_spread_deg": round(az_spread, 2),
        "low_diversity": bool(low_diversity),
        "degenerate_geometry": bool(degenerate),
        "suspect": bool(suspect),
        "confidence": round(max(0.0, min(1.0, conf)), 3),
    })
    return d
