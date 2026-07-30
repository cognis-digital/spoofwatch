"""Terrain-aided navigation by profile correlation (pure stdlib).

When GNSS is jammed or spoofed, a platform can still fix its own position from
the *shape of the ground beneath it*. A radar (or laser) altimeter differenced
against a barometric/inertial altitude yields a **terrain-clearance profile** —
the elevation of the ground directly under the vehicle, sampled at intervals
along the flown track. Correlating that measured profile against a stored
reference elevation strip (a DEM corridor along the intended route) locates where
along the reference the platform actually is. This is the classical idea behind
TERCOM / terrain-referenced navigation, reduced to an offline, deterministic
sliding-window correlation.

The reference is a 1-D elevation strip sampled at a fixed ground-sample distance
along a known bearing from a known anchor point. The measured profile (shorter)
is slid across it; at every along-track offset a similarity score is computed and
the best-matching offset is converted back into an along-track distance and,
given the anchor geometry, into a latitude/longitude **fix**.

Three similarity metrics are offered:

* ``mad``  — mean absolute difference (lower is better)
* ``msd`` / ``rmsd`` — mean (root) squared difference (lower is better)
* ``ncc``  — zero-mean normalized cross-correlation, ``-1..1`` (higher is better)

``ncc`` is the default because it is invariant to a constant altitude bias (an
uncalibrated baro/altimeter offset shifts every sample equally and cancels),
which ``mad``/``msd`` are not. Flat, featureless terrain carries no along-track
information; the module detects that (low profile relief) and reports low
confidence rather than a false fix. It also reports an **ambiguity margin** — how
much better the winning offset is than the next-best, non-adjacent candidate — so
periodic terrain (repeating ridges) that could alias into a wrong lock is flagged
rather than trusted.

Detection / navigation only. No NumPy; along-track projection uses the repo's
flat-earth convention. Distances reuse :func:`spoofwatch.records.haversine_km`
style constants indirectly through the shared 111 km/deg approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# valid similarity metrics; only ``ncc`` is "higher is better"
METRICS = ("mad", "msd", "rmsd", "ncc")
HIGHER_IS_BETTER = {"ncc"}

# minimum samples in a measured window for a meaningful correlation
MIN_WINDOW = 4
# terrain relief (std of the measured profile, metres) below which the ground is
# treated as featureless and any fix is reported as low-information
MIN_RELIEF_M = 5.0
# ambiguity: a winning offset whose margin over the next non-adjacent candidate
# is below this (relative for mad/msd, absolute for ncc) is flagged ambiguous
AMBIGUITY_MARGIN = 0.05
# candidates within this many samples of the winner are treated as the same peak
# and excluded when searching for the runner-up
PEAK_GUARD = 2
# km-per-degree flat-earth approximation, matching the repo's other nav modules
KM_PER_DEG = 111.0


@dataclass
class ReferenceProfile:
    """A reference elevation strip sampled along a known track.

    ``elevations`` are ground elevations (metres) at ``spacing_km`` intervals,
    ordered along ``bearing_deg`` (degrees clockwise from North) starting at the
    geographic anchor ``(start_lat, start_lon)``. The anchor/bearing are optional;
    without them a match still yields an along-track offset but no lat/lon fix.
    """

    elevations: list
    spacing_km: float = 0.5
    start_lat: float | None = None
    start_lon: float | None = None
    bearing_deg: float = 0.0

    def position_at_km(self, along_km):
        """Latitude/longitude at ``along_km`` down-track from the anchor.

        Returns ``None`` when the strip has no geographic anchor.
        """
        if self.start_lat is None or self.start_lon is None:
            return None
        return _project_along(self.start_lat, self.start_lon,
                              self.bearing_deg, along_km)


def _project_along(lat, lon, bearing_deg, dist_km):
    """Flat-earth projection of a point ``dist_km`` along ``bearing_deg``."""
    b = math.radians(bearing_deg)
    dlat = (dist_km / KM_PER_DEG) * math.cos(b)
    dlon = (dist_km / (KM_PER_DEG * max(0.1, math.cos(math.radians(lat))))) \
        * math.sin(b)
    return lat + dlat, lon + dlon


def _mean(xs):
    return sum(xs) / len(xs)


def _zero_mean(xs):
    m = _mean(xs)
    return [x - m for x in xs]


def mad(a, b):
    """Mean absolute difference between two equal-length profiles (metres)."""
    if len(a) != len(b):
        raise ValueError("profiles must be equal length")
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def msd(a, b):
    """Mean squared difference between two equal-length profiles (metres²)."""
    if len(a) != len(b):
        raise ValueError("profiles must be equal length")
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def rmsd(a, b):
    """Root-mean-squared difference between two equal-length profiles (metres)."""
    return math.sqrt(msd(a, b))


def ncc(a, b):
    """Zero-mean normalized cross-correlation of two equal-length profiles.

    Returns a value in ``[-1, 1]``; ``1`` is a perfect (bias-invariant) match.
    Returns ``0.0`` when either profile is flat (zero variance), i.e. carries no
    correlatable information.
    """
    if len(a) != len(b):
        raise ValueError("profiles must be equal length")
    za, zb = _zero_mean(a), _zero_mean(b)
    na = math.sqrt(sum(x * x for x in za))
    nb = math.sqrt(sum(y * y for y in zb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(za, zb))
    return max(-1.0, min(1.0, dot / (na * nb)))


def _score(a, b, metric):
    if metric == "mad":
        return mad(a, b)
    if metric == "msd":
        return msd(a, b)
    if metric == "rmsd":
        return rmsd(a, b)
    if metric == "ncc":
        return ncc(a, b)
    raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")


def resample(elevations, src_spacing_km, dst_spacing_km):
    """Linearly resample an elevation profile to a new ground-sample distance.

    Used to bring a measured profile onto the reference strip's spacing before
    correlation. The output spans the same physical length as the input.
    """
    if src_spacing_km <= 0 or dst_spacing_km <= 0:
        raise ValueError("spacings must be positive")
    n = len(elevations)
    if n == 0:
        return []
    if n == 1:
        return [float(elevations[0])]
    total_km = (n - 1) * src_spacing_km
    out_n = max(2, int(round(total_km / dst_spacing_km)) + 1)
    out = []
    for i in range(out_n):
        # span the same physical length exactly so both endpoints are preserved
        x_km = total_km * i / (out_n - 1)
        pos = x_km / src_spacing_km
        lo = int(math.floor(pos))
        if lo >= n - 1:
            out.append(float(elevations[-1]))
            continue
        frac = pos - lo
        out.append(float(elevations[lo]) * (1 - frac)
                   + float(elevations[lo + 1]) * frac)
    return out


def slide_scores(measured, reference, metric="ncc"):
    """Score ``measured`` at every valid along-track offset within ``reference``.

    Returns a list of ``(offset_samples, score)`` pairs, one per position where
    the measured window fits entirely inside the reference strip. Raises when the
    measured window is longer than the reference or below :data:`MIN_WINDOW`.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")
    m, r = list(measured), list(reference)
    if len(m) < MIN_WINDOW:
        raise ValueError(f"measured window needs >= {MIN_WINDOW} samples")
    if len(m) > len(r):
        raise ValueError("measured window longer than reference strip")
    n_off = len(r) - len(m) + 1
    return [(off, _score(m, r[off:off + len(m)], metric))
            for off in range(n_off)]


def _best_and_runner_up(scores, higher_is_better, guard=PEAK_GUARD):
    """Winning ``(offset, score)`` and the best non-adjacent runner-up."""
    best = max(scores, key=lambda p: p[1]) if higher_is_better \
        else min(scores, key=lambda p: p[1])
    best_off = best[0]
    others = [p for p in scores if abs(p[0] - best_off) > guard]
    if not others:
        return best, None
    runner = max(others, key=lambda p: p[1]) if higher_is_better \
        else min(others, key=lambda p: p[1])
    return best, runner


def _confidence(best_score, runner_score, metric, relief_m):
    """0..1 fix confidence from ambiguity margin and terrain information.

    Blends how much the winner beats the runner-up (uniqueness) with an absolute
    goodness-of-fit term, then penalises featureless terrain.
    """
    higher = metric in HIGHER_IS_BETTER
    if higher:  # ncc: scores in [-1, 1]
        fit = max(0.0, best_score)
        margin = (best_score - runner_score) if runner_score is not None else 1.0
        margin = max(0.0, margin)
        uniqueness = min(1.0, margin / AMBIGUITY_MARGIN) if AMBIGUITY_MARGIN else 1.0
    else:  # mad/msd/rmsd: lower better, >= 0
        denom = (best_score + runner_score) if runner_score is not None else best_score
        if runner_score is None or denom <= 0:
            margin = 1.0
        else:
            margin = (runner_score - best_score) / denom
        margin = max(0.0, margin)
        uniqueness = min(1.0, margin / AMBIGUITY_MARGIN) if AMBIGUITY_MARGIN else 1.0
        # a perfect distance fit is 0; map small residuals toward fit=1
        fit = 1.0 / (1.0 + best_score)
    relief_factor = min(1.0, relief_m / MIN_RELIEF_M) if MIN_RELIEF_M else 1.0
    return max(0.0, min(1.0, 0.5 * (uniqueness + fit) * relief_factor))


def _relief_m(measured):
    """Standard deviation (metres) of the measured profile — its along-track relief."""
    if len(measured) < 2:
        return 0.0
    mu = _mean(measured)
    var = sum((x - mu) ** 2 for x in measured) / len(measured)
    return math.sqrt(max(0.0, var))


def best_match(measured, reference, metric="ncc"):
    """Best along-track offset of ``measured`` within a plain ``reference`` list.

    Returns a dict with the winning ``offset_samples`` and ``score``, the
    runner-up score, the ambiguity ``margin``, an ``ambiguous`` flag, the terrain
    ``relief_m``, a ``flat_terrain`` flag, and a 0..1 ``confidence``. This is the
    metric-level primitive; :func:`terrain_fix` adds geographic projection.
    """
    scores = slide_scores(measured, reference, metric)
    higher = metric in HIGHER_IS_BETTER
    best, runner = _best_and_runner_up(scores, higher)
    runner_score = runner[1] if runner is not None else None
    relief = _relief_m(measured)
    conf = _confidence(best[1], runner_score, metric, relief)

    if runner_score is None:
        margin = None
    elif higher:
        margin = best[1] - runner_score
    else:
        denom = best[1] + runner_score
        margin = ((runner_score - best[1]) / denom) if denom > 0 else 1.0

    ambiguous = False
    if margin is not None:
        ambiguous = margin < AMBIGUITY_MARGIN
    flat = relief < MIN_RELIEF_M
    return {
        "metric": metric,
        "n_offsets": len(scores),
        "offset_samples": best[0],
        "score": round(best[1], 6),
        "runner_up_score": None if runner_score is None else round(runner_score, 6),
        "margin": None if margin is None else round(margin, 6),
        "ambiguous": bool(ambiguous or flat),
        "relief_m": round(relief, 3),
        "flat_terrain": bool(flat),
        "confidence": round(conf, 3),
    }


def terrain_fix(measured, reference: ReferenceProfile, metric="ncc",
                measured_spacing_km=None):
    """Fix position by correlating a measured profile against a reference strip.

    ``measured`` is the terrain-clearance profile sampled along the track, oldest
    sample first and the **most recent sample last** (i.e. the platform's current
    position is the trailing edge of the window). If ``measured_spacing_km`` is
    given and differs from the reference spacing, the measured profile is
    resampled onto the reference's ground-sample distance first.

    Returns the :func:`best_match` dict augmented with the along-track distances
    of the matched window (``window_start_km`` / ``fix_along_km`` at the trailing
    edge) and, when the reference is geo-anchored, the ``fix_lat`` / ``fix_lon``
    position estimate.
    """
    m = list(measured)
    if measured_spacing_km is not None and \
            not math.isclose(measured_spacing_km, reference.spacing_km):
        m = resample(m, measured_spacing_km, reference.spacing_km)

    res = best_match(m, reference.elevations, metric)
    off = res["offset_samples"]
    spacing = reference.spacing_km
    window_start_km = off * spacing
    fix_along_km = window_start_km + (len(m) - 1) * spacing

    res["window_start_km"] = round(window_start_km, 4)
    res["fix_along_km"] = round(fix_along_km, 4)
    res["window_len_samples"] = len(m)

    pos = reference.position_at_km(fix_along_km)
    if pos is not None:
        res["fix_lat"] = round(pos[0], 6)
        res["fix_lon"] = round(pos[1], 6)
    else:
        res["fix_lat"] = None
        res["fix_lon"] = None
    return res
