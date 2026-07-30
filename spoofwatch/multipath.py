"""GNSS multipath / signal-reflection detection (pure stdlib).

A clean satellite signal reaches the antenna along the single direct line of
sight. Near the ground, water, buildings, or the platform's own airframe, a
copy of that signal also arrives *reflected* off a nearby surface. The reflected
ray travels a little farther, so it lands out of phase with the direct ray; as
the geometry slowly changes the two rays drift in and out of phase and interfere
**constructively then destructively**. That interference leaves two deterministic
fingerprints that this module reads:

1. **Carrier-to-noise fading.** The summed direct+reflected power rises and falls
   in a quasi-periodic ripple, so the channel's C/N0 *fluctuates* far more than
   the gentle, monotone rise an honest signal shows as it climbs the sky. After
   removing the slow elevation trend, a multipath channel's residual C/N0 wobble
   is large and oscillatory (many zero-crossings); a clean channel's is small.

2. **Elevation-dependent range residuals.** Reflections are strongest at **low
   elevation**, where the incidence angle onto ground/water is shallow and the
   antenna's gain toward the reflection is high. A satellite low on the horizon
   therefore carries a much larger code-minus-carrier (multipath) residual than
   one overhead. Binning the residual by elevation, a multipath channel shows the
   low-elevation band's residual scatter dwarfing the high-elevation band's, and
   ``|residual|`` sloping downward with rising elevation.

Neither fingerprint is a spoof — multipath is a benign propagation artefact — so
this is a **quality / awareness** monitor: it tells an integrity engine which
channels are reflection-contaminated so their range errors are not mistaken for a
teleport, a jam, or a spoof (and so they can be de-weighted). Detection and
characterisation only; nothing here nominates, targets, or exploits anything.

The multipath observable is the standard **code-minus-carrier** combination
(``code_m - phase_m``, metres) referenced to each track's first epoch so the
unknown integer carrier ambiguity cancels — the same convention as
:mod:`spoofwatch.codecarrier`. Callers that only have C/N0 may leave it zero and
still use fingerprint (1). Purely relative and statistical: no per-receiver
calibration, no external tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# detrended C/N0 wobble (dB-Hz, std) above this is multipath-grade fading
FLUCT_STD_DBHZ = 1.5
# elevation (deg) splitting the "low" (reflection-prone) and "high" bands
LOW_EL_DEG = 30.0
# low-band residual scatter this many times the high-band's flags the
# classic elevation-dependent multipath structure
RESIDUAL_RATIO = 2.0
# |residual| sloping downward with elevation faster than this (m/deg, magnitude)
# corroborates elevation-dependent multipath
RESIDUAL_SLOPE_M_PER_DEG = 0.02
# fewest epochs on a track before its fluctuation/pattern can be judged
MIN_EPOCHS = 6
# fewest samples that must fall in each elevation band to compare them
MIN_PER_BAND = 3


@dataclass
class Epoch:
    """One epoch of one satellite's observation.

    ``cn0`` is carrier-to-noise density (dB-Hz); ``el_deg`` the elevation above
    the horizon (deg). ``code_m`` / ``phase_m`` are the code pseudorange and
    carrier phase already scaled to metres; their difference is the multipath
    observable. Leave the ranges at their defaults to run the C/N0-only path.
    """

    ts: float
    sat_id: str
    cn0: float
    el_deg: float
    code_m: float = 0.0
    phase_m: float = 0.0
    constellation: str = "GPS"


@dataclass
class SatMultipath:
    sat_id: str
    n_epochs: int
    fluct_std_dbhz: float           # detrended C/N0 wobble amplitude
    n_crossings: int                # zero-crossings of the detrended wobble
    low_residual_std_m: float       # code-minus-carrier scatter, low elevation
    high_residual_std_m: float      # code-minus-carrier scatter, high elevation
    residual_ratio: float | None    # low/high scatter ratio (None if unknowable)
    residual_slope_m_per_deg: float # |residual| vs elevation slope (m/deg)
    fading: bool                    # C/N0 fingerprint tripped
    elevation_dependent: bool       # range-residual fingerprint tripped
    multipath: bool
    confidence: float               # 0..1
    reason: str = ""


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _stdev(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def linfit(xs, ys):
    """Ordinary least-squares ``y = slope*x + intercept``.

    Returns ``(slope, intercept)``. Zero-variance ``x`` (all-equal ``xs``) yields
    a zero slope and the mean of ``ys`` as the intercept.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = _mean(xs)
    my = _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    slope = sxy / sxx
    return slope, my - slope * mx


def by_sat(epochs):
    """Group epochs by satellite id, each track sorted chronologically."""
    tracks = {}
    for e in epochs:
        tracks.setdefault(e.sat_id, []).append(e)
    for t in tracks.values():
        t.sort(key=lambda e: e.ts)
    return tracks


def residual_series(track):
    """Code-minus-carrier multipath residual per epoch, ambiguity-removed.

    Returns a list of ``(el_deg, residual_m)`` where ``residual_m`` is
    ``(code_m - phase_m)`` referenced to the track's first epoch, so the unknown
    integer carrier ambiguity cancels and the series starts at zero. Elevation is
    carried alongside for the elevation-dependence test.
    """
    track = sorted(track, key=lambda e: e.ts)
    if not track:
        return []
    ref = track[0].code_m - track[0].phase_m
    return [(e.el_deg, (e.code_m - e.phase_m) - ref) for e in track]


def detrend_cn0(track):
    """C/N0 with its slow elevation trend removed, leaving the fading wobble.

    Fits ``cn0`` against ``el_deg`` and returns the residuals in chronological
    order. On an honest channel these are small and noise-like; a multipath
    channel's are large and oscillatory. A near-constant elevation window reduces
    this to simple mean-removal (the fit slope collapses to zero).
    """
    track = sorted(track, key=lambda e: e.ts)
    if not track:
        return []
    els = [e.el_deg for e in track]
    cn0s = [e.cn0 for e in track]
    slope, intercept = linfit(els, cn0s)
    return [c - (slope * el + intercept) for el, c in zip(els, cn0s)]


def zero_crossings(values):
    """Number of sign changes in a zero-mean-ish series (oscillation count).

    Consecutive samples straddling zero count as one crossing; exact zeros are
    skipped so a flat run does not inflate the tally. Quasi-periodic multipath
    fading produces many crossings; monotone or lightly-noisy signals few.
    """
    crossings = 0
    prev = None
    for v in values:
        if v == 0.0:
            continue
        s = 1 if v > 0 else -1
        if prev is not None and s != prev:
            crossings += 1
        prev = s
    return crossings


def elevation_bands(series, low_el=LOW_EL_DEG):
    """Split a ``(el_deg, residual_m)`` series into low- and high-elevation bands.

    Returns ``(low_residuals, high_residuals)`` — the residual values with
    elevation strictly below ``low_el`` and at-or-above it, respectively.
    """
    low = [r for el, r in series if el < low_el]
    high = [r for el, r in series if el >= low_el]
    return low, high


def residual_ratio(series, low_el=LOW_EL_DEG, min_per_band=MIN_PER_BAND):
    """Low-band / high-band residual-scatter ratio (elevation dependence).

    A large ratio is the signature of reflection contamination concentrated at
    low elevation. Returns ``None`` when either band is too sparse to compare or
    the high band has (near) zero scatter, in which case the ratio is undefined.
    """
    low, high = elevation_bands(series, low_el)
    if len(low) < min_per_band or len(high) < min_per_band:
        return None
    hi_std = _stdev(high)
    if hi_std <= 1e-9:
        return None
    return _stdev(low) / hi_std


def residual_slope(series):
    """Slope of ``|residual|`` against elevation (m per degree).

    Multipath residuals shrink as a satellite climbs, so a *negative* slope
    corroborates elevation-dependent reflection. Returns the signed slope.
    """
    if not series:
        return 0.0
    els = [el for el, _ in series]
    mags = [abs(r) for _, r in series]
    slope, _ = linfit(els, mags)
    return slope


def analyze_sat(sat_id, track, fluct_std=FLUCT_STD_DBHZ, low_el=LOW_EL_DEG,
                ratio_thresh=RESIDUAL_RATIO,
                slope_thresh=RESIDUAL_SLOPE_M_PER_DEG,
                min_epochs=MIN_EPOCHS, min_per_band=MIN_PER_BAND):
    """Characterise one satellite track for multipath. Returns a :class:`SatMultipath`.

    Two fingerprints are evaluated independently:

    * **fading** — the detrended C/N0 wobble std exceeds ``fluct_std`` *and* the
      wobble actually oscillates (several zero-crossings), ruling out a single
      monotone drift.
    * **elevation_dependent** — the low/high residual-scatter ratio exceeds
      ``ratio_thresh`` *and* ``|residual|`` slopes downward with elevation faster
      than ``slope_thresh``.

    A track is ``multipath`` when either fingerprint trips. Confidence blends the
    normalised exceedance of both, so a channel showing both fingerprints scores
    highest.
    """
    track = sorted(track, key=lambda e: e.ts)
    n = len(track)
    resid = residual_series(track)

    if n < min_epochs:
        return SatMultipath(sat_id, n, 0.0, 0, 0.0, 0.0, None, 0.0,
                            False, False, False, 0.0,
                            reason=f"need >= {min_epochs} epochs")

    wobble = detrend_cn0(track)
    fluct = _stdev(wobble)
    crossings = zero_crossings(wobble)

    low, high = elevation_bands(resid, low_el)
    low_std = _stdev(low)
    high_std = _stdev(high)
    ratio = residual_ratio(resid, low_el, min_per_band)
    slope = residual_slope(resid)

    # fingerprint (1): oscillatory C/N0 fading, not a lone monotone drift
    fading = fluct > fluct_std and crossings >= 2
    # fingerprint (2): scatter concentrated low + shrinking with elevation
    elevation_dependent = (ratio is not None and ratio > ratio_thresh
                           and slope < -slope_thresh)

    multipath = fading or elevation_dependent

    fluct_term = max(0.0, (fluct - fluct_std) / fluct_std) if fading else 0.0
    ratio_term = 0.0
    if elevation_dependent and ratio is not None:
        ratio_term = max(0.0, (ratio - ratio_thresh) / ratio_thresh)
    conf = 0.0
    reason = ""
    if multipath:
        conf = min(1.0, 0.55 * min(1.0, fluct_term) + 0.55 * min(1.0, ratio_term))
        parts = []
        if fading:
            parts.append("oscillatory C/N0 fading")
        if elevation_dependent:
            parts.append("elevation-dependent range residuals")
        reason = "multipath: " + " + ".join(parts)

    return SatMultipath(
        sat_id=sat_id,
        n_epochs=n,
        fluct_std_dbhz=round(fluct, 3),
        n_crossings=crossings,
        low_residual_std_m=round(low_std, 4),
        high_residual_std_m=round(high_std, 4),
        residual_ratio=None if ratio is None else round(ratio, 3),
        residual_slope_m_per_deg=round(slope, 5),
        fading=bool(fading),
        elevation_dependent=bool(elevation_dependent),
        multipath=bool(multipath),
        confidence=round(conf, 3),
        reason=reason,
    )


def check(epochs, fluct_std=FLUCT_STD_DBHZ, low_el=LOW_EL_DEG,
          ratio_thresh=RESIDUAL_RATIO, slope_thresh=RESIDUAL_SLOPE_M_PER_DEG,
          min_epochs=MIN_EPOCHS, min_per_band=MIN_PER_BAND):
    """Screen a multi-satellite epoch set for multipath-contaminated channels.

    Groups ``epochs`` by satellite, runs :func:`analyze_sat` on each track, and
    returns a dict with every satellite's :class:`SatMultipath` (as a dict), the
    sorted list of ``flagged`` satellites, and an overall 0..1 confidence equal to
    the strongest single-channel confidence (multipath is per-channel, so it does
    not compound across satellites).
    """
    tracks = by_sat(epochs)
    per = []
    flagged = []
    best_conf = 0.0
    for sat_id, track in sorted(tracks.items()):
        r = analyze_sat(sat_id, track, fluct_std, low_el, ratio_thresh,
                        slope_thresh, min_epochs, min_per_band)
        per.append(_as_dict(r))
        if r.multipath:
            flagged.append(sat_id)
            best_conf = max(best_conf, r.confidence)
    return {"per_satellite": per, "flagged": sorted(flagged),
            "n_flagged": len(flagged), "n_satellites": len(tracks),
            "multipath": bool(flagged), "confidence": round(best_conf, 3)}


def _as_dict(r: SatMultipath):
    return {
        "sat_id": r.sat_id,
        "n_epochs": r.n_epochs,
        "fluct_std_dbhz": r.fluct_std_dbhz,
        "n_crossings": r.n_crossings,
        "low_residual_std_m": r.low_residual_std_m,
        "high_residual_std_m": r.high_residual_std_m,
        "residual_ratio": r.residual_ratio,
        "residual_slope_m_per_deg": r.residual_slope_m_per_deg,
        "fading": r.fading,
        "elevation_dependent": r.elevation_dependent,
        "multipath": r.multipath,
        "confidence": r.confidence,
        "reason": r.reason,
    }
