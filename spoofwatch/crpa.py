"""Multi-antenna (CRPA-style) spatial spoofing detection (pure stdlib).

A single antenna sees only *that* a signal arrived; an **antenna array** — the
kind a controlled-reception-pattern antenna (CRPA) carries — sees *where from*.
Each element of the array samples the same wavefront at a slightly different
point, so the phase measured across the elements encodes the wave's
direction-of-arrival (DoA). Estimate one DoA per satellite and a decisive
geometric tell appears:

* A **genuine constellation** is scattered across the whole sky hemisphere, so
  its per-satellite DoA unit vectors point in many different directions — their
  spatial scatter fills three dimensions.
* A **ground (or single-aperture) spoofer** transmits every counterfeit
  "satellite" from *one* physical place, so every DoA vector points the same way.
  The array's spatial covariance collapses toward **rank one**: all the energy
  lands on a single eigen-direction.

This module reconstructs the DoA of each satellite from the array's per-element
phases (a small least-squares inversion of the baseline geometry, via
:mod:`spoofwatch.linalg`), then measures how single-directional the resulting set
is with two complementary statistics:

1. **Mean pairwise separation** — the average angle between every pair of DoA
   vectors. Wide for a real sky, near zero for one source.
2. **Eigen-concentration** — the largest eigenvalue of the 3×3 DoA scatter matrix
   as a fraction of its trace, in ``[1/3, 1]``. ``1`` means perfectly rank-one
   (one direction); ``1/3`` means isotropic spread. Computed with a closed-form
   symmetric-3×3 eigensolver — no NumPy, no iteration.

This is a *hardware-optional, advanced* signal: most position feeds carry no
per-element phase, but a receiver with an array antenna can expose one of the
most decisive spoofing tells available. It is a complement to
:mod:`spoofwatch.aoa` (which scores a single already-known set of DoA angles) and
to :mod:`spoofwatch.skygeom` (which works from each satellite's *claimed*
az/el) — here the DoA is *measured* by the array itself and cannot be forged by
the message content.

Detection and awareness only. Nothing here nominates, locates, targets, or
exploits a transmitter; it reports the spatial spoofing *signature* so an
integrity engine can discount a compromised solution.

Convention: azimuth is degrees clockwise from North, elevation degrees above the
horizon; the local East/North/Up line-of-sight unit vector is
``(cos el·sin az, cos el·cos az, sin el)`` — the same convention as
:mod:`spoofwatch.aoa` and :mod:`spoofwatch.skygeom`. Element positions and
wavelength are in the same length unit (metres by default); only their ratio
matters. Phases are treated as already unwrapped (absolute), i.e. the array
baselines are short enough — or the receiver's phase tracking good enough — that
there is no full-cycle ambiguity; this is stated in :mod:`docs.LIMITATIONS`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import linalg

# --- detector thresholds -------------------------------------------------

# a genuine sky's DoA vectors separate by more than this (deg) on average;
# below it the arrivals are collapsing toward a single direction
MIN_MEAN_SEPARATION_DEG = 15.0
# eigen-concentration above this (largest eigenvalue / trace of the DoA scatter
# matrix) means the spatial covariance is essentially rank-one -> one source
CONCENTRATION_SUSPECT = 0.90
# a perfectly single-direction set concentrates to exactly 1.0
CONCENTRATION_MAX = 1.0
# minimum satellites needed to judge the spatial spread
MIN_SATS = 4
# minimum array elements to resolve a 3-D direction (>= 3 independent baselines)
MIN_ELEMENTS = 4
# per-satellite least-squares phase residual (rad, RMS) above this means the
# measured phases do not fit a single plane wave — reported for awareness only
PLANE_WAVE_RESIDUAL_RAD = 0.30


@dataclass
class AntennaArray:
    """A phased array of antenna elements at known local positions.

    ``positions`` is a list of ``(east, north, up)`` element coordinates in the
    same length unit as ``wavelength`` (metres by default). The first element is
    treated as the phase reference. Resolving a full 3-D direction needs at least
    four **non-coplanar** elements (three independent baselines); a coplanar array
    can only fix the direction up to an elevation-sign ambiguity and is reported
    as ill-posed rather than silently mis-solved.
    """

    positions: list
    wavelength: float = 0.1903          # GPS L1 carrier wavelength, metres

    def baselines(self):
        """Element-to-reference baseline vectors ``p_i - p_0`` (skips the reference)."""
        p0 = self.positions[0]
        return [[p[k] - p0[k] for k in range(3)] for p in self.positions[1:]]


@dataclass
class SatMeasurement:
    """One satellite's per-element carrier phase measured across the array.

    ``phases`` is a list of phase samples (radians), one per array element and in
    the same order as :attr:`AntennaArray.positions`. Phases are treated as
    unwrapped/absolute (see the module note on cycle ambiguity).
    """

    sat_id: str
    phases: list


def enu_los(az_deg, el_deg):
    """Local East/North/Up unit line-of-sight vector for an az/el pair."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    ce = math.cos(el)
    return (ce * math.sin(az), ce * math.cos(az), math.sin(el))


def element_phases(array, az_deg, el_deg, bias_rad=0.0):
    """Forward model: phase (rad) at each element for a plane wave from az/el.

    ``phi_i = (2π/λ)·(k · p_i) + bias`` where ``k`` is the unit direction toward
    the source and ``p_i`` the element position. The common ``bias`` (an arbitrary
    reference phase) cancels in every baseline difference, so it never affects the
    estimated direction; it exists only so callers can synthesise realistic
    absolute phases. Deterministic — the basis for the synth helper and tests.
    """
    k = enu_los(az_deg, el_deg)
    scale = 2.0 * math.pi / array.wavelength
    out = []
    for p in array.positions:
        out.append(scale * (k[0] * p[0] + k[1] * p[1] + k[2] * p[2]) + bias_rad)
    return out


def synth_measurements(array, sources, bias_rad=0.0):
    """Build deterministic :class:`SatMeasurement` objects for ``sources``.

    ``sources`` is an iterable of ``(sat_id, az_deg, el_deg)``. Handy for demos
    and tests: feed a spread-out sky and the detector stays quiet; feed several
    sources at (nearly) one az/el and it lights up.
    """
    return [SatMeasurement(sid, element_phases(array, az, el, bias_rad))
            for sid, az, el in sources]


def estimate_direction(array, phases):
    """Estimate the arrival unit vector of one satellite from array phases.

    Differences the phases against the reference element, converts each to a
    baseline projection ``k · b_i = (λ/2π)·Δφ_i``, and solves the overdetermined
    ``B k = y`` in least squares (:func:`spoofwatch.linalg.lstsq`). Returns
    ``(unit_vector, residual_rms_rad)`` where the residual is the RMS phase
    mismatch (rad) to a single plane wave — small for a clean arrival, larger for
    multipath or noise. Raises :class:`ValueError` on a rank-deficient
    (e.g. coplanar) array.
    """
    if len(array.positions) < MIN_ELEMENTS:
        raise ValueError(f"need >= {MIN_ELEMENTS} array elements")
    if len(phases) != len(array.positions):
        raise ValueError("phase count must match the number of array elements")
    B = array.baselines()
    scale = array.wavelength / (2.0 * math.pi)
    y = [(phases[i + 1] - phases[0]) * scale for i in range(len(B))]
    k, resid = linalg.lstsq(B, y)     # raises ValueError if singular
    norm = math.sqrt(k[0] * k[0] + k[1] * k[1] + k[2] * k[2])
    if norm < 1e-12:
        unit = (0.0, 0.0, 0.0)
    else:
        unit = (k[0] / norm, k[1] / norm, k[2] / norm)
    # residual is a projection error in metres; back to an equivalent phase (rad)
    rms_m = math.sqrt(sum(r * r for r in resid) / len(resid)) if resid else 0.0
    resid_rad = rms_m / scale if scale > 0 else 0.0
    return unit, resid_rad


def estimate_directions(array, measurements):
    """Estimate a DoA unit vector for every satellite measurement.

    Returns ``(vectors, residuals)`` — a list of ``(sat_id, unit_vector)`` and a
    parallel list of ``(sat_id, residual_rms_rad)``, in the input order.
    """
    vectors, residuals = [], []
    for m in measurements:
        unit, resid = estimate_direction(array, m.phases)
        vectors.append((m.sat_id, unit))
        residuals.append((m.sat_id, resid))
    return vectors, residuals


def _angle_deg(a, b):
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def mean_pairwise_separation_deg(vectors):
    """Average angle (deg) between every unordered pair of unit vectors.

    Zero-length vectors (a failed estimate) are dropped first. Fewer than two
    usable vectors yields ``0.0``. Large for a sky-spread constellation, near zero
    when all arrivals share one direction.
    """
    vs = [v for v in vectors if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) > 1e-18]
    n = len(vs)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _angle_deg(vs[i], vs[j])
            count += 1
    return total / count if count else 0.0


def scatter_matrix(vectors):
    """Symmetric 3×3 scatter matrix ``Σ vᵢ vᵢᵀ`` of the DoA unit vectors."""
    S = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for v in vectors:
        for a in range(3):
            for b in range(3):
                S[a][b] += v[a] * v[b]
    return S


def _eigvals_sym3(A):
    """Eigenvalues of a symmetric 3×3 matrix, descending (closed form).

    Smith's trigonometric solution — exact, deterministic, no iteration and no
    NumPy. Returns ``(e1, e2, e3)`` with ``e1 >= e2 >= e3``.
    """
    p1 = A[0][1] ** 2 + A[0][2] ** 2 + A[1][2] ** 2
    tr = A[0][0] + A[1][1] + A[2][2]
    if p1 <= 1e-18:
        # already diagonal
        d = sorted([A[0][0], A[1][1], A[2][2]], reverse=True)
        return (d[0], d[1], d[2])
    q = tr / 3.0
    p2 = ((A[0][0] - q) ** 2 + (A[1][1] - q) ** 2 + (A[2][2] - q) ** 2 + 2.0 * p1)
    p = math.sqrt(p2 / 6.0)
    # B = (A - qI) / p
    B = [[(A[i][j] - (q if i == j else 0.0)) / p for j in range(3)] for i in range(3)]
    detB = (B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1])
            - B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0])
            + B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]))
    r = max(-1.0, min(1.0, detB / 2.0))
    phi = math.acos(r) / 3.0
    e1 = q + 2.0 * p * math.cos(phi)
    e3 = q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)
    e2 = tr - e1 - e3
    vals = sorted([e1, e2, e3], reverse=True)
    return (vals[0], vals[1], vals[2])


def eigen_concentration(vectors):
    """Largest eigenvalue of the DoA scatter matrix as a fraction of its trace.

    Lies in ``[1/3, 1]`` for unit vectors: ``1`` when every arrival shares one
    direction (rank-one covariance, the spoofer signature) and ``1/3`` for an
    isotropic spread. Returns ``0.0`` when there are no usable vectors.
    """
    vs = [v for v in vectors if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) > 1e-18]
    if not vs:
        return 0.0
    S = scatter_matrix(vs)
    e1, e2, e3 = _eigvals_sym3(S)
    trace = e1 + e2 + e3
    if trace <= 1e-18:
        return 0.0
    return max(0.0, min(1.0, e1 / trace))


def check_directions(vectors, min_sep_deg=MIN_MEAN_SEPARATION_DEG,
                     concentration_suspect=CONCENTRATION_SUSPECT,
                     min_sats=MIN_SATS):
    """Score a set of DoA unit vectors for the single-source spoofing signature.

    ``vectors`` is an iterable of ``(sat_id, unit_vector)`` (as returned by
    :func:`estimate_directions`) or of bare 3-tuples. Combines the mean-pairwise
    separation and eigen-concentration tests: a genuine sky is wide and
    low-concentration, a single source is narrow and rank-one. Returns a dict with
    both metrics, a boolean ``single_source`` flag, and a 0..1 confidence. When
    fewer than ``min_sats`` usable directions are present the check is unavailable.
    """
    units = []
    for v in vectors:
        vec = v[1] if (isinstance(v, tuple) and len(v) == 2
                       and isinstance(v[1], (tuple, list))) else v
        if (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]) > 1e-18:
            units.append((vec[0], vec[1], vec[2]))

    if len(units) < min_sats:
        return {"available": False, "n_sats": len(units),
                "reason": f"need >= {min_sats} satellites with a usable direction",
                "single_source": False}

    sep = mean_pairwise_separation_deg(units)
    conc = eigen_concentration(units)

    narrow = sep < min_sep_deg
    rank_one = conc >= concentration_suspect
    single = narrow or rank_one

    # normalised strength of each tell (0..1), only when it fires
    sep_term = max(0.0, (min_sep_deg - sep) / min_sep_deg) if narrow else 0.0
    denom = (CONCENTRATION_MAX - concentration_suspect) or 1.0
    conc_term = max(0.0, (conc - concentration_suspect) / denom) if rank_one else 0.0
    conf = min(1.0, max(sep_term, conc_term)) if single else 0.0

    return {
        "available": True,
        "n_sats": len(units),
        "mean_separation_deg": round(sep, 2),
        "separation_threshold_deg": min_sep_deg,
        "eigen_concentration": round(conc, 4),
        "concentration_threshold": concentration_suspect,
        "narrow_spread": bool(narrow),
        "rank_one_covariance": bool(rank_one),
        "single_source": bool(single),
        "confidence": round(conf, 3),
    }


def check(array, measurements, min_sep_deg=MIN_MEAN_SEPARATION_DEG,
          concentration_suspect=CONCENTRATION_SUSPECT, min_sats=MIN_SATS,
          residual_rad=PLANE_WAVE_RESIDUAL_RAD):
    """End-to-end array-spoofing check from per-element phase measurements.

    Estimates each satellite's DoA from the array phases, then runs
    :func:`check_directions` on the set. The result is augmented with the
    per-satellite plane-wave residuals and the count of channels whose phases fit
    a single plane wave poorly (``noisy_channels`` — an awareness field, not part
    of the spoof decision). Ill-posed arrays (too few or coplanar elements) are
    reported as unavailable rather than raising.
    """
    meas = list(measurements)
    try:
        vectors, residuals = estimate_directions(array, meas)
    except ValueError as exc:
        return {"available": False, "n_sats": len(meas),
                "reason": str(exc), "single_source": False}

    res = check_directions(vectors, min_sep_deg, concentration_suspect, min_sats)
    noisy = [sid for sid, r in residuals if r > residual_rad]
    res["max_residual_rad"] = round(max((r for _, r in residuals), default=0.0), 4)
    res["noisy_channels"] = sorted(noisy)
    res["n_noisy_channels"] = len(noisy)
    return res
