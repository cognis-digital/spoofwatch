import math

import pytest

from spoofwatch import crpa


# --- fixtures -------------------------------------------------------------

def _array(wavelength=0.1903):
    # a compact, non-coplanar 4-element array (metres) — three independent
    # baselines, so a full 3-D direction is resolvable
    return crpa.AntennaArray(
        positions=[(0.0, 0.0, 0.0), (0.10, 0.0, 0.0),
                   (0.0, 0.10, 0.0), (0.05, 0.05, 0.08)],
        wavelength=wavelength,
    )


def _bigger_array():
    # a 6-element non-coplanar array for over-determined estimation
    return crpa.AntennaArray(
        positions=[(0.0, 0.0, 0.0), (0.12, 0.0, 0.0), (0.0, 0.12, 0.0),
                   (0.06, 0.06, 0.09), (-0.08, 0.03, 0.05), (0.02, -0.07, 0.04)],
        wavelength=0.1903,
    )


def _sky_sources():
    # a well-spread constellation across the sky
    return [("a", 10, 60), ("b", 120, 40), ("c", 250, 55),
            ("d", 300, 25), ("e", 180, 70), ("f", 60, 15)]


def _single_source_sources(n=6, az=45.0, el=30.0, spread=0.3):
    # every "satellite" from nearly one direction (a ground spoofer)
    return [(f"s{i}", az + spread * i, el + 0.5 * spread * i) for i in range(n)]


def _sky(array=None):
    array = array or _array()
    return crpa.synth_measurements(array, _sky_sources())


def _spoof(array=None, **kw):
    array = array or _array()
    return crpa.synth_measurements(array, _single_source_sources(**kw))


# --- forward model / estimation round-trip --------------------------------

def test_element_phases_length_matches_array():
    arr = _array()
    ph = crpa.element_phases(arr, 120, 40)
    assert len(ph) == len(arr.positions)


def test_reference_element_phase_is_bias_only():
    arr = _array()
    ph = crpa.element_phases(arr, 33, 41, bias_rad=1.234)
    # reference element sits at the origin -> only the common bias remains
    assert ph[0] == pytest.approx(1.234)


@pytest.mark.parametrize("az,el", [
    (0, 45), (10, 60), (90, 10), (120, 40), (200, 5),
    (250, 55), (300, 25), (359, 80), (45, 89), (180, 0.5),
])
def test_estimate_recovers_direction(az, el):
    arr = _array()
    ph = crpa.element_phases(arr, az, el)
    unit, resid = crpa.estimate_direction(arr, ph)
    expect = crpa.enu_los(az, el)
    for a, b in zip(unit, expect):
        assert a == pytest.approx(b, abs=1e-6)
    assert resid == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("bias", [0.0, 0.7, -2.3, 10.0])
def test_common_bias_does_not_change_direction(bias):
    arr = _array()
    base = crpa.estimate_direction(arr, crpa.element_phases(arr, 137, 22))[0]
    shifted = crpa.estimate_direction(arr, crpa.element_phases(arr, 137, 22, bias_rad=bias))[0]
    for a, b in zip(base, shifted):
        assert a == pytest.approx(b, abs=1e-9)


def test_estimate_returns_unit_vector():
    arr = _bigger_array()
    unit, _ = crpa.estimate_direction(arr, crpa.element_phases(arr, 77, 33))
    norm = math.sqrt(sum(c * c for c in unit))
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_overdetermined_array_recovers_direction():
    arr = _bigger_array()
    unit, resid = crpa.estimate_direction(arr, crpa.element_phases(arr, 215, 47))
    expect = crpa.enu_los(215, 47)
    for a, b in zip(unit, expect):
        assert a == pytest.approx(b, abs=1e-6)
    assert resid == pytest.approx(0.0, abs=1e-6)


def test_estimate_too_few_elements_raises():
    arr = crpa.AntennaArray(positions=[(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0)])
    with pytest.raises(ValueError):
        crpa.estimate_direction(arr, [0.0, 1.0, 2.0])


def test_estimate_phase_count_mismatch_raises():
    arr = _array()
    with pytest.raises(ValueError):
        crpa.estimate_direction(arr, [0.0, 1.0, 2.0])   # 3 != 4 elements


def test_coplanar_array_raises_singular():
    # all elements share up=0 -> baselines are coplanar -> rank-deficient
    arr = crpa.AntennaArray(positions=[(0, 0, 0), (0.1, 0, 0),
                                       (0, 0.1, 0), (0.1, 0.1, 0)])
    with pytest.raises(ValueError):
        crpa.estimate_direction(arr, crpa.element_phases(arr, 50, 30))


# --- baselines ------------------------------------------------------------

def test_baselines_skip_reference():
    arr = _array()
    b = arr.baselines()
    assert len(b) == len(arr.positions) - 1


def test_baselines_are_relative_to_reference():
    arr = _array()
    b = arr.baselines()
    assert b[0] == [0.10, 0.0, 0.0]


# --- pairwise separation --------------------------------------------------

def test_sky_separation_large():
    vecs, _ = crpa.estimate_directions(_array(), _sky())
    assert crpa.mean_pairwise_separation_deg([v for _, v in vecs]) > crpa.MIN_MEAN_SEPARATION_DEG


def test_single_source_separation_small():
    vecs, _ = crpa.estimate_directions(_array(), _spoof())
    assert crpa.mean_pairwise_separation_deg([v for _, v in vecs]) < crpa.MIN_MEAN_SEPARATION_DEG


def test_separation_identical_vectors_zero():
    v = crpa.enu_los(45, 30)
    assert crpa.mean_pairwise_separation_deg([v] * 5) == pytest.approx(0.0, abs=1e-9)


def test_separation_orthogonal_is_ninety():
    vs = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    assert crpa.mean_pairwise_separation_deg(vs) == pytest.approx(90.0, abs=1e-9)


def test_separation_single_vector_zero():
    assert crpa.mean_pairwise_separation_deg([(1.0, 0.0, 0.0)]) == 0.0


def test_separation_empty_zero():
    assert crpa.mean_pairwise_separation_deg([]) == 0.0


def test_separation_drops_zero_vectors():
    vs = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    # the zero vector is ignored -> just the two orthogonal ones
    assert crpa.mean_pairwise_separation_deg(vs) == pytest.approx(90.0, abs=1e-9)


# --- eigen-concentration --------------------------------------------------

def test_concentration_identical_is_one():
    v = crpa.enu_los(45, 30)
    assert crpa.eigen_concentration([v] * 8) == pytest.approx(1.0, abs=1e-9)


def test_concentration_isotropic_is_third():
    # three orthonormal axes -> perfectly isotropic scatter -> 1/3
    vs = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    assert crpa.eigen_concentration(vs) == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_concentration_bounded():
    vecs, _ = crpa.estimate_directions(_array(), _sky())
    c = crpa.eigen_concentration([v for _, v in vecs])
    assert 1.0 / 3.0 - 1e-9 <= c <= 1.0 + 1e-9


def test_concentration_sky_below_suspect():
    vecs, _ = crpa.estimate_directions(_array(), _sky())
    assert crpa.eigen_concentration([v for _, v in vecs]) < crpa.CONCENTRATION_SUSPECT


def test_concentration_spoof_above_suspect():
    vecs, _ = crpa.estimate_directions(_array(), _spoof())
    assert crpa.eigen_concentration([v for _, v in vecs]) > crpa.CONCENTRATION_SUSPECT


def test_concentration_empty_zero():
    assert crpa.eigen_concentration([]) == 0.0


def test_concentration_all_zero_vectors_zero():
    assert crpa.eigen_concentration([(0.0, 0.0, 0.0)] * 4) == 0.0


# --- symmetric-3x3 eigensolver (property sweep) ---------------------------

def _sym3_from_vectors(vs):
    return crpa.scatter_matrix(vs)


@pytest.mark.parametrize("seed", list(range(20)))
def test_eigvals_match_trace_and_det(seed):
    # deterministic pseudo-random symmetric matrix from a small LCG
    st = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    vals = []
    for _ in range(6):
        st = (st * 1103515245 + 12345) & 0x7FFFFFFF
        vals.append((st % 2000) / 100.0 - 10.0)
    a, b, c, d, e, f = vals
    A = [[a, d, e], [d, b, f], [e, f, c]]
    e1, e2, e3 = crpa._eigvals_sym3(A)
    trace = a + b + c
    det = (a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e))
    assert e1 >= e2 >= e3
    assert e1 + e2 + e3 == pytest.approx(trace, abs=1e-6)
    assert e1 * e2 * e3 == pytest.approx(det, abs=1e-4)


def test_eigvals_diagonal_matrix():
    A = [[3.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 1.0]]
    assert crpa._eigvals_sym3(A) == pytest.approx((7.0, 3.0, 1.0))


def test_eigvals_descending_order():
    vs = [crpa.enu_los(10, 20), crpa.enu_los(200, 70), crpa.enu_los(120, 5)]
    e1, e2, e3 = crpa._eigvals_sym3(_sym3_from_vectors(vs))
    assert e1 >= e2 >= e3 >= -1e-9


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.5, 100.0])
def test_eigvals_scale_linearly(scale):
    vs = [crpa.enu_los(10, 20), crpa.enu_los(200, 70), crpa.enu_los(120, 5),
          crpa.enu_los(300, 40)]
    S = _sym3_from_vectors(vs)
    base = crpa._eigvals_sym3(S)
    scaled = crpa._eigvals_sym3([[scale * S[i][j] for j in range(3)] for i in range(3)])
    for b, s in zip(base, scaled):
        assert s == pytest.approx(scale * b, rel=1e-6, abs=1e-9)


# --- check_directions -----------------------------------------------------

def test_check_directions_sky_clean():
    vecs, _ = crpa.estimate_directions(_array(), _sky())
    r = crpa.check_directions(vecs)
    assert r["available"] is True
    assert r["single_source"] is False
    assert r["confidence"] == 0.0
    assert r["narrow_spread"] is False
    assert r["rank_one_covariance"] is False


def test_check_directions_spoof_flagged():
    vecs, _ = crpa.estimate_directions(_array(), _spoof())
    r = crpa.check_directions(vecs)
    assert r["available"] is True
    assert r["single_source"] is True
    assert r["confidence"] > 0.0
    assert r["narrow_spread"] is True
    assert r["rank_one_covariance"] is True


def test_check_directions_accepts_bare_tuples():
    vs = [crpa.enu_los(45, 30) for _ in range(5)]
    r = crpa.check_directions(vs)
    assert r["single_source"] is True


def test_check_directions_too_few_unavailable():
    vecs, _ = crpa.estimate_directions(_array(), _sky()[:2])
    r = crpa.check_directions(vecs)
    assert r["available"] is False
    assert r["single_source"] is False


def test_check_directions_confidence_bounded():
    vecs, _ = crpa.estimate_directions(_array(), _spoof())
    r = crpa.check_directions(vecs)
    assert 0.0 <= r["confidence"] <= 1.0


def test_check_directions_n_sats():
    vecs, _ = crpa.estimate_directions(_array(), _sky())
    r = crpa.check_directions(vecs)
    assert r["n_sats"] == 6


def test_check_directions_reports_metrics():
    vecs, _ = crpa.estimate_directions(_array(), _sky())
    r = crpa.check_directions(vecs)
    assert "mean_separation_deg" in r
    assert "eigen_concentration" in r
    assert r["separation_threshold_deg"] == crpa.MIN_MEAN_SEPARATION_DEG
    assert r["concentration_threshold"] == crpa.CONCENTRATION_SUSPECT


def test_check_directions_ignores_failed_estimates():
    vs = [crpa.enu_los(10, 60), crpa.enu_los(120, 40), (0.0, 0.0, 0.0),
          crpa.enu_los(250, 55), crpa.enu_los(300, 25)]
    r = crpa.check_directions(vs)
    # four usable spread directions -> available and not single-source
    assert r["available"] is True
    assert r["n_sats"] == 4
    assert r["single_source"] is False


# --- check (end-to-end) ---------------------------------------------------

def test_check_sky_clean():
    r = crpa.check(_array(), _sky())
    assert r["available"] is True
    assert r["single_source"] is False
    assert r["confidence"] == 0.0
    assert r["n_noisy_channels"] == 0


def test_check_spoof_flagged():
    r = crpa.check(_array(), _spoof())
    assert r["available"] is True
    assert r["single_source"] is True
    assert r["confidence"] > 0.0


def test_check_reports_residual_fields():
    r = crpa.check(_array(), _sky())
    assert "max_residual_rad" in r
    assert "noisy_channels" in r
    assert r["max_residual_rad"] == pytest.approx(0.0, abs=1e-6)


def test_check_too_few_sats_unavailable():
    r = crpa.check(_array(), _sky()[:2])
    assert r["available"] is False
    assert r["single_source"] is False


def test_check_ill_posed_array_unavailable():
    arr = crpa.AntennaArray(positions=[(0, 0, 0), (0.1, 0, 0),
                                       (0, 0.1, 0), (0.1, 0.1, 0)])
    r = crpa.check(arr, crpa.synth_measurements(arr, _sky_sources()))
    assert r["available"] is False
    assert r["single_source"] is False


def test_check_flags_noisy_channel():
    # an overdetermined array (5 baselines, 3 unknowns) leaves a residual when a
    # channel's phases are inconsistent with a single plane wave
    arr = _bigger_array()
    meas = crpa.synth_measurements(arr, _sky_sources())
    meas[0].phases = [p + (5.0 if i == 2 else 0.0) for i, p in enumerate(meas[0].phases)]
    r = crpa.check(arr, meas)
    assert meas[0].sat_id in r["noisy_channels"]
    assert r["n_noisy_channels"] >= 1


def test_check_noisy_channel_does_not_force_spoof():
    # a noisy channel is an awareness field, not a spoof vote
    arr = _bigger_array()
    meas = crpa.synth_measurements(arr, _sky_sources())
    meas[0].phases = [p + (5.0 if i == 2 else 0.0) for i, p in enumerate(meas[0].phases)]
    r = crpa.check(arr, meas)
    assert r["single_source"] is False


@pytest.mark.parametrize("n", [4, 5, 6, 8, 10])
def test_check_spoof_flagged_various_counts(n):
    r = crpa.check(_array(), _spoof(n=n))
    assert r["single_source"] is True
    assert r["n_sats"] == n


@pytest.mark.parametrize("az,el", [(0, 30), (90, 45), (180, 20), (270, 60), (330, 10)])
def test_check_spoof_from_any_direction_flagged(az, el):
    r = crpa.check(_array(), _spoof(az=az, el=el))
    assert r["single_source"] is True


@pytest.mark.parametrize("spread", [0.0, 0.1, 0.5, 1.0])
def test_check_tight_cluster_flagged(spread):
    r = crpa.check(_array(), _spoof(spread=spread))
    assert r["single_source"] is True


@pytest.mark.parametrize("wavelength", [0.19, 0.244, 0.05, 1.0])
def test_check_independent_of_wavelength(wavelength):
    # the detector depends only on geometry, so wavelength choice must not matter
    arr = _array(wavelength=wavelength)
    assert crpa.check(arr, _sky(arr))["single_source"] is False
    assert crpa.check(arr, _spoof(arr))["single_source"] is True


# --- threshold configurability & monotonicity -----------------------------

def test_separation_threshold_configurable():
    vecs, _ = crpa.estimate_directions(_array(), _spoof())
    # with a near-zero threshold even a tight cluster won't trip the spread test
    r = crpa.check_directions(vecs, min_sep_deg=0.001, concentration_suspect=1.5)
    assert r["narrow_spread"] is False
    assert r["rank_one_covariance"] is False
    assert r["single_source"] is False


def test_concentration_threshold_configurable():
    vecs, _ = crpa.estimate_directions(_array(), _sky())
    # forcing a low concentration threshold makes even a real sky look rank-one
    r = crpa.check_directions(vecs, min_sep_deg=0.0, concentration_suspect=0.4)
    assert r["rank_one_covariance"] is True
    assert r["single_source"] is True


def test_confidence_grows_as_cluster_tightens():
    loose = crpa.check(_array(), _spoof(spread=3.0))
    tight = crpa.check(_array(), _spoof(spread=0.1))
    # both may flag, but the tighter cluster is at least as confident
    assert tight["confidence"] >= loose["confidence"]


def test_synth_measurements_shape():
    arr = _array()
    meas = crpa.synth_measurements(arr, _sky_sources())
    assert len(meas) == len(_sky_sources())
    assert all(len(m.phases) == len(arr.positions) for m in meas)
    assert all(isinstance(m, crpa.SatMeasurement) for m in meas)


def test_module_registered_in_all():
    import spoofwatch
    assert "crpa" in spoofwatch.__all__
