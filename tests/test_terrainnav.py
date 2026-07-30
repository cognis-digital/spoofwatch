import math

import pytest

from spoofwatch import terrainnav as tn
from spoofwatch.terrainnav import ReferenceProfile


# a deterministic, information-rich reference terrain strip (metres)
REF = [100, 120, 90, 140, 200, 170, 110, 130, 210, 250,
       180, 90, 60, 100, 160, 220, 190, 130, 80, 120,
       150, 210, 260, 200, 140, 90, 70, 110, 170, 230]


def _slice(ref, start, n):
    return list(ref[start:start + n])


# ----------------------------------------------------------------------------
# metric primitives
# ----------------------------------------------------------------------------

def test_mad_zero_for_identical():
    a = [1.0, 2.0, 3.0, 4.0]
    assert tn.mad(a, a) == 0.0


def test_msd_and_rmsd_relationship():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.5, 2.5, 2.0, 5.0]
    assert math.isclose(tn.rmsd(a, b), math.sqrt(tn.msd(a, b)))


def test_mad_known_value():
    a = [0.0, 0.0, 0.0, 0.0]
    b = [1.0, 2.0, 3.0, 4.0]
    assert math.isclose(tn.mad(a, b), 2.5)


def test_msd_known_value():
    a = [0.0, 0.0, 0.0, 0.0]
    b = [1.0, 1.0, 1.0, 1.0]
    assert math.isclose(tn.msd(a, b), 1.0)


def test_ncc_perfect_match_is_one():
    a = [1.0, 3.0, 2.0, 5.0, 4.0]
    assert math.isclose(tn.ncc(a, a), 1.0)


def test_ncc_anticorrelated_is_minus_one():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [4.0, 3.0, 2.0, 1.0]
    assert math.isclose(tn.ncc(a, b), -1.0, abs_tol=1e-9)


def test_ncc_bias_invariant():
    a = [10.0, 30.0, 20.0, 50.0, 40.0]
    b = [x + 137.0 for x in a]          # constant altimeter bias
    assert math.isclose(tn.ncc(a, b), 1.0)


def test_ncc_scale_invariant():
    a = [10.0, 30.0, 20.0, 50.0, 40.0]
    b = [x * 3.0 for x in a]
    assert math.isclose(tn.ncc(a, b), 1.0)


def test_ncc_flat_profile_is_zero():
    a = [5.0, 5.0, 5.0, 5.0]
    b = [1.0, 2.0, 3.0, 4.0]
    assert tn.ncc(a, b) == 0.0


def test_ncc_bounds():
    a = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    b = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0]
    v = tn.ncc(a, b)
    assert -1.0 <= v <= 1.0


@pytest.mark.parametrize("fn", [tn.mad, tn.msd, tn.rmsd, tn.ncc])
def test_metric_length_mismatch_raises(fn):
    with pytest.raises(ValueError):
        fn([1.0, 2.0, 3.0], [1.0, 2.0])


# ----------------------------------------------------------------------------
# resampling
# ----------------------------------------------------------------------------

def test_resample_same_spacing_identity():
    e = [10.0, 20.0, 15.0, 30.0]
    out = tn.resample(e, 1.0, 1.0)
    assert out == e


def test_resample_halving_spacing_interpolates_midpoints():
    e = [0.0, 10.0, 20.0]           # 2 km total at 1 km spacing
    out = tn.resample(e, 1.0, 0.5)  # -> 5 samples at 0.5 km
    assert len(out) == 5
    assert math.isclose(out[1], 5.0)
    assert math.isclose(out[3], 15.0)


def test_resample_preserves_endpoints():
    e = [3.0, 8.0, 1.0, 9.0, 4.0]
    out = tn.resample(e, 2.0, 0.7)
    assert math.isclose(out[0], 3.0)
    assert math.isclose(out[-1], 4.0)


def test_resample_single_sample():
    assert tn.resample([42.0], 1.0, 0.5) == [42.0]


def test_resample_empty():
    assert tn.resample([], 1.0, 0.5) == []


def test_resample_bad_spacing_raises():
    with pytest.raises(ValueError):
        tn.resample([1.0, 2.0], 0.0, 1.0)
    with pytest.raises(ValueError):
        tn.resample([1.0, 2.0], 1.0, -1.0)


def test_resample_length_for_coarsening():
    e = list(range(11))             # 10 km at 1 km
    out = tn.resample([float(x) for x in e], 1.0, 2.0)  # -> 6 samples at 2 km
    assert len(out) == 6


# ----------------------------------------------------------------------------
# sliding-window scoring
# ----------------------------------------------------------------------------

def test_slide_scores_count():
    m = _slice(REF, 5, 6)
    scores = tn.slide_scores(m, REF, "ncc")
    assert len(scores) == len(REF) - len(m) + 1


def test_slide_scores_offsets_are_sequential():
    m = _slice(REF, 3, 5)
    scores = tn.slide_scores(m, REF, "mad")
    assert [off for off, _ in scores] == list(range(len(REF) - len(m) + 1))


def test_slide_window_too_short_raises():
    with pytest.raises(ValueError):
        tn.slide_scores([1.0, 2.0, 3.0], REF, "ncc")


def test_slide_window_longer_than_ref_raises():
    with pytest.raises(ValueError):
        tn.slide_scores([float(i) for i in range(len(REF) + 1)], REF, "ncc")


def test_slide_unknown_metric_raises():
    with pytest.raises(ValueError):
        tn.slide_scores(_slice(REF, 0, 5), REF, "bogus")


# ----------------------------------------------------------------------------
# best_match — offset recovery
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("start", [0, 3, 7, 11, 18, 24])
@pytest.mark.parametrize("metric", ["mad", "msd", "rmsd", "ncc"])
def test_exact_offset_recovered(start, metric):
    m = _slice(REF, start, 6)
    res = tn.best_match(m, REF, metric)
    assert res["offset_samples"] == start


@pytest.mark.parametrize("metric", ["mad", "msd", "rmsd", "ncc"])
def test_exact_match_high_confidence(metric):
    m = _slice(REF, 8, 7)
    res = tn.best_match(m, REF, metric)
    assert res["confidence"] > 0.5
    assert res["ambiguous"] is False


@pytest.mark.parametrize("metric", ["mad", "msd", "rmsd"])
def test_exact_match_zero_error_score(metric):
    m = _slice(REF, 4, 6)
    res = tn.best_match(m, REF, metric)
    assert math.isclose(res["score"], 0.0, abs_tol=1e-9)


def test_exact_match_ncc_score_one():
    m = _slice(REF, 4, 6)
    res = tn.best_match(m, REF, "ncc")
    assert math.isclose(res["score"], 1.0, abs_tol=1e-6)


@pytest.mark.parametrize("start", [2, 6, 10, 15, 20])
def test_offset_recovered_with_bias_using_ncc(start):
    # a constant altimeter bias must not move the NCC winner
    m = [x + 250.0 for x in _slice(REF, start, 6)]
    res = tn.best_match(m, REF, "ncc")
    assert res["offset_samples"] == start


@pytest.mark.parametrize("start", [1, 5, 9, 13, 19, 23])
@pytest.mark.parametrize("noise", [1.0, 3.0, 6.0])
def test_offset_recovered_under_small_noise(start, noise):
    # deterministic zig-zag "noise" small relative to terrain relief
    m = [v + (noise if i % 2 == 0 else -noise)
         for i, v in enumerate(_slice(REF, start, 7))]
    res = tn.best_match(m, REF, "ncc")
    assert res["offset_samples"] == start


def test_n_offsets_reported():
    m = _slice(REF, 0, 8)
    res = tn.best_match(m, REF, "ncc")
    assert res["n_offsets"] == len(REF) - len(m) + 1


# ----------------------------------------------------------------------------
# flat terrain & ambiguity
# ----------------------------------------------------------------------------

def test_flat_terrain_flagged_low_confidence():
    flat_ref = [100.0] * 20
    m = [100.0] * 6
    res = tn.best_match(m, flat_ref, "ncc")
    assert res["flat_terrain"] is True
    assert res["ambiguous"] is True
    assert res["confidence"] < 0.2


def test_low_relief_penalizes_confidence():
    m = _slice(REF, 8, 6)
    rich = tn.best_match(m, REF, "ncc")["confidence"]
    # nearly flat window -> low relief -> lower confidence
    flat_m = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0]
    flat_ref = flat_m + [100.0, 101.0, 100.0, 101.0, 100.0, 101.0]
    poor = tn.best_match(flat_m, flat_ref, "ncc")["confidence"]
    assert rich > poor


def test_periodic_terrain_is_ambiguous():
    # a repeating ridge pattern aliases: many equally good offsets
    period = [10.0, 40.0, 20.0, 60.0]
    ref = period * 8
    m = period + period[:2]           # 6 samples spanning >1 period
    res = tn.best_match(m, ref, "ncc")
    assert res["ambiguous"] is True


def test_relief_reported_matches_std():
    m = [10.0, 20.0, 30.0, 40.0]
    mu = sum(m) / len(m)
    std = math.sqrt(sum((x - mu) ** 2 for x in m) / len(m))
    res = tn.best_match(m, REF[:15] + m + REF[:5], "mad")
    # find the exact window we planted
    assert math.isclose(res["relief_m"], round(std, 3), abs_tol=1e-3)


def test_margin_none_when_only_one_offset():
    m = list(REF)[:-0] if False else list(REF)  # full length -> single offset
    res = tn.best_match(m, REF, "ncc")
    assert res["n_offsets"] == 1
    assert res["margin"] is None


# ----------------------------------------------------------------------------
# terrain_fix — geographic projection
# ----------------------------------------------------------------------------

def test_terrain_fix_offset_and_along_km():
    ref = ReferenceProfile(elevations=REF, spacing_km=0.5)
    m = _slice(REF, 10, 6)
    res = tn.terrain_fix(m, ref, "ncc")
    assert res["offset_samples"] == 10
    assert math.isclose(res["window_start_km"], 10 * 0.5)
    assert math.isclose(res["fix_along_km"], 10 * 0.5 + 5 * 0.5)


def test_terrain_fix_no_anchor_no_latlon():
    ref = ReferenceProfile(elevations=REF, spacing_km=0.5)
    res = tn.terrain_fix(_slice(REF, 4, 6), ref, "ncc")
    assert res["fix_lat"] is None
    assert res["fix_lon"] is None


def test_terrain_fix_with_anchor_east_bearing():
    ref = ReferenceProfile(elevations=REF, spacing_km=1.0,
                           start_lat=40.0, start_lon=-100.0, bearing_deg=90.0)
    m = _slice(REF, 8, 6)
    res = tn.terrain_fix(m, ref, "ncc")
    # bearing due east -> latitude essentially unchanged, longitude increases
    assert math.isclose(res["fix_lat"], 40.0, abs_tol=1e-6)
    assert res["fix_lon"] > -100.0


def test_terrain_fix_with_anchor_north_bearing():
    ref = ReferenceProfile(elevations=REF, spacing_km=1.0,
                           start_lat=40.0, start_lon=-100.0, bearing_deg=0.0)
    m = _slice(REF, 5, 6)
    res = tn.terrain_fix(m, ref, "ncc")
    assert res["fix_lat"] > 40.0
    assert math.isclose(res["fix_lon"], -100.0, abs_tol=1e-6)


def test_terrain_fix_latlon_matches_projection():
    ref = ReferenceProfile(elevations=REF, spacing_km=2.0,
                           start_lat=35.0, start_lon=10.0, bearing_deg=45.0)
    m = _slice(REF, 7, 6)
    res = tn.terrain_fix(m, ref, "ncc")
    exp = ref.position_at_km(res["fix_along_km"])
    assert math.isclose(res["fix_lat"], round(exp[0], 6))
    assert math.isclose(res["fix_lon"], round(exp[1], 6))


def test_terrain_fix_resamples_measured():
    # measured taken at half the reference spacing -> must be resampled first
    ref = ReferenceProfile(elevations=REF, spacing_km=1.0,
                           start_lat=0.0, start_lon=0.0, bearing_deg=90.0)
    fine = tn.resample(_slice(REF, 6, 6), 1.0, 0.5)   # denser sampling
    res = tn.terrain_fix(fine, ref, "ncc", measured_spacing_km=0.5)
    assert res["offset_samples"] == 6
    assert res["window_len_samples"] == 6


def test_terrain_fix_window_len_reported():
    ref = ReferenceProfile(elevations=REF, spacing_km=0.5)
    m = _slice(REF, 3, 9)
    res = tn.terrain_fix(m, ref, "ncc")
    assert res["window_len_samples"] == 9


def test_reference_position_at_km_none_without_anchor():
    ref = ReferenceProfile(elevations=REF, spacing_km=0.5)
    assert ref.position_at_km(3.0) is None


def test_reference_position_at_km_projects():
    ref = ReferenceProfile(elevations=REF, spacing_km=1.0,
                           start_lat=0.0, start_lon=0.0, bearing_deg=90.0)
    lat, lon = ref.position_at_km(111.0)
    assert math.isclose(lat, 0.0, abs_tol=1e-9)
    assert math.isclose(lon, 1.0, abs_tol=1e-2)


# ----------------------------------------------------------------------------
# determinism & invariants
# ----------------------------------------------------------------------------

def test_determinism_repeated_calls():
    m = _slice(REF, 9, 7)
    r1 = tn.best_match(m, REF, "ncc")
    r2 = tn.best_match(m, REF, "ncc")
    assert r1 == r2


def test_confidence_in_unit_interval():
    for start in range(0, 20, 3):
        for metric in tn.METRICS:
            res = tn.best_match(_slice(REF, start, 6), REF, metric)
            assert 0.0 <= res["confidence"] <= 1.0


def test_score_rounding_shape():
    res = tn.best_match(_slice(REF, 2, 6), REF, "ncc")
    assert isinstance(res["offset_samples"], int)
    assert isinstance(res["ambiguous"], bool)
    assert isinstance(res["flat_terrain"], bool)


@pytest.mark.parametrize("metric", ["mad", "msd", "rmsd", "ncc"])
def test_margin_nonnegative_when_present(metric):
    res = tn.best_match(_slice(REF, 6, 6), REF, metric)
    if res["margin"] is not None:
        assert res["margin"] >= 0.0


def test_higher_is_better_only_ncc():
    assert tn.HIGHER_IS_BETTER == {"ncc"}
    for m in ("mad", "msd", "rmsd"):
        assert m not in tn.HIGHER_IS_BETTER


def test_metrics_tuple_complete():
    assert set(tn.METRICS) == {"mad", "msd", "rmsd", "ncc"}


@pytest.mark.parametrize("start", [0, 4, 8, 12, 16, 20, 24])
def test_fix_along_km_monotonic_with_offset(start):
    ref = ReferenceProfile(elevations=REF, spacing_km=0.5)
    res = tn.terrain_fix(_slice(REF, start, 6), ref, "ncc")
    assert math.isclose(res["fix_along_km"], (start + 5) * 0.5)


def test_module_exported():
    import spoofwatch
    assert "terrainnav" in spoofwatch.__all__
