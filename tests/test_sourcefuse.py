"""Multi-source PNT arbiter: inverse-variance fusion + fault detection/exclusion."""
import math

import pytest

from spoofwatch.sourcefuse import (
    FusedFix,
    NavFix,
    PntArbiter,
    SourceClass,
    TrustLevel,
    fuse_sources,
)


def fix(name, x, y, sigma, cls):
    return NavFix(name, x, y, sigma, cls)


# ---- NavFix ----

def test_navfix_bad_sigma():
    with pytest.raises(ValueError):
        NavFix("s", 0, 0, 0, SourceClass.GNSS)


# ---- fusion basics ----

def test_empty_is_unsafe():
    f = fuse_sources([])
    assert f.trust is TrustLevel.UNSAFE
    assert math.isinf(f.sigma_m)


def test_single_source_unsafe_but_positioned():
    f = fuse_sources([fix("ins", 10, 20, 5, SourceClass.INERTIAL)])
    assert f.trust is TrustLevel.UNSAFE   # only 1 class
    assert f.x == pytest.approx(10) and f.y == pytest.approx(20)


def test_inverse_variance_weighting():
    # Tight source dominates the fused position.
    fixes = [fix("gnss", 0, 0, 1.0, SourceClass.GNSS),
             fix("ins", 100, 0, 10.0, SourceClass.INERTIAL)]
    f = fuse_sources(fixes)
    assert f.x < 10   # pulled strongly toward the tight (sigma=1) source


def test_fused_sigma_smaller_than_any_source():
    fixes = [fix("a", 0, 0, 4.0, SourceClass.GNSS),
             fix("b", 0, 0, 4.0, SourceClass.VISUAL),
             fix("c", 0, 0, 4.0, SourceClass.TERRAIN)]
    f = fuse_sources(fixes)
    assert f.sigma_m < 4.0


def test_three_classes_trusted():
    fixes = [fix("gnss", 0, 0, 2, SourceClass.GNSS),
             fix("vio", 1, 0, 2, SourceClass.VISUAL),
             fix("terr", 0, 1, 2, SourceClass.TERRAIN)]
    f = fuse_sources(fixes)
    assert f.trust is TrustLevel.TRUSTED


def test_two_classes_degraded():
    fixes = [fix("gnss", 0, 0, 2, SourceClass.GNSS),
             fix("vio", 1, 0, 2, SourceClass.VISUAL)]
    assert fuse_sources(fixes).trust is TrustLevel.DEGRADED


# ---- fault detection & exclusion ----

def test_spoofed_source_excluded():
    # Three consistent sources + one wild outlier (spoofed GNSS) far away.
    fixes = [fix("ins", 0, 0, 3, SourceClass.INERTIAL),
             fix("vio", 1, 0, 3, SourceClass.VISUAL),
             fix("terr", 0, 1, 3, SourceClass.TERRAIN),
             fix("spoof", 5000, 5000, 3, SourceClass.GNSS)]
    f = fuse_sources(fixes)
    assert "spoof" in f.sources_rejected
    assert "spoof" not in f.sources_used
    # Fused position stays near the honest cluster.
    assert math.hypot(f.x, f.y) < 50


def test_consistent_sources_none_rejected():
    fixes = [fix("a", 0, 0, 2, SourceClass.GNSS),
             fix("b", 1, 1, 2, SourceClass.VISUAL),
             fix("c", -1, 0, 2, SourceClass.TERRAIN)]
    f = fuse_sources(fixes)
    assert f.sources_rejected == ()
    assert len(f.sources_used) == 3


def test_excluding_outlier_leaves_two_degraded():
    fixes = [fix("gnss", 0, 0, 2, SourceClass.GNSS),
             fix("vio", 1, 0, 2, SourceClass.VISUAL),
             fix("spoof", 9000, 0, 2, SourceClass.CELESTIAL)]
    f = fuse_sources(fixes)
    assert "spoof" in f.sources_rejected
    assert f.trust is TrustLevel.DEGRADED


def test_gate_controls_strictness():
    fixes = [fix("a", 0, 0, 2, SourceClass.GNSS),
             fix("b", 0, 0, 2, SourceClass.VISUAL),
             fix("c", 12, 0, 2, SourceClass.TERRAIN)]   # ~4.2 sigma off
    strict = fuse_sources(fixes, gate=3.0)
    loose = fuse_sources(fixes, gate=10.0)
    assert "c" in strict.sources_rejected
    assert "c" not in loose.sources_rejected


def test_to_dict():
    d = fuse_sources([fix("a", 0, 0, 2, SourceClass.GNSS),
                      fix("b", 1, 0, 2, SourceClass.VISUAL)]).to_dict()
    for k in ("x", "y", "sigma_m", "trust", "sources_used", "sources_rejected", "reasoning"):
        assert k in d


# ---- arbiter coasting ----

def test_arbiter_coasts_when_unsafe():
    arb = PntArbiter(coast_growth_m_per_epoch=5.0)
    # Good fix first (establishes last_sigma).
    good = [fix("gnss", 0, 0, 2, SourceClass.GNSS),
            fix("vio", 0, 0, 2, SourceClass.VISUAL),
            fix("terr", 0, 0, 2, SourceClass.TERRAIN)]
    arb.step(good)
    assert arb.coasting_epochs == 0
    # Now only inertial -> unsafe -> uncertainty grows each epoch.
    only_ins = [fix("ins", 0, 0, 2, SourceClass.INERTIAL)]
    f1 = arb.step(only_ins)
    f2 = arb.step(only_ins)
    assert f1.trust is TrustLevel.UNSAFE
    assert f2.sigma_m > f1.sigma_m
    assert arb.coasting_epochs == 2


def test_arbiter_resets_on_recovery():
    arb = PntArbiter()
    only_ins = [fix("ins", 0, 0, 2, SourceClass.INERTIAL)]
    arb.step(only_ins)
    assert arb.coasting_epochs == 1
    good = [fix("gnss", 0, 0, 2, SourceClass.GNSS),
            fix("vio", 0, 0, 2, SourceClass.VISUAL),
            fix("terr", 0, 0, 2, SourceClass.TERRAIN)]
    arb.step(good)
    assert arb.coasting_epochs == 0


# ---- property sweeps ----

@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_more_sources_tighter_sigma(n):
    classes = list(SourceClass)
    fixes = [fix(f"s{i}", 0, 0, 3.0, classes[i % len(classes)]) for i in range(n)]
    f = fuse_sources(fixes)
    assert f.sigma_m == pytest.approx(3.0 / math.sqrt(n), rel=1e-6)


@pytest.mark.parametrize("dx", [50, 100, 500, 5000])
def test_far_outlier_always_excluded(dx):
    fixes = [fix("a", 0, 0, 2, SourceClass.GNSS),
             fix("b", 1, 0, 2, SourceClass.VISUAL),
             fix("c", 0, 1, 2, SourceClass.TERRAIN),
             fix("bad", dx, dx, 2, SourceClass.INERTIAL)]
    f = fuse_sources(fixes)
    assert "bad" in f.sources_rejected


@pytest.mark.parametrize("nclasses,expected", [
    (1, TrustLevel.UNSAFE), (2, TrustLevel.DEGRADED),
    (3, TrustLevel.TRUSTED), (4, TrustLevel.TRUSTED),
])
def test_trust_level_by_class_count(nclasses, expected):
    classes = list(SourceClass)[:nclasses]
    fixes = [fix(f"s{i}", i * 0.1, 0, 2, classes[i]) for i in range(nclasses)]
    assert fuse_sources(fixes).trust is expected


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 5.0])
def test_identical_sources_keep_position(sigma):
    fixes = [fix("a", 7, 3, sigma, SourceClass.GNSS),
             fix("b", 7, 3, sigma, SourceClass.VISUAL),
             fix("c", 7, 3, sigma, SourceClass.TERRAIN)]
    f = fuse_sources(fixes)
    assert f.x == pytest.approx(7) and f.y == pytest.approx(3)
