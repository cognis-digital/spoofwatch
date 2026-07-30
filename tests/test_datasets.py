"""Tests for the GNSS dataset registry + C/N0 plausibility gate."""
import pytest

from spoofwatch.datasets import (
    GNSS_DATASETS,
    SIGNAL_POWER_REF,
    Cn0Flag,
    DatasetKind,
    GnssDataset,
    SignalPowerRef,
    cn0_plausible,
    datasets_by_kind,
    open_datasets,
)

VALID_LICENSES = {"open", "research", "non-commercial", "proprietary"}


# ---- registry ----

def test_registry_populated():
    assert len(GNSS_DATASETS) >= 12


def test_entries_well_formed():
    for d in GNSS_DATASETS:
        assert d.url.startswith("http") or d.kind is DatasetKind.TAXONOMY
        assert d.name and d.bands and d.size and d.note
        assert d.license in VALID_LICENSES
        assert isinstance(d.kind, DatasetKind)


def test_names_unique():
    names = [d.name for d in GNSS_DATASETS]
    assert len(names) == len(set(names))


def test_canonical_datasets_present():
    names = {d.name for d in GNSS_DATASETS}
    for expected in ("TEXBAT", "OAKBAT", "Jammertest-2024", "GPSJam"):
        assert expected in names


def test_by_kind_filter():
    spoof = datasets_by_kind(DatasetKind.SPOOFING_IQ)
    assert spoof and all(d.kind is DatasetKind.SPOOFING_IQ for d in spoof)
    assert any(d.name == "TEXBAT" for d in spoof)


def test_kind_partition_covers_registry():
    total = sum(len(datasets_by_kind(k)) for k in DatasetKind)
    assert total == len(GNSS_DATASETS)


def test_open_datasets_redistributable():
    for d in open_datasets():
        assert d.redistributable and d.license == "open"
    names = {d.name for d in open_datasets()}
    assert "OAKBAT" in names and "GPSJam" in names


def test_to_dict_keys():
    d = GNSS_DATASETS[0].to_dict()
    for k in ("name", "url", "kind", "bands", "size", "license", "note"):
        assert k in d


# ---- signal-power reference ----

def test_reference_has_core_signals():
    for sig in ("L1CA", "L5", "E1"):
        assert sig in SIGNAL_POWER_REF
        assert isinstance(SIGNAL_POWER_REF[sig], SignalPowerRef)


def test_l1ca_reference_values():
    r = SIGNAL_POWER_REF["L1CA"]
    assert r.min_power_dbw == pytest.approx(-158.5)
    assert r.nominal_cn0_dbhz == pytest.approx(45.5)


# ---- C/N0 plausibility gate ----

def test_nominal_cn0_not_suspicious():
    a = cn0_plausible(45.0, "L1CA", margin_db=6.0)
    assert a.flag is Cn0Flag.NOMINAL
    assert not a.suspicious


def test_overpowered_flags_high():
    a = cn0_plausible(58.0, "L1CA", margin_db=6.0)
    assert a.flag is Cn0Flag.HIGH
    assert a.suspicious
    assert "spoofer" in a.reasoning


def test_degraded_flags_low():
    a = cn0_plausible(30.0, "L1CA", margin_db=6.0)
    assert a.flag is Cn0Flag.LOW
    assert a.suspicious
    assert "jamming" in a.reasoning or "obstruction" in a.reasoning


def test_unknown_signal():
    a = cn0_plausible(45.0, "BOGUS")
    assert a.flag is Cn0Flag.UNKNOWN
    assert a.nominal_cn0_dbhz is None
    assert not a.suspicious


def test_deviation_sign():
    hi = cn0_plausible(55.0, "L1CA")
    lo = cn0_plausible(35.0, "L1CA")
    assert hi.deviation_db > 0
    assert lo.deviation_db < 0


def test_negative_margin_raises():
    with pytest.raises(ValueError):
        cn0_plausible(45.0, "L1CA", margin_db=-1.0)


def test_case_insensitive_signal():
    a = cn0_plausible(45.5, "l1ca")
    assert a.flag is Cn0Flag.NOMINAL


def test_assessment_to_dict():
    d = cn0_plausible(58.0, "L1CA").to_dict()
    for k in ("signal", "measured_cn0_dbhz", "nominal_cn0_dbhz", "flag",
              "deviation_db", "suspicious", "reasoning"):
        assert k in d


# ---- property sweeps ----

@pytest.mark.parametrize("cn0", [20, 25, 30, 35, 40, 45, 50, 55, 60])
def test_flag_monotonic_in_cn0(cn0):
    a = cn0_plausible(cn0, "L1CA", margin_db=6.0)
    nominal = SIGNAL_POWER_REF["L1CA"].nominal_cn0_dbhz
    if cn0 > nominal + 6.0:
        assert a.flag is Cn0Flag.HIGH
    elif cn0 < nominal - 6.0:
        assert a.flag is Cn0Flag.LOW
    else:
        assert a.flag is Cn0Flag.NOMINAL


@pytest.mark.parametrize("sig", ["L1CA", "L2C", "L5", "E1", "E5A"])
def test_all_signals_gate_nominal(sig):
    ref = SIGNAL_POWER_REF[sig]
    a = cn0_plausible(ref.nominal_cn0_dbhz, sig)
    assert a.flag is Cn0Flag.NOMINAL


@pytest.mark.parametrize("margin", [2.0, 4.0, 6.0, 10.0])
def test_wider_margin_more_tolerant(margin):
    # 52 dB-Hz on L1CA (nominal 45.5): suspicious at tight margins, nominal at wide.
    a = cn0_plausible(52.0, "L1CA", margin_db=margin)
    expected_high = (52.0 - 45.5) > margin
    assert (a.flag is Cn0Flag.HIGH) == expected_high
