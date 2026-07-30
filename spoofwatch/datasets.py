"""GNSS spoofing/jamming dataset registry + nominal-signal-power reference gate.

Two things a fielded GNSS-integrity monitor needs that this module supplies from
open, citable sources:

1. A registry of the public spoofing / jamming / interference corpora a real
   deployment would validate against (TEXBAT, OAKBAT, FGI-JSDR, Jammertest,
   Fraunhofer highway captures, GPSJam event maps, ...).

2. A nominal received-power / carrier-to-noise reference per GNSS signal (from the
   published SPS / IS-GPS-200 minimums) and a *plausibility gate*: a measured C/N0
   sitting well ABOVE the nominal ceiling is a classic overpowered-spoofer tell,
   while one well below suggests jamming or obstruction. This turns a raw C/N0 into
   an interpretable integrity flag — no black box, just documented signal physics.

Detection / integrity only — nothing here generates or aids spoofing. Pure stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class DatasetKind(str, Enum):
    SPOOFING_IQ = "spoofing_iq"
    JAMMING_IQ = "jamming_iq"
    INTERFERENCE_EVENT = "interference_event"
    IONO_REFERENCE = "iono_reference"
    TAXONOMY = "taxonomy"


@dataclass(frozen=True)
class GnssDataset:
    name: str
    url: str
    kind: DatasetKind
    bands: str
    size: str
    license: str            # open | research | non-commercial | proprietary
    note: str

    @property
    def redistributable(self) -> bool:
        return self.license == "open"

    def to_dict(self) -> dict:
        return {"name": self.name, "url": self.url, "kind": self.kind.value,
                "bands": self.bands, "size": self.size, "license": self.license,
                "note": self.note}


# Real, citable GNSS interference corpora (July 2026 survey).
GNSS_DATASETS: List[GnssDataset] = [
    GnssDataset("TEXBAT", "https://radionavlab.ae.utexas.edu/texbat/",
                DatasetKind.SPOOFING_IQ, "GPS L1 C/A",
                "8 scenarios, 25 Msps 16-bit I/Q (tens of GB each)", "research",
                "Texas Spoofing Test Battery — the canonical civil-GPS spoofing corpus "
                "(clean + ds1-ds8)"),
    GnssDataset("OAKBAT", "https://github.com/oakbat/",
                DatasetKind.SPOOFING_IQ, "GPS L1 C/A + Galileo E1",
                "matched to TEXBAT ds1-ds6", "open",
                "Oak Ridge Spoofing & Interference Test Battery — fully reproducible "
                "TEXBAT analog with full config metadata (US DOE)"),
    GnssDataset("FGI-SpoofRepo", "https://etsin.fairdata.fi/dataset/367379a8-7d78-4b08-91f0-8027ce7a621b",
                DatasetKind.SPOOFING_IQ, "GPS L1/L5 + Galileo E1/E5a",
                "3 scenario classes", "research",
                "Finnish Geospatial Research Institute multi-band synchronous / "
                "asynchronous / meaconing I/Q"),
    GnssDataset("FGI-OSNMA", "https://etsin.fairdata.fi/dataset/09dc5c1b-933d-4efd-aa66-be2c07fab3b3",
                DatasetKind.SPOOFING_IQ, "GPS L1 C/A + Galileo E1",
                "clean vs spoofed", "research",
                "raw I/Q for Galileo OSNMA authentication testing (Jammertest 2023)"),
    GnssDataset("FGI-Jammertest-2023", "https://etsin.fairdata.fi/dataset/2eee8cfd-db00-4e46-bd62-d7632c09c700",
                DatasetKind.JAMMING_IQ, "GPS L1 C/A + Galileo E1",
                "1 jamming + 2 spoofing scenarios", "research",
                "Andoya Jammertest 2023 high-power jamming + coherent/incoherent spoofing"),
    GnssDataset("Jammertest-2024", "https://zenodo.org/records/15911589",
                DatasetKind.JAMMING_IQ, "multi-band",
                "375 MB compressed / 167.8 GB total", "open",
                "controlled jamming/spoofing/meaconing, stationary + dynamic; raw UBX + "
                "RINEX + RF monitoring (GPL-3.0)"),
    GnssDataset("Fraunhofer-Highway-ds1", "https://dx.doi.org/10.21227/xpm9-tt28",
                DatasetKind.JAMMING_IQ, "Galileo E1 + E6",
                "~175 GB, 62.5 Msps 8-bit, 11 classes", "non-commercial",
                "wideband in-the-wild interference snapshots over a German highway "
                "(CC BY-NC-SA)"),
    GnssDataset("Zenodo-Jammer-IQ-Classes", "https://zenodo.org/records/4629685",
                DatasetKind.JAMMING_IQ, "GPS band",
                "6 classes, 1000 train / 250 test", "open",
                "labeled jammer waveforms (DME/narrowband/AM/chirp/FM/none) with MATLAB "
                "generator (CC BY 4.0)"),
    GnssDataset("GPSJam", "https://gpsjam.org/",
                DatasetKind.INTERFERENCE_EVENT, "L1 (ADS-B-derived)",
                "daily H3-hex GeoJSON, 2022-present", "open",
                "daily global interference map: hex flagged when >=10% of aircraft "
                "report low nav integrity (CC-BY)"),
    GnssDataset("Flightradar24-GPS-Jamming", "https://www.flightradar24.com/data/gps-jamming",
                DatasetKind.INTERFERENCE_EVENT, "L1 (ADS-B-derived)",
                "live regional aggregation", "proprietary",
                "ADS-B-derived live map of aircraft-reported GPS interference by region"),
    GnssDataset("NASA-CDDIS-TEC", "https://cddis.nasa.gov/",
                DatasetKind.IONO_REFERENCE, "multi-constellation",
                "continuous global network", "open",
                "global TEC maps + raw RINEX to model natural ionospheric degradation "
                "vs attack (NASA open data)"),
    GnssDataset("Madrigal-Ionosphere", "http://cedar.openmadrigal.org/",
                DatasetKind.IONO_REFERENCE, "multi-instrument",
                "large archive", "open",
                "global TEC / S4 / phase-scintillation to distinguish natural fades from "
                "attack (periodic multipath vs aperiodic scintillation)"),
    GnssDataset("UT-RNL-Spoofing-Taxonomy", "https://radionavlab.ae.utexas.edu/",
                DatasetKind.TAXONOMY, "n/a", "reference", "research",
                "canonical attack taxonomy (meaconing / selective-delay / non-coherent / "
                "coherent matched-power) + detection-metric definitions"),
]


def datasets_by_kind(kind: DatasetKind) -> List[GnssDataset]:
    return [d for d in GNSS_DATASETS if d.kind == kind]


def open_datasets() -> List[GnssDataset]:
    """Datasets whose license permits redistribution."""
    return [d for d in GNSS_DATASETS if d.redistributable]


# ---------------------------------------------------------------------------
# Nominal received-power / carrier-to-noise reference (from published SPS specs).
# Values are the customary minimum received power (dBW) and the corresponding
# open-sky nominal C/N0 (dB-Hz) for a typical patch antenna + receiver.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalPowerRef:
    signal: str
    min_power_dbw: float       # spec minimum received power
    nominal_cn0_dbhz: float     # typical open-sky C/N0 for a good receiver

    def to_dict(self) -> dict:
        return {"signal": self.signal, "min_power_dbw": self.min_power_dbw,
                "nominal_cn0_dbhz": self.nominal_cn0_dbhz}


# IS-GPS-200 / SPS reference points.
SIGNAL_POWER_REF: Dict[str, SignalPowerRef] = {
    "L1CA": SignalPowerRef("L1CA", -158.5, 45.5),
    "L2C": SignalPowerRef("L2C", -160.0, 43.0),
    "L5": SignalPowerRef("L5", -157.9, 46.0),
    "E1": SignalPowerRef("E1", -157.0, 46.0),
    "E5A": SignalPowerRef("E5A", -155.0, 48.0),
}


class Cn0Flag(str, Enum):
    NOMINAL = "nominal"
    LOW = "low"                 # below nominal band -> jamming / obstruction
    HIGH = "high"               # above nominal band -> possible overpowered spoofer
    UNKNOWN = "unknown"         # unrecognized signal


@dataclass(frozen=True)
class Cn0Assessment:
    signal: str
    measured_cn0_dbhz: float
    nominal_cn0_dbhz: Optional[float]
    flag: Cn0Flag
    deviation_db: Optional[float]
    reasoning: str

    @property
    def suspicious(self) -> bool:
        return self.flag in (Cn0Flag.LOW, Cn0Flag.HIGH)

    def to_dict(self) -> dict:
        dev = round(self.deviation_db, 2) if self.deviation_db is not None else None
        return {"signal": self.signal, "measured_cn0_dbhz": self.measured_cn0_dbhz,
                "nominal_cn0_dbhz": self.nominal_cn0_dbhz, "flag": self.flag.value,
                "deviation_db": dev, "suspicious": self.suspicious,
                "reasoning": self.reasoning}


def cn0_plausible(measured_cn0_dbhz: float, signal: str = "L1CA",
                  margin_db: float = 6.0) -> Cn0Assessment:
    """Flag a measured C/N0 against the published nominal for its signal.

    A receiver's open-sky C/N0 clusters near the nominal value. A reading a full
    ``margin_db`` ABOVE nominal is implausible from a satellite at spec power and is
    a hallmark of an overpowered spoofer trying to capture the tracking loops;
    a reading well BELOW nominal indicates jamming, obstruction, or deep fade.
    """
    if margin_db < 0:
        raise ValueError("margin_db must be >= 0")
    ref = SIGNAL_POWER_REF.get(signal.upper())
    if ref is None:
        return Cn0Assessment(signal, measured_cn0_dbhz, None, Cn0Flag.UNKNOWN, None,
                             f"no reference for signal '{signal}'")
    dev = measured_cn0_dbhz - ref.nominal_cn0_dbhz
    if dev > margin_db:
        flag = Cn0Flag.HIGH
        why = (f"C/N0 {measured_cn0_dbhz:.1f} dB-Hz is {dev:.1f} dB above nominal "
               f"{ref.nominal_cn0_dbhz:.1f} — implausibly strong; possible spoofer")
    elif dev < -margin_db:
        flag = Cn0Flag.LOW
        why = (f"C/N0 {measured_cn0_dbhz:.1f} dB-Hz is {-dev:.1f} dB below nominal "
               f"{ref.nominal_cn0_dbhz:.1f} — degraded; possible jamming/obstruction")
    else:
        flag = Cn0Flag.NOMINAL
        why = (f"C/N0 {measured_cn0_dbhz:.1f} dB-Hz within +/-{margin_db:.0f} dB of "
               f"nominal {ref.nominal_cn0_dbhz:.1f}")
    return Cn0Assessment(signal, measured_cn0_dbhz, ref.nominal_cn0_dbhz, flag,
                         round(dev, 3), why)
