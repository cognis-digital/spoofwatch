"""Per-satellite C/N0 vs elevation-angle consistency monitor (pure stdlib).

:mod:`spoofwatch.signal` watches the *aggregate* carrier-to-noise-density across
all channels epoch to epoch. This module goes finer: it exploits the physics of
an honest sky. A genuine constellation's C/N0 **rises with elevation** — a
satellite near the horizon shines through more atmosphere and off the antenna's
low-gain skirt, so it is weaker; one overhead is strong. Plot C/N0 against
elevation and honest signals slope upward with natural scatter.

A single-antenna spoofer forging every "satellite" from one transmitter cannot
reproduce that: every counterfeit arrives at essentially the *same* power
regardless of the elevation it claims, so the C/N0-vs-elevation relationship goes
**flat** and the spread across channels collapses — often at a uniformly *high*
level as the spoofer overpowers the real constellation. This monitor fits the
C/N0-vs-elevation line, flags that flat/uniform signature, and separately picks
out individual PRNs whose power sits far above what their claimed elevation
predicts (a partial or meaconing overpower). A companion :func:`flatline` test
catches PRNs whose C/N0 is frozen unnaturally still across time.

Purely relative/statistical — no per-receiver calibration required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# genuine skies rise faster than this many dB-Hz per degree of elevation
FLAT_SLOPE_DBHZ_PER_DEG = 0.10
# a spoofer's all-equal-power channels sit within this C/N0 spread (dB-Hz)
UNIFORM_SPREAD_DBHZ = 3.0
# "overpowered" reference level: uniform high C/N0 raises suspicion further
HIGH_CN0_DBHZ = 45.0
# a PRN this far above its elevation-predicted C/N0 is an overpower outlier
OUTLIER_DBHZ = 8.0
# fewest satellites to fit a meaningful template
MIN_SATS = 4
# a per-PRN C/N0 time series flatter than this stdev (dB-Hz) is "frozen"
FROZEN_STDEV_DBHZ = 0.30


@dataclass
class SatObs:
    sat_id: str
    cn0: float                      # carrier-to-noise density (dB-Hz)
    el_deg: float                   # claimed elevation above horizon (deg)
    constellation: str = "GPS"


@dataclass
class Cn0Verdict:
    suspect: bool
    confidence: float               # 0..1
    slope_dbhz_per_deg: float       # C/N0 vs elevation slope
    intercept_dbhz: float
    spread_dbhz: float              # cross-channel C/N0 stdev
    mean_cn0_dbhz: float
    n_sats: int
    flat: bool                      # elevation slope collapsed
    uniform: bool                   # cross-channel spread collapsed
    outliers: list = field(default_factory=list)   # overpowered PRNs
    reason: str = ""


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _stdev(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def linfit(xs, ys):
    """Ordinary least-squares fit ``y = slope*x + intercept``.

    Returns ``(slope, intercept)``. A zero-variance ``x`` (all equal elevation)
    yields a zero slope and the mean as intercept.
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


def predicted_cn0(slope, intercept, el_deg):
    """C/N0 the fitted elevation template predicts for a given elevation."""
    return slope * el_deg + intercept


def outliers(obs, thresh_dbhz=OUTLIER_DBHZ):
    """PRNs whose C/N0 sits ``thresh_dbhz`` above the elevation-template prediction.

    These are the classic partial-overpower / meaconing tell: one or a few
    channels far brighter than their claimed geometry warrants. Returns a list of
    dicts, brightest excess first.
    """
    obs = list(obs)
    if len(obs) < MIN_SATS:
        return []
    slope, intercept = linfit([o.el_deg for o in obs], [o.cn0 for o in obs])
    out = []
    for o in obs:
        excess = o.cn0 - predicted_cn0(slope, intercept, o.el_deg)
        if excess >= thresh_dbhz:
            out.append({"sat_id": o.sat_id, "constellation": o.constellation,
                        "cn0_dbhz": round(o.cn0, 2), "el_deg": round(o.el_deg, 2),
                        "excess_dbhz": round(excess, 2)})
    out.sort(key=lambda d: -d["excess_dbhz"])
    return out


def check(obs, flat_slope=FLAT_SLOPE_DBHZ_PER_DEG,
          uniform_spread=UNIFORM_SPREAD_DBHZ, high_cn0=HIGH_CN0_DBHZ,
          outlier_dbhz=OUTLIER_DBHZ, min_sats=MIN_SATS):
    """Score a single epoch's per-satellite C/N0/elevation set for spoofing.

    Returns a :class:`Cn0Verdict`. The spoof signature is a **flat** C/N0-vs-
    elevation slope combined with a **uniform** (collapsed) cross-channel spread;
    a uniformly *high* level pushes confidence higher still. Overpower
    ``outliers`` are reported regardless.
    """
    obs = list(obs)
    n = len(obs)
    if n < min_sats:
        return Cn0Verdict(False, 0.0, 0.0, 0.0, 0.0, round(_mean([o.cn0 for o in obs]), 2),
                          n, False, False, [], reason=f"need >= {min_sats} satellites")

    cn0s = [o.cn0 for o in obs]
    els = [o.el_deg for o in obs]
    slope, intercept = linfit(els, cn0s)
    spread = _stdev(cn0s)
    mean_cn0 = _mean(cn0s)

    flat = slope < flat_slope
    uniform = spread < uniform_spread
    ol = outliers(obs, thresh_dbhz=outlier_dbhz)

    suspect = flat and uniform
    conf = 0.0
    reason = ""
    if suspect:
        # flatter slope + tighter spread + higher power => stronger evidence
        slope_term = min(1.0, max(0.0, (flat_slope - slope) / max(flat_slope, 1e-6)))
        spread_term = min(1.0, max(0.0, (uniform_spread - spread) / uniform_spread))
        power_term = min(1.0, max(0.0, (mean_cn0 - high_cn0) / high_cn0 + 0.5))
        conf = min(1.0, 0.5 * slope_term + 0.3 * spread_term + 0.2 * power_term)
        reason = "flat, uniform C/N0-vs-elevation (single-source overpower)"
    elif ol:
        reason = "overpowered PRN outlier(s)"

    return Cn0Verdict(
        suspect=bool(suspect),
        confidence=round(conf, 3),
        slope_dbhz_per_deg=round(slope, 4),
        intercept_dbhz=round(intercept, 2),
        spread_dbhz=round(spread, 2),
        mean_cn0_dbhz=round(mean_cn0, 2),
        n_sats=n,
        flat=bool(flat),
        uniform=bool(uniform),
        outliers=ol,
        reason=reason,
    )


def flatline(series, frozen_stdev=FROZEN_STDEV_DBHZ, min_len=4):
    """Find PRNs whose C/N0 is frozen unnaturally still across time.

    ``series`` maps ``sat_id -> [cn0, cn0, ...]`` (chronological). Genuine C/N0
    always dithers with scintillation and antenna motion; a replayed/synthesised
    channel can sit implausibly constant. Returns a dict with the flagged PRNs and
    a 0..1 confidence scaled by how many channels are frozen.
    """
    frozen = []
    checked = 0
    for sat_id, vals in series.items():
        vals = list(vals)
        if len(vals) < min_len:
            continue
        checked += 1
        sd = _stdev(vals)
        if sd <= frozen_stdev:
            frozen.append({"sat_id": sat_id, "stdev_dbhz": round(sd, 4),
                           "mean_dbhz": round(_mean(vals), 2), "samples": len(vals)})
    frozen.sort(key=lambda d: d["stdev_dbhz"])
    conf = round(min(1.0, len(frozen) / max(1, checked)), 3) if frozen else 0.0
    return {"frozen": frozen, "n_frozen": len(frozen), "n_checked": checked,
            "suspect": bool(frozen), "confidence": conf}
