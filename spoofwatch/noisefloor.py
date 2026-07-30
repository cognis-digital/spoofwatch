"""AGC / noise-floor jamming monitor with running baseline + J/N estimate (stdlib).

Where :mod:`spoofwatch.signal` differences C/N0 and AGC frame-to-frame, this
module keeps a **running baseline** of the receiver's quiet-time noise level and
watches for the sustained excursion a jammer produces. A GNSS front end fights
rising in-band interference by dropping its Automatic Gain Control; equivalently,
the effective noise floor climbs. Either series feeds the same detector — an
elevated level relative to the learned baseline is the **jamming-to-noise ratio**
(J/N), the number that actually characterises a jammer's bite.

To avoid chattering on every noise wobble the detector uses **hysteresis**: a
jamming episode *opens* when J/N crosses an ``on`` threshold and only *closes*
when it falls back under a lower ``off`` threshold. Each episode is reported with
its onset/offset timestamps, duration, and peak/mean J/N — an event log an
operator can act on. The baseline adapts by exponential moving average over
samples judged quiet, so slow environmental drift is tracked while an attack is
not learned away.

A sample carries either a noise-floor reading (dB; higher = more interference) or
an AGC reading (dB; higher = more gain, so a *drop* means interference). The
detector normalises both to one effective interference level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# J/N (dB) at which a jamming episode opens, and the lower level at which it closes
JN_ON_DB = 6.0
JN_OFF_DB = 3.0
# EMA smoothing for the quiet-time baseline (0..1; smaller = slower adaptation)
BASELINE_ALPHA = 0.1
# samples used to seed the baseline before detection begins
LEARN_SAMPLES = 5


@dataclass
class NoiseSample:
    ts: float
    noise_dbm: float | None = None     # measured noise floor (dB / dBm); higher = worse
    agc_db: float | None = None        # AGC gain (dB); higher = more gain

    def level(self):
        """Effective interference level (higher = more interference).

        Uses the noise floor if present, else inverts AGC gain so a gain drop
        reads as a level rise. Raises if neither field is set.
        """
        if self.noise_dbm is not None:
            return self.noise_dbm
        if self.agc_db is not None:
            return -self.agc_db
        raise ValueError("NoiseSample needs noise_dbm or agc_db")


@dataclass
class JamEvent:
    start_ts: float
    end_ts: float | None            # None while still ongoing at series end
    peak_jn_db: float
    mean_jn_db: float
    duration_s: float
    n_samples: int
    ongoing: bool = False


@dataclass
class NoiseState:
    ts: float
    level: float
    baseline: float
    jn_db: float
    jamming: bool


def jn_ratio(level, baseline):
    """Jamming-to-noise ratio (dB) of a level above its baseline."""
    return level - baseline


def monitor(samples, on_db=JN_ON_DB, off_db=JN_OFF_DB, alpha=BASELINE_ALPHA,
            learn=LEARN_SAMPLES):
    """Run the hysteretic jamming monitor over a chronological sample series.

    Returns a dict with per-sample ``states`` (:class:`NoiseState`) and the
    detected jamming ``events`` (:class:`JamEvent`). The baseline is seeded from
    the mean of the first ``learn`` samples and thereafter EMA-adapts on quiet
    samples only, so an attack is never learned into the baseline.
    """
    samples = sorted(samples, key=lambda s: s.ts)
    if not samples:
        return {"states": [], "events": [], "baseline_dbm": None,
                "jamming_now": False}

    levels = [s.level() for s in samples]
    seed = levels[:max(1, min(learn, len(levels)))]
    baseline = sum(seed) / len(seed)

    states = []
    events = []
    in_event = False
    cur = None
    for s, lvl in zip(samples, levels):
        jn = jn_ratio(lvl, baseline)
        if not in_event:
            if jn >= on_db:
                in_event = True
                cur = {"start_ts": s.ts, "peak": jn, "sum": jn, "n": 1}
            else:
                # quiet -> adapt baseline toward this level
                baseline = (1 - alpha) * baseline + alpha * lvl
        else:
            cur["peak"] = max(cur["peak"], jn)
            cur["sum"] += jn
            cur["n"] += 1
            if jn < off_db:
                in_event = False
                events.append(JamEvent(
                    start_ts=cur["start_ts"], end_ts=s.ts,
                    peak_jn_db=round(cur["peak"], 3),
                    mean_jn_db=round(cur["sum"] / cur["n"], 3),
                    duration_s=round(s.ts - cur["start_ts"], 3),
                    n_samples=cur["n"], ongoing=False))
                cur = None
        states.append(NoiseState(s.ts, round(lvl, 3), round(baseline, 3),
                                 round(jn, 3), in_event))

    if in_event and cur is not None:
        last = samples[-1]
        events.append(JamEvent(
            start_ts=cur["start_ts"], end_ts=None,
            peak_jn_db=round(cur["peak"], 3),
            mean_jn_db=round(cur["sum"] / cur["n"], 3),
            duration_s=round(last.ts - cur["start_ts"], 3),
            n_samples=cur["n"], ongoing=True))

    return {"states": states, "events": events,
            "baseline_dbm": round(baseline, 3),
            "jamming_now": in_event}


def summarize(samples, **kw):
    """Roll the monitor output into headline counts + a 0..1 jamming confidence."""
    res = monitor(samples, **kw)
    events = res["events"]
    peak = max((e.peak_jn_db for e in events), default=0.0)
    total = sum(e.duration_s for e in events)
    on_db = kw.get("on_db", JN_ON_DB)
    # confidence scales with how far the worst J/N exceeds the onset threshold
    conf = round(min(1.0, max(0.0, (peak - on_db) / (on_db * 2))), 3) if events else 0.0
    return {
        "samples": len(res["states"]),
        "jam_events": len(events),
        "peak_jn_db": round(peak, 3),
        "total_jam_duration_s": round(total, 3),
        "jamming_now": res["jamming_now"],
        "baseline_dbm": res["baseline_dbm"],
        "confidence": conf,
    }
