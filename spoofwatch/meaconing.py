"""Meaconing / replay (delayed-rebroadcast) detector (pure stdlib).

Meaconing is *record-and-replay*: an attacker captures genuine GNSS signals and
re-broadcasts them after a delay, dragging every victim in range to the recorded
location with a **consistent time offset**. It differs from a live signal
generator (which the co-location detector already catches) in two tells this
module looks for:

1. **Consistent-offset co-location** — many distinct aircraft/ships snap to one
   point *and* their clock/time offsets cluster tightly around a common value
   (the replay delay), rather than being scattered.
2. **Content replay** — the *same* navigation-message content (a hash/id the
   receiver can expose) reappears later at a fixed delay, i.e. yesterday's sky
   played back today.

This extends, and is complementary to, :mod:`spoofwatch.spoofing`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .records import haversine_km

COLOCATE_KM = 3.0
MIN_AIRCRAFT = 4
WINDOW_S = 120.0
# time-offset samples within this many seconds count as "a common offset"
OFFSET_CONSISTENCY_S = 5.0
# a replayed nav message at least this many seconds after the original
MIN_REPLAY_DELAY_S = 10.0


@dataclass
class NavObservation:
    id: str
    ts: float
    lat: float
    lon: float
    time_offset_s: float | None = None   # receiver clock offset vs truth (s)
    nav_msg: str | None = None           # opaque nav-message content id/hash


def _stdev(v):
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def detect_consistent_offset(observations, colocate_km=COLOCATE_KM,
                             min_aircraft=MIN_AIRCRAFT, window_s=WINDOW_S,
                             offset_consistency_s=OFFSET_CONSISTENCY_S):
    """Co-located clusters whose per-aircraft time offsets share a common value.

    Returns meaconing events; a tight offset spread (vs a scattered one) is the
    discriminator against a live spoofer, and drives the confidence.
    """
    events = []
    buckets = {}
    for o in observations:
        buckets.setdefault(int(o.ts // window_s), []).append(o)
    used = set()
    for _, obs in buckets.items():
        for anchor in obs:
            key = (round(anchor.lat, 3), round(anchor.lon, 3))
            if key in used:
                continue
            near = [o for o in obs
                    if haversine_km(anchor.lat, anchor.lon, o.lat, o.lon) <= colocate_km]
            ids = {o.id for o in near}
            offsets = [o.time_offset_s for o in near if o.time_offset_s is not None]
            if len(ids) >= min_aircraft and len(offsets) >= min_aircraft:
                spread = _stdev(offsets)
                if spread <= offset_consistency_s:
                    used.add(key)
                    clat = sum(o.lat for o in near) / len(near)
                    clon = sum(o.lon for o in near) / len(near)
                    common = sorted(offsets)[len(offsets) // 2]
                    events.append({
                        "type": "meaconing", "ids": ids,
                        "point": (round(clat, 5), round(clon, 5)),
                        "n": len(ids),
                        "common_offset_s": round(common, 3),
                        "offset_spread_s": round(spread, 3),
                        "confidence": round(min(1.0, (len(ids) / (min_aircraft * 2))
                                                * (1.0 - spread / (offset_consistency_s * 2))), 2),
                    })
    return events


def detect_content_replay(observations, min_delay_s=MIN_REPLAY_DELAY_S):
    """Identical nav-message content reappearing later — a record-and-replay tell.

    Groups observations by ``nav_msg`` id; any content seen at two timestamps
    separated by at least ``min_delay_s`` is a replay candidate.
    """
    by_msg = {}
    for o in observations:
        if o.nav_msg is not None:
            by_msg.setdefault(o.nav_msg, []).append(o)
    events = []
    for msg, obs in by_msg.items():
        times = sorted(o.ts for o in obs)
        if len(times) < 2:
            continue
        delay = times[-1] - times[0]
        if delay >= min_delay_s:
            events.append({
                "type": "content_replay", "nav_msg": msg,
                "ids": {o.id for o in obs},
                "first_ts": times[0], "last_ts": times[-1],
                "replay_delay_s": round(delay, 3),
                "occurrences": len(times),
                "confidence": round(min(1.0, 0.5 + 0.1 * len(times)), 2),
            })
    return events


def detect(observations, **kw):
    """Run both meaconing tells; return the combined event list."""
    co = detect_consistent_offset(observations,
                                  **{k: v for k, v in kw.items()
                                     if k in ("colocate_km", "min_aircraft",
                                              "window_s", "offset_consistency_s")})
    rp = detect_content_replay(observations,
                               **{k: v for k, v in kw.items() if k in ("min_delay_s",)})
    return co + rp
