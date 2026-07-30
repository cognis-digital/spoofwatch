"""OSNMA authentication-flag ingestion & correlation (pure stdlib).

Galileo's Open Service Navigation Message Authentication (OSNMA) went to
operational Initial Service in 2025 — the first GNSS to let a receiver
cryptographically verify that the navigation message really came from Galileo
and was not spoofed. spoofwatch does **not** implement the crypto; it *consumes
the receiver's verdict*. A downstream OSNMA-capable receiver reports, per epoch,
one of:

* ``VERIFIED``     — signature checked out (authentic),
* ``AUTH_FAILED``  — signature present but invalid (a strong spoof tell),
* ``NOT_VERIFIED`` — no OSNMA data available (unknown / legacy).

An ``AUTH_FAILED`` verdict that coincides with a geometric spoof signal
(teleport / co-location / RAIM fault) is near-conclusive, so we *elevate* those
events to high confidence. An ``AUTH_FAILED`` on its own is still surfaced as a
standalone authentication event.
"""

from __future__ import annotations

from dataclasses import dataclass

VERIFIED = "VERIFIED"
AUTH_FAILED = "AUTH_FAILED"
NOT_VERIFIED = "NOT_VERIFIED"
_VALID = {VERIFIED, AUTH_FAILED, NOT_VERIFIED}

# confidence floor an OSNMA-corroborated spoof event is lifted to
CORROBORATED_CONFIDENCE = 0.98


@dataclass
class OsnmaReport:
    id: str                 # aircraft/vessel/receiver id
    ts: float
    status: str             # one of VERIFIED / AUTH_FAILED / NOT_VERIFIED

    def __post_init__(self):
        s = str(self.status).upper().replace("-", "_")
        if s not in _VALID:
            raise ValueError(f"invalid OSNMA status: {self.status!r}")
        self.status = s


def auth_failed_ids(osnma_reports):
    """Set of ids that reported at least one AUTH_FAILED verdict."""
    return {r.id for r in osnma_reports if r.status == AUTH_FAILED}


def correlate(spoof_events, osnma_reports):
    """Elevate spoof events whose aircraft also reported an OSNMA auth failure.

    ``spoof_events`` is the list produced by :func:`spoofwatch.spoofing.detect`
    (each has ``ids`` and ``confidence``). Returns a new list — the input is not
    mutated — with an ``osnma`` block on any event that overlaps an auth-failed
    id, and its confidence raised to at least :data:`CORROBORATED_CONFIDENCE`.
    """
    failed = auth_failed_ids(osnma_reports)
    out = []
    for e in spoof_events:
        ev = dict(e)
        overlap = set(e.get("ids", set())) & failed
        if overlap:
            ev["osnma"] = {"corroborated": True, "auth_failed_ids": sorted(overlap)}
            ev["confidence"] = max(float(e.get("confidence", 0.0)),
                                   CORROBORATED_CONFIDENCE)
        out.append(ev)
    return out


def standalone_auth_events(osnma_reports):
    """Auth-failure events for ids not otherwise flagged (receiver-only tell).

    Groups AUTH_FAILED verdicts per id into a single event with the count and
    time span, so an OSNMA-only spoof (no position anomaly yet) is still raised.
    """
    by_id = {}
    for r in osnma_reports:
        if r.status == AUTH_FAILED:
            by_id.setdefault(r.id, []).append(r.ts)
    events = []
    for _id, times in sorted(by_id.items()):
        times.sort()
        events.append({"type": "osnma_auth_fail", "ids": {_id},
                       "n_failures": len(times),
                       "first_ts": times[0], "last_ts": times[-1],
                       "confidence": round(min(1.0, 0.6 + 0.1 * len(times)), 2)})
    return events
