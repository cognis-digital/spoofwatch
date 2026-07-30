"""Top-level orchestration: run jamming + spoofing detection over a report set."""

from __future__ import annotations

from . import jamming, meaconing, osnma, spoofing


def analyze(reports, **kw):
    zones = jamming.detect(reports, **{k: v for k, v in kw.items()
                                       if k in ("cell_km", "integ_thresh", "min_cell", "min_frac")})
    spoofs = spoofing.detect(reports)
    return {
        "reports": len(reports),
        "jamming_zones": zones,
        "spoof_events": [{k: (sorted(v) if isinstance(v, set) else v) for k, v in s.items()}
                         for s in spoofs],
        "summary": {"jamming_zones": len(zones),
                    "spoof_events": len(spoofs),
                    "aircraft_spoofed": len({i for s in spoofs for i in s["ids"]})},
    }


def _jsonify(events):
    return [{k: (sorted(v) if isinstance(v, set) else v) for k, v in e.items()}
            for e in events]


def analyze_full(reports, osnma_reports=None, nav_observations=None, **kw):
    """Extended orchestration folding in the additive signals.

    A superset of :func:`analyze` that is fully backward-compatible: it still
    returns ``reports``/``jamming_zones``/``spoof_events``/``summary``, and
    additionally correlates optional OSNMA authentication verdicts (elevating
    corroborated spoofs and surfacing auth-only events) and runs the
    meaconing/replay detector when nav observations are supplied.
    """
    base = analyze(reports, **kw)
    spoofs = spoofing.detect(reports)

    osnma_events = []
    if osnma_reports:
        spoofs = osnma.correlate(spoofs, osnma_reports)
        osnma_events = osnma.standalone_auth_events(osnma_reports)

    meacon_events = []
    if nav_observations:
        meacon_events = meaconing.detect(nav_observations)

    base["spoof_events"] = _jsonify(spoofs)
    base["osnma_events"] = _jsonify(osnma_events)
    base["meaconing_events"] = _jsonify(meacon_events)
    base["summary"]["osnma_auth_fails"] = len(osnma_events)
    base["summary"]["meaconing_events"] = len(meacon_events)
    base["summary"]["aircraft_spoofed"] = len({i for s in spoofs for i in s["ids"]})
    return base
