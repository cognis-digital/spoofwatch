"""Top-level orchestration: run jamming + spoofing detection over a report set."""

from __future__ import annotations

from . import jamming, spoofing


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
