"""End-to-end example: synth a contested-airspace scene → detect → map."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoofwatch import detect, geo, jamming, spoofing, synth  # noqa: E402


def main():
    reports, truth = synth.generate(seed=7)
    print(f"{len(reports)} reports; planted jamming zone={truth['jamming_zones'][0]} "
          f"spoof point={truth['spoof_points'][0]} spoofed aircraft={len(truth['spoofed_ids'])}")

    res = detect.analyze(reports)
    print("detected:", json.dumps(res["summary"]))
    for z in res["jamming_zones"]:
        print(f"  jamming zone: ({z['center_lat']},{z['center_lon']}) r={z['radius_km']}km "
              f"conf={z['confidence']}")
    for e in res["spoof_events"][:3]:
        print(f"  spoof {e['type']}: @{e['point']} aircraft={e['n']} conf={e['confidence']}")

    fc = geo.geojson(jamming.detect(reports), spoofing.detect(reports))
    print(f"  geojson features: {len(fc['features'])}")


if __name__ == "__main__":
    main()
