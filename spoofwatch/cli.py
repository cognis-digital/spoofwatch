"""Spoofwatch CLI."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, detect as detectmod, geo as geomod, jamming, records, spoofing
from . import synth as synthmod


def _load(path):
    return records.load_csv(path) if path.lower().endswith(".csv") else records.load_json(path)


def cmd_analyze(args):
    reps = _load(args.feed)
    print(json.dumps(detectmod.analyze(reps, cell_km=args.cell_km), indent=2))
    return 0


def cmd_map(args):
    reps = _load(args.feed)
    zones = jamming.detect(reps, cell_km=args.cell_km)
    spoofs = spoofing.detect(reps)
    fc = geomod.geojson(zones, spoofs)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(fc, f, indent=2)
        print(f"[+] {len(fc['features'])} features -> {args.out}")
    else:
        print(json.dumps(fc, indent=2))
    return 0


def cmd_synth(args):
    reps, truth = synthmod.generate(seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([r.as_dict() | {"speed_kt": r.speed_kt} for r in reps], f)
    print(f"[+] {len(reps)} reports -> {args.out}  "
          f"(jamming_zones={len(truth['jamming_zones'])} "
          f"spoofed_aircraft={len(truth['spoofed_ids'])})")
    return 0


def cmd_demo(args):
    reps, truth = synthmod.generate(seed=args.seed)
    res = detectmod.analyze(reps)
    print(f"Spoofwatch demo (seed {args.seed}) — {len(reps)} reports")
    jz = truth["jamming_zones"][0]
    print(f"  planted jamming zone : ({jz[0]:.2f},{jz[1]:.2f}) r={jz[2]:.0f}km")
    for z in res["jamming_zones"]:
        print(f"  DETECTED jamming     : ({z['center_lat']:.2f},{z['center_lon']:.2f}) "
              f"r={z['radius_km']}km  degraded={z['degraded_reports']}  conf={z['confidence']}")
    print(f"  planted spoof point  : ({truth['spoof_points'][0][0]:.2f},{truth['spoof_points'][0][1]:.2f})"
          f"  spoofed aircraft={len(truth['spoofed_ids'])}")
    for e in res["spoof_events"]:
        print(f"  DETECTED spoof       : {e['type']:11} @ ({e['point'][0]:.2f},{e['point'][1]:.2f})  "
              f"aircraft={e['n']}  conf={e['confidence']}")
    return 0


def cmd_selfcheck(args):
    from bench.run_all import evaluate
    r = evaluate()
    ok = (r["jamming_detect_rate"] >= 0.9 and (r["jamming_loc_err_km"] or 1e9) <= 25
          and (r["spoof_recall"] or 0) >= 0.8 and (r["spoof_precision"] or 0) >= 0.8
          and (r["spoof_origin_err_km"] or 1e9) <= 10)
    print(json.dumps(r, indent=2))
    print("SELFCHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def build_parser():
    p = argparse.ArgumentParser(prog="spoofwatch",
                                description="Detect & map GNSS jamming/spoofing from position feeds — Cognis Digital")
    p.add_argument("--version", action="version", version=f"spoofwatch {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="detect jamming + spoofing in a feed (JSON/CSV)")
    a.add_argument("feed"); a.add_argument("--cell-km", type=float, default=20.0)
    a.set_defaults(func=cmd_analyze)

    m = sub.add_parser("map", help="emit GeoJSON of jamming zones + spoof origins")
    m.add_argument("feed"); m.add_argument("--cell-km", type=float, default=20.0)
    m.add_argument("--out"); m.set_defaults(func=cmd_map)

    s = sub.add_parser("synth", help="write a synthetic feed with planted interference")
    s.add_argument("--seed", type=int, default=7); s.add_argument("--out", default="feed.json")
    s.set_defaults(func=cmd_synth)

    dm = sub.add_parser("demo", help="generate + analyze a synthetic scene, print a summary")
    dm.add_argument("--seed", type=int, default=7); dm.set_defaults(func=cmd_demo)

    sc = sub.add_parser("selfcheck", help="run the benchmark and assert metrics pass")
    sc.set_defaults(func=cmd_selfcheck)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
