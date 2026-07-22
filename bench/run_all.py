"""Spoofwatch benchmark — deterministic synthetic airspace, real metrics.

Regenerates RESULTS.md. Over N scenes with planted jamming zones + spoofing:
  - jamming zone detection rate + localization error (km)
  - spoofing aircraft recall + precision
  - spoof-origin localization error (km)

Run:  python bench/run_all.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoofwatch import detect as detectmod, synth as synthmod  # noqa: E402
from spoofwatch.records import haversine_km  # noqa: E402

SEEDS = list(range(1, 21))


def evaluate(seeds=None):
    seeds = seeds or SEEDS
    jam_found = 0; jam_errs = []
    recalls = []; precisions = []; origin_errs = []
    fp_zones = 0
    for s in seeds:
        reports, truth = synthmod.generate(seed=s)
        res = detectmod.analyze(reports)

        # --- jamming ---
        jz_lat, jz_lon, jz_r = truth["jamming_zones"][0]
        dists = [haversine_km(jz_lat, jz_lon, z["center_lat"], z["center_lon"])
                 for z in res["jamming_zones"]]
        near = [d for d in dists if d <= jz_r + 25]
        if near:
            jam_found += 1; jam_errs.append(min(near))
        fp_zones += sum(1 for d in dists if d > jz_r + 25)

        # --- spoofing (aircraft ids) ---
        truth_ids = truth["spoofed_ids"]
        det_ids = {i for e in res["spoof_events"] for i in e["ids"]}
        if truth_ids:
            recalls.append(len(det_ids & truth_ids) / len(truth_ids))
        if det_ids:
            precisions.append(len(det_ids & truth_ids) / len(det_ids))

        # --- spoof origin localization ---
        sp_lat, sp_lon = truth["spoof_points"][0]
        colo = [e for e in res["spoof_events"] if e["type"] == "colocation"]
        if colo:
            origin_errs.append(min(haversine_km(sp_lat, sp_lon, e["point"][0], e["point"][1])
                                   for e in colo))

    n = len(seeds)
    return {
        "scenes": n,
        "jamming_detect_rate": round(jam_found / n, 3),
        "jamming_loc_err_km": round(sum(jam_errs) / len(jam_errs), 1) if jam_errs else None,
        "jamming_false_zones_total": fp_zones,
        "spoof_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
        "spoof_precision": round(sum(precisions) / len(precisions), 3) if precisions else None,
        "spoof_origin_err_km": round(sum(origin_errs) / len(origin_errs), 1) if origin_errs else None,
    }


def write_results(res, path="RESULTS.md"):
    lines = [
        "# Spoofwatch — benchmark results", "",
        "Deterministic synthetic airspace with planted jamming zones + spoofing, scored "
        "against ground truth. Regenerate with `python bench/run_all.py`. Synthetic — measures "
        "algorithm correctness, not fielded accuracy on live ADS-B/AIS.", "",
        f"- Scenes: **{res['scenes']}**",
        f"- Jamming-zone detection rate: **{res['jamming_detect_rate']*100:.0f}%**",
        f"- Jamming-zone localization error (mean): **{res['jamming_loc_err_km']} km**",
        f"- False jamming zones (total across scenes): **{res['jamming_false_zones_total']}**",
        f"- Spoofed-aircraft recall: **{res['spoof_recall']}**",
        f"- Spoofed-aircraft precision: **{res['spoof_precision']}**",
        f"- Spoof-origin localization error (mean): **{res['spoof_origin_err_km']} km**",
        "",
    ]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    res = evaluate()
    write_results(res)
    print(json.dumps(res, indent=2))
