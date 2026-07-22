# Spoofwatch — benchmark results

Deterministic synthetic airspace with planted jamming zones + spoofing, scored against ground truth. Regenerate with `python bench/run_all.py`. Synthetic — measures algorithm correctness, not fielded accuracy on live ADS-B/AIS.

- Scenes: **20**
- Jamming-zone detection rate: **100%**
- Jamming-zone localization error (mean): **10.6 km**
- False jamming zones (total across scenes): **2**
- Spoofed-aircraft recall: **1.0**
- Spoofed-aircraft precision: **0.997**
- Spoof-origin localization error (mean): **0.3 km**

