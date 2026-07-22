import io
import json
from contextlib import redirect_stdout

from bench.run_all import evaluate
from spoofwatch import cli, detect, geo, synth


def test_grid_cell_and_center():
    g = geo.Grid(54.0, 20.0, cell_km=20.0)
    ix, iy = g.cell(54.2, 20.3)
    lat, lon = g.center(ix, iy)
    assert abs(lat - 54.2) < 0.3 and abs(lon - 20.3) < 0.3


def test_geojson_structure():
    reps, _ = synth.generate(seed=7)
    res = detect.analyze(reps)
    # rebuild zones/spoofs objects for geojson via the modules
    from spoofwatch import jamming, spoofing
    fc = geo.geojson(jamming.detect(reps), spoofing.detect(reps))
    assert fc["type"] == "FeatureCollection"
    kinds = {f["properties"]["kind"] for f in fc["features"]}
    assert any(k.startswith("spoof_") for k in kinds)
    assert "jamming_zone" in kinds
    for f in fc["features"]:
        lon, lat = f["geometry"]["coordinates"]
        assert -180 <= lon <= 180 and -90 <= lat <= 90


def test_bench_meets_thresholds():
    r = evaluate()
    assert r["jamming_detect_rate"] >= 0.9
    assert r["jamming_loc_err_km"] <= 25
    assert r["spoof_recall"] >= 0.8
    assert r["spoof_precision"] >= 0.8
    assert r["spoof_origin_err_km"] <= 10


def test_bench_deterministic():
    assert evaluate([1, 2, 3]) == evaluate([1, 2, 3])


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, buf.getvalue()


def test_cli_demo_and_selfcheck():
    rc, out = _run(["demo", "--seed", "7"])
    assert rc == 0 and "jamming" in out.lower()
    rc, out = _run(["selfcheck"])
    assert rc == 0 and "PASS" in out


def test_cli_synth_analyze_map(tmp_path):
    feed = str(tmp_path / "feed.json")
    rc, _ = _run(["synth", "--seed", "7", "--out", feed])
    assert rc == 0
    rc, out = _run(["analyze", feed])
    assert rc == 0
    data = json.loads(out)
    assert data["summary"]["jamming_zones"] >= 1
    rc, out = _run(["map", feed])
    assert rc == 0
    assert json.loads(out)["type"] == "FeatureCollection"
