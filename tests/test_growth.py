"""Tests for the dynamic dataset-growth engine and harvest frontier."""
import pytest

from spoofwatch.frontier import GEO_TILES, build_frontier, source_pairs
from spoofwatch.growth import (
    Endpoint,
    GrowthEngine,
    HarvestStore,
    Record,
    SourceStats,
    expand,
    synthetic_fetcher,
)


def test_expand_base_plus_terms():
    eps = expand([("S", "http://x")], ["a", "b", "c"], param_name="q")
    keys = {e.key for e in eps}
    assert "S:_all" in keys and "S:q=a" in keys
    assert len(eps) == 4


def test_store_merge_dedup():
    store = HarvestStore()
    r1 = Record("u1", "S", "e", 0.0)
    assert store.merge([r1, Record("u2", "S", "e", 0.0)]) == 2
    assert store.merge([r1]) == 0
    assert store.size == 2


def test_reliability_penalizes_errors():
    good = SourceStats(fetches=10, records=100, new_records=100, errors=0)
    bad = SourceStats(fetches=10, records=100, new_records=100, errors=8)
    assert good.reliability > bad.reliability


def _engine():
    eng = GrowthEngine(fetcher=synthetic_fetcher(3))
    eng.add_endpoints(expand([("S", "http://x")], [str(i) for i in range(10)]))
    return eng


def test_engine_grows_over_cycles():
    reports = _engine().grow(cycles=3, start=0.0, dt=1.0)
    sizes = [r.total_size for r in reports]
    assert sizes[0] < sizes[1] < sizes[2]


def test_engine_dedup_same_timestamp():
    eng = _engine()
    a = eng.run_cycle(now=5.0)
    b = eng.run_cycle(now=5.0)
    assert a.new_records > 0 and b.new_records == 0


def test_engine_survives_fetcher_errors():
    def boom(endpoint, now):
        raise RuntimeError("down")
    eng = GrowthEngine(fetcher=boom)
    eng.add_endpoints([Endpoint("S", "s:0", "http://x")])
    assert eng.run_cycle(now=0.0).new_records == 0
    assert eng.stats["S"].errors == 1


def test_frontier_discovery_compounds():
    def discover(records):
        return [Endpoint("D", "disc:" + records[0].uid, "http://d")]
    eng = GrowthEngine(fetcher=synthetic_fetcher(2), discoverer=discover)
    eng.add_endpoints([Endpoint("S", "s:0", "http://x")])
    before = eng.frontier_size
    eng.grow(cycles=2, start=0.0)
    assert eng.frontier_size > before


def test_tile_grid_size():
    assert len(GEO_TILES) == 2592


def test_frontier_reaches_thousands():
    endpoints = build_frontier()
    assert len(endpoints) >= 2000
    assert len({e.key for e in endpoints}) == len(endpoints)


def test_frontier_feeds_engine():
    endpoints = build_frontier()
    eng = GrowthEngine(fetcher=synthetic_fetcher(1))
    eng.add_endpoints(endpoints)
    rep = eng.run_cycle(now=0.0, max_endpoints=50)
    assert rep.new_records == 50


def test_source_pairs_have_urls():
    pairs = source_pairs()
    assert pairs
    assert all(u.startswith("http") for (n, u) in pairs)
