"""Build a large harvest frontier from this repo's real source registry.

Crossing the curated sources with a global geographic tile grid turns the
registry into tens of thousands of concrete, addressable harvest endpoints for
:mod:`spoofwatch.growth`. Awareness/analysis only; no targeting content.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .growth import Endpoint, expand


def _tile_grid(step_deg: int = 5) -> Tuple[str, ...]:
    """A global lat/lon tile vocabulary at ``step_deg`` resolution.

    5 degrees -> 36 x 72 = 2,592 tiles; each source crossed with these yields
    thousands of endpoints.
    """
    if step_deg <= 0 or 180 % step_deg != 0:
        raise ValueError("step_deg must be a positive divisor of 180")
    tiles: List[str] = []
    lat = -90
    while lat < 90:
        lon = -180
        while lon < 180:
            tiles.append("{0}_{1}".format(lat, lon))
            lon += step_deg
        lat += step_deg
    return tuple(tiles)


GEO_TILES: Tuple[str, ...] = _tile_grid(5)


def source_pairs() -> List[Tuple[str, str]]:
    """(name, url) pairs from the repo's real dataset registry."""
    from .datasets import GNSS_DATASETS
    return [(d.name, d.url) for d in GNSS_DATASETS]


def build_frontier(vocabulary: Optional[Sequence[str]] = None) -> List[Endpoint]:
    """Cross the real sources with a geographic vocabulary into many endpoints."""
    vocab = tuple(vocabulary) if vocabulary is not None else GEO_TILES
    pairs = [(n, u) for (n, u) in source_pairs() if isinstance(u, str) and u.startswith("http")]
    return expand(pairs, vocab, param_name="tile", base_only=True)
