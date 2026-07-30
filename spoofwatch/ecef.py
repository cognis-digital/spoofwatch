"""WGS-84 geodetic <-> ECEF conversions + a satellite-geometry helper (stdlib).

Supports the RAIM solver and its synthetic tests: turn a lat/lon/alt into an
Earth-Centred-Earth-Fixed (ECEF) point, and place a "satellite" at a given
azimuth/elevation above a receiver so a realistic line-of-sight geometry can be
built without ephemeris data. Regional/analysis grade, not survey grade.
"""

from __future__ import annotations

import math

WGS84_A = 6378137.0                      # semi-major axis (m)
WGS84_F = 1.0 / 298.257223563            # flattening
WGS84_E2 = WGS84_F * (2 - WGS84_F)       # first eccentricity squared
GPS_RANGE_M = 20_200_000.0               # nominal GPS orbit slant range


def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sl * sl)
    x = (N + alt_m) * cl * math.cos(lon)
    y = (N + alt_m) * cl * math.sin(lon)
    z = (N * (1 - WGS84_E2) + alt_m) * sl
    return [x, y, z]


def ecef_to_geodetic(x, y, z):
    """Bowring / iterative inverse. Returns (lat_deg, lon_deg, alt_m)."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - WGS84_E2))
    for _ in range(8):
        sl = math.sin(lat)
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * sl * sl)
        alt = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1 - WGS84_E2 * N / (N + alt)))
    sl = math.sin(lat)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sl * sl)
    alt = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), alt


def enu_to_ecef_vector(e, n, u, lat_deg, lon_deg):
    """Rotate a local East/North/Up vector into ECEF axes at a reference point."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    x = -so * e - sl * co * n + cl * co * u
    y = co * e - sl * so * n + cl * so * u
    z = cl * n + sl * u
    return [x, y, z]


def sat_ecef_from_azel(lat_deg, lon_deg, az_deg, el_deg, rng_m=GPS_RANGE_M, alt_m=0.0):
    """ECEF position of a satellite seen from (lat,lon) at azimuth/elevation.

    Azimuth is degrees clockwise from North; elevation is degrees above the
    horizon. Places the satellite ``rng_m`` along that line of sight — enough to
    exercise the solver geometry with a plausible dilution-of-precision.
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    e = math.cos(el) * math.sin(az)
    n = math.cos(el) * math.cos(az)
    u = math.sin(el)
    los = enu_to_ecef_vector(e, n, u, lat_deg, lon_deg)
    rx = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return [rx[i] + rng_m * los[i] for i in range(3)]
