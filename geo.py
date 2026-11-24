"""NYC TLC taxi zones used by the system.

IDs and names follow the public TLC zone lookup table
(https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv). Centroids are
approximate and only used for travel-time estimates; `scripts/download_data.py`
can replace them with centroids computed from the official shapefile.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    id: int
    name: str
    borough: str
    lat: float
    lon: float
    venue: str | None = None  # demand generator (arena, airport, theatre district...)


ZONES: list[Zone] = [
    Zone(132, "JFK Airport", "Queens", 40.6413, -73.7781, "airport"),
    Zone(138, "LaGuardia Airport", "Queens", 40.7769, -73.8740, "airport"),
    Zone(161, "Midtown Center", "Manhattan", 40.7580, -73.9780),
    Zone(162, "Midtown East", "Manhattan", 40.7560, -73.9700),
    Zone(163, "Midtown North", "Manhattan", 40.7640, -73.9790),
    Zone(164, "Midtown South", "Manhattan", 40.7480, -73.9850),
    Zone(186, "Penn Station/Madison Sq West", "Manhattan", 40.7505, -73.9934, "arena"),
    Zone(230, "Times Sq/Theatre District", "Manhattan", 40.7590, -73.9845, "theatre"),
    Zone(48, "Clinton East", "Manhattan", 40.7620, -73.9900),
    Zone(100, "Garment District", "Manhattan", 40.7540, -73.9900),
    Zone(170, "Murray Hill", "Manhattan", 40.7480, -73.9780),
    Zone(234, "Union Sq", "Manhattan", 40.7359, -73.9911),
    Zone(79, "East Village", "Manhattan", 40.7265, -73.9815, "nightlife"),
    Zone(249, "West Village", "Manhattan", 40.7340, -74.0040, "nightlife"),
    Zone(68, "East Chelsea", "Manhattan", 40.7440, -73.9970),
    Zone(246, "West Chelsea/Hudson Yards", "Manhattan", 40.7530, -74.0040),
    Zone(107, "Gramercy", "Manhattan", 40.7370, -73.9840),
    Zone(237, "Upper East Side South", "Manhattan", 40.7650, -73.9620),
    Zone(142, "Lincoln Square East", "Manhattan", 40.7730, -73.9820, "theatre"),
    Zone(87, "Financial District North", "Manhattan", 40.7090, -74.0080),
]
ZONE_IDS: list[int] = [z.id for z in ZONES]
ZONE_BY_ID: dict[int, Zone] = {z.id: z for z in ZONES}
ZONE_INDEX: dict[int, int] = {z.id: i for i, z in enumerate(ZONES)}


def haversine_km(a: Zone, b: Zone) -> float:
    r = 6371.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp, dl = p2 - p1, math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def distance_matrix() -> list[list[float]]:
    return [[haversine_km(a, b) for b in ZONES] for a in ZONES]


def neighbors(zone_id: int, radius_km: float = 1.6) -> list[int]:
    z = ZONE_BY_ID[zone_id]
    return [o.id for o in ZONES if o.id != zone_id and haversine_km(z, o) <= radius_km]


def zone_name(zone_id: int) -> str:
    z = ZONE_BY_ID.get(zone_id)
    return z.name if z else f"Zone {zone_id}"
