"""Geometry helpers shared by the data build and the service.

Everything is measured in metres on a flat local grid centred on the town. Over a few kilometres that is
accurate to well under a metre, which avoids needing a projection library.
"""

import math
from typing import Any

import numpy as np
import shapely
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

# Central Montpelier, Vermont: west, south, east, north.
BBOX = (-72.592, 44.252, -72.562, 44.268)
ORIGIN_LON = (BBOX[0] + BBOX[2]) / 2
ORIGIN_LAT = (BBOX[1] + BBOX[3]) / 2

_M_PER_DEG_LAT = 110_574.0
_M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(ORIGIN_LAT))


def to_metres(lon: float, lat: float) -> tuple[float, float]:
    return (lon - ORIGIN_LON) * _M_PER_DEG_LON, (lat - ORIGIN_LAT) * _M_PER_DEG_LAT


def to_lon_lat(x: float, y: float) -> tuple[float, float]:
    return ORIGIN_LON + x / _M_PER_DEG_LON, ORIGIN_LAT + y / _M_PER_DEG_LAT


def project(geometry: BaseGeometry) -> BaseGeometry:
    """Project a lon/lat geometry onto the local grid."""
    scale = np.array([_M_PER_DEG_LON, _M_PER_DEG_LAT])
    return shapely.transform(geometry, lambda coords: (coords - (ORIGIN_LON, ORIGIN_LAT)) * scale)


def project_geojson(geometry: dict[str, Any]) -> BaseGeometry:
    return project(shape(geometry))
