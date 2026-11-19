"""The geo risk API: flood zone, elevation, distance to water and an aerial image for each property.

It serves only the files that `build_data.py` wrote to `data/`, so it makes no outbound requests.
Run with `make serve` (`uv run uvicorn geo_service.app:app --port 8001`).
"""

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import logfire
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .geo import project_geojson, to_metres

DATA_DIR = Path(__file__).parent / 'data'

Zone = Literal['floodway', '1_percent', '0.2_percent', 'minimal']
RIVER_WIDTH_M = 80
# Most severe first: a property takes the first zone it falls in.
ZONE_ORDER: list[Zone] = ['floodway', '1_percent', '0.2_percent']


class Property(BaseModel):
    id: str
    address: str
    kind: str
    """What the building is used for, e.g. 'house', 'flats' or 'restaurant'."""
    units: int
    lon: float
    lat: float


class FloodZone(BaseModel):
    zone: Zone
    """The most severe flood zone the property is in: the regulatory floodway, the 1% annual chance
    zone, the 0.2% annual chance zone, or 'minimal' when it is outside all of them."""
    distance_to_1_percent_zone_m: float
    """Metres to the edge of the 1% annual chance zone, 0 when the property is inside it."""


class Elevation(BaseModel):
    elevation_m: float
    """Ground elevation at the property, in metres above sea level."""
    water_elevation_m: float
    """Elevation of the nearest river or brook."""
    height_above_water_m: float


class DistanceToWater(BaseModel):
    distance_m: float
    water_name: str | None


def _load_geojson(name: str) -> list[dict[str, Any]]:
    return json.loads((DATA_DIR / name).read_text())['features']


properties = {p['id']: p for p in json.loads((DATA_DIR / 'properties.json').read_text())}
points = {id: Point(to_metres(p['lon'], p['lat'])) for id, p in properties.items()}

_zone_features = _load_geojson('flood_zones.geojson')
zones: dict[Zone, BaseGeometry] = {
    zone: unary_union([project_geojson(f['geometry']) for f in _zone_features if f['properties']['zone'] == zone])
    for zone in ZONE_ORDER
}
# The floodway is part of the 1% zone, so distances to "the 1% zone" measure to either.
one_percent_zone = unary_union([zones['floodway'], zones['1_percent']])

water = [(f['properties']['name'], project_geojson(f['geometry'])) for f in _load_geojson('water.geojson')]

app = FastAPI(title='Geo risk API')
logfire.configure(service_name='geo-service')
logfire.instrument_fastapi(app)


def _point(property_id: str) -> Point:
    try:
        return points[property_id]
    except KeyError:
        raise HTTPException(404, f'Unknown property {property_id!r}')


@app.get('/properties')
async def list_properties() -> list[Property]:
    return [Property(**p) for p in properties.values()]


@app.get('/properties/{property_id}')
async def get_property(property_id: str) -> Property:
    _point(property_id)
    return Property(**properties[property_id])


@app.get('/properties/{property_id}/flood-zone')
async def flood_zone(
    property_id: str,
    buffer_m: Annotated[float, Query(ge=0, le=500, description='Grow every flood zone by this many metres.')] = 0,
) -> FloodZone:
    point = _point(property_id)
    zone = next((z for z in ZONE_ORDER if zones[z].distance(point) <= buffer_m), 'minimal')
    distance = max(one_percent_zone.distance(point) - buffer_m, 0)
    return FloodZone(zone=zone, distance_to_1_percent_zone_m=round(distance, 1))


@app.get('/properties/{property_id}/elevation')
async def elevation(property_id: str) -> Elevation:
    _point(property_id)
    p = properties[property_id]
    return Elevation(
        elevation_m=p['elevation_m'],
        water_elevation_m=p['water_elevation_m'],
        height_above_water_m=round(p['elevation_m'] - p['water_elevation_m'], 1),
    )


@app.get('/properties/{property_id}/distance-to-water')
async def distance_to_water(property_id: str) -> DistanceToWater:
    point = _point(property_id)
    distances = [(geometry.distance(point), name) for name, geometry in water]
    distance = min(d for d, _ in distances)
    # River polygons carry no name, only the centre lines do, so name the water after the nearest
    # named centre line within a river's width of the bank.
    named = [(d, name) for d, name in distances if name and d <= distance + RIVER_WIDTH_M]
    return DistanceToWater(distance_m=round(distance, 1), water_name=min(named)[1] if named else None)


@app.get('/properties/{property_id}/aerial-image')
async def aerial_image(property_id: str) -> FileResponse:
    _point(property_id)
    return FileResponse(DATA_DIR / 'images' / f'{property_id}.jpg', media_type='image/jpeg')


@app.get('/map/{layer}')
async def map_layer(layer: Literal['flood-zones', 'water']) -> FileResponse:
    """The flood zones or rivers as GeoJSON, for drawing the map."""
    return FileResponse(DATA_DIR / f'{layer.replace("-", "_")}.geojson', media_type='application/geo+json')
