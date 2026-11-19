"""Download and preprocess the open data behind the geo risk service.

Run once with `make build-data` (`uv run -m geo_service.build_data`). It is the only code in this example
that talks to a third party; the service and the demo read what it writes to `geo_service/data/`.

Sources, all for central Montpelier, Vermont:
- Address points: Vermont E911 site locations, from the Vermont Open Geodata Portal (VCGI).
- Flood zones: FEMA National Flood Hazard Layer, from Esri's hosted copy (FEMA's own server refuses
  connections from outside the US).
- Rivers: USGS National Hydrography Dataset.
- Elevation: USGS 3DEP, sampled at each property and at the nearest point on the water.
- Aerial imagery: USDA NAIP, from the USGS image service, one tile per property (see `download_images.py`).

Raw responses are cached under `raw_data/`, so a second run makes no requests.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx2
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import nearest_points, unary_union

from .download_images import download_images
from .geo import BBOX, project, to_lon_lat, to_metres

THIS_DIR = Path(__file__).parent
RAW_DIR = THIS_DIR / 'raw_data'
DATA_DIR = THIS_DIR / 'data'

PROPERTY_COUNT = 300

VCGI = 'https://services1.arcgis.com/BkFxaEFNwHqX3tAw/arcgis/rest/services'
ADDRESSES_URL = f'{VCGI}/FS_VCGI_OPENDATA_Emergency_ESITE_point_SP_v1/FeatureServer/0/query'
FLOOD_URL = (
    'https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services'
    '/USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer/0/query'
)
NHD_URL = 'https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/{layer}/query'
ELEVATION_URL = 'https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/getSamples'

BBOX_QUERY = {
    'geometry': ','.join(str(v) for v in BBOX),
    'geometryType': 'esriGeometryEnvelope',
    'inSR': '4326',
    'outSR': '4326',
    'spatialRel': 'esriSpatialRelIntersects',
    'f': 'geojson',
}

# Site types worth insuring; sheds, utilities and the like are left out.
SITE_TYPES = {
    'SINGLE FAMILY DWELLING': 'house',
    'MULTI-FAMILY DWELLING': 'flats',
    'COMMERCIAL': 'commercial',
    'COMMERCIAL W/RESIDENCE': 'mixed use',
    'OFFICE BUILDING': 'office',
    'OTHER COMMERCIAL': 'commercial',
    'RESTAURANT': 'restaurant',
    'RETAIL FACILITY': 'retail',
}


async def cached_json(client: httpx2.AsyncClient, name: str, url: str, params: dict[str, Any]) -> Any:
    path = RAW_DIR / f'{name}.json'
    if not path.exists():
        response = await client.post(url, data=params)
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            raise RuntimeError(f'{name}: {data["error"]}')
        path.write_text(json.dumps(data))
    return json.loads(path.read_text())


def flood_class(properties: dict[str, Any]) -> str | None:
    """Collapse FEMA's zone codes into the classes the service reports."""
    if properties.get('ZONE_SUBTY') == 'Floodway':
        return 'floodway'
    if properties.get('SFHA_TF') == 'T':
        return '1_percent'
    if '0.2' in (properties.get('ZONE_SUBTY') or ''):
        return '0.2_percent'
    return None


async def sample_elevations(client: httpx2.AsyncClient, name: str, points: list[tuple[float, float]]) -> list[float]:
    """Ground elevation in metres at each lon/lat point."""
    elevations: list[float] = []
    for start in range(0, len(points), 100):
        batch = points[start : start + 100]
        geometry = {'points': [list(p) for p in batch], 'spatialReference': {'wkid': 4326}}
        params = {
            'geometry': json.dumps(geometry),
            'geometryType': 'esriGeometryMultipoint',
            'returnFirstValueOnly': 'true',
            'f': 'json',
        }
        data = await cached_json(client, f'{name}_{start}', ELEVATION_URL, params)
        samples = sorted(data['samples'], key=lambda s: s['locationId'])
        if len(samples) != len(batch):
            raise RuntimeError(f'{name}: {len(samples)} elevation samples for {len(batch)} points')
        elevations.extend(round(float(s['value']), 2) for s in samples)
    return elevations


async def main() -> None:
    for directory in RAW_DIR, DATA_DIR:
        directory.mkdir(parents=True, exist_ok=True)

    headers = {'User-Agent': 'pydantic-demo-data-build/0.1'}
    async with httpx2.AsyncClient(timeout=120, headers=headers) as client:
        addresses = await cached_json(client, 'addresses', ADDRESSES_URL, {**BBOX_QUERY, 'outFields': '*'})
        flood = await cached_json(
            client, 'flood_zones', FLOOD_URL, {**BBOX_QUERY, 'outFields': 'FLD_ZONE,ZONE_SUBTY,SFHA_TF'}
        )
        river_areas = await cached_json(
            client, 'river_areas', NHD_URL.format(layer=9), {**BBOX_QUERY, 'outFields': 'gnis_name'}
        )
        flowlines = await cached_json(
            client, 'flowlines', NHD_URL.format(layer=6), {**BBOX_QUERY, 'outFields': 'gnis_name'}
        )

        # Flood zones and water, clipped to a little beyond the town so the files stay small.
        clip = box(BBOX[0] - 0.005, BBOX[1] - 0.005, BBOX[2] + 0.005, BBOX[3] + 0.005)
        zone_features: list[dict[str, Any]] = []
        for feature in flood['features']:
            zone = flood_class(feature['properties'])
            geometry = shape(feature['geometry']).buffer(0).intersection(clip)
            if zone and not geometry.is_empty:
                properties = {'zone': zone, 'fema_zone': feature['properties']['FLD_ZONE']}
                zone_features.append({'type': 'Feature', 'properties': properties, 'geometry': mapping(geometry)})

        water_features: list[dict[str, Any]] = []
        for feature in [*river_areas['features'], *flowlines['features']]:
            geometry = shape(feature['geometry']).intersection(clip)
            if not geometry.is_empty:
                properties = {'name': feature['properties'].get('gnis_name')}
                water_features.append({'type': 'Feature', 'properties': properties, 'geometry': mapping(geometry)})
        water = unary_union([project(shape(f['geometry'])) for f in water_features])

        # Pick the properties: everything in or near a flood zone first, then the rest in a stable
        # pseudo-random order, so the portfolio has plenty of borderline cases.
        mapped_zones = unary_union([project(shape(f['geometry'])) for f in zone_features])
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        seen: set[str] = set()
        for feature in addresses['features']:
            p = feature['properties']
            address = (p.get('PRIMARYADDRESS') or '').strip()
            if p.get('SITETYPE') not in SITE_TYPES or not address or address in seen:
                continue
            seen.add(address)
            lon, lat = feature['geometry']['coordinates'][:2]
            near_flood = Point(to_metres(lon, lat)).distance(mapped_zones) < 60
            order = int(hashlib.sha256(address.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            candidates.append((order if near_flood else 1 + order, address, feature))
        candidates.sort(key=lambda c: c[:2])
        chosen = sorted(candidates[:PROPERTY_COUNT], key=lambda c: c[1])

        properties: list[dict[str, Any]] = []
        water_points: list[tuple[float, float]] = []
        for index, (_, address, feature) in enumerate(chosen, start=1):
            p = feature['properties']
            lon, lat = feature['geometry']['coordinates'][:2]
            nearest_on_water, _ = nearest_points(water, Point(to_metres(lon, lat)))
            water_points.append(to_lon_lat(nearest_on_water.x, nearest_on_water.y))
            properties.append(
                {
                    'id': f'P{index:03d}',
                    'address': address.title(),
                    'kind': SITE_TYPES[p['SITETYPE']],
                    'units': p.get('UNITCOUNT') or 1,
                    'lon': round(lon, 6),
                    'lat': round(lat, 6),
                }
            )

        elevations = await sample_elevations(client, 'elevation_property', [(p['lon'], p['lat']) for p in properties])
        water_elevations = await sample_elevations(client, 'elevation_water', water_points)
        for p, elevation, water_elevation in zip(properties, elevations, water_elevations):
            p['elevation_m'] = elevation
            p['water_elevation_m'] = water_elevation

    (DATA_DIR / 'properties.json').write_text(json.dumps(properties, indent=1))
    (DATA_DIR / 'flood_zones.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': zone_features}))
    (DATA_DIR / 'water.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': water_features}))
    await download_images(properties)
    print(
        f'{len(properties)} properties, {len(zone_features)} flood zone polygons, {len(water_features)} water features'
    )


if __name__ == '__main__':
    asyncio.run(main())
