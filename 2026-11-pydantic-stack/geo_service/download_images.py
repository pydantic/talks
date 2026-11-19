"""Download the aerial image for each property in `data/properties.json`.

Run with `make images` (`uv run -m geo_service.download_images`). The images are about 10 MB, so they
are not committed; the make targets that start the service run this first. Images already in
`data/images/` are skipped, so a second run makes no requests.

Imagery is USDA NAIP, from the USGS image service, one tile centred on each property.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2

from .geo import to_lon_lat, to_metres

DATA_DIR = Path(__file__).parent / 'data'
IMAGE_DIR = DATA_DIR / 'images'

IMAGE_PIXELS = 384
IMAGE_METRES = 110  # ground covered by one side of an image tile
IMAGERY_URL = 'https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage'


async def fetch_image(client: httpx2.AsyncClient, limit: asyncio.Semaphore, property_id: str, x: float, y: float):
    half = IMAGE_METRES / 2
    west, south = to_lon_lat(x - half, y - half)
    east, north = to_lon_lat(x + half, y + half)
    params = {
        'bbox': f'{west},{south},{east},{north}',
        'bboxSR': '4326',
        'size': f'{IMAGE_PIXELS},{IMAGE_PIXELS}',
        'format': 'jpg',
        'f': 'image',
    }
    async with limit:
        response = await client.get(IMAGERY_URL, params=params)
    response.raise_for_status()
    if not response.headers['content-type'].startswith('image/'):
        raise RuntimeError(f'{property_id}: imagery service returned {response.text[:200]}')
    # Write then rename, so an interrupted run never leaves half an image behind.
    path = IMAGE_DIR / f'{property_id}.jpg'
    partial = path.with_suffix('.part')
    partial.write_bytes(response.content)
    partial.rename(path)


async def download_images(properties: list[dict[str, Any]]) -> int:
    """Fetch the image for every property that does not have one yet, and return how many were fetched."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    missing = [p for p in properties if not (IMAGE_DIR / f'{p["id"]}.jpg').exists()]
    if missing:
        headers = {'User-Agent': 'pydantic-demo-data-build/0.1'}
        limit = asyncio.Semaphore(6)
        async with httpx2.AsyncClient(timeout=120, headers=headers) as client, asyncio.TaskGroup() as group:
            for p in missing:
                group.create_task(fetch_image(client, limit, p['id'], *to_metres(p['lon'], p['lat'])))
    return len(missing)


async def main() -> None:
    properties = json.loads((DATA_DIR / 'properties.json').read_text())
    fetched = await download_images(properties)
    if fetched:
        print(f'Downloaded {fetched} aerial images to {IMAGE_DIR}')


if __name__ == '__main__':
    asyncio.run(main())
