"""The agent's tools. With code mode they become async Python functions inside the sandbox.

Each lookup is one request to the geo risk API, so a question about the whole portfolio is hundreds of
requests, made from one piece of model-written code.
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, TypedDict

import httpx
from pydantic_ai import RunContext

from . import decision, vision
from .policy import Facts, Zone

GEO_SERVICE_URL = os.getenv('GEO_SERVICE_URL', 'http://localhost:8001')


@dataclass
class Deps:
    http: httpx.AsyncClient
    events: asyncio.Queue[dict[str, Any]] | None = None
    """Where `show_map` sends map updates for the web UI; None when there is no UI."""
    model_limit: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(16))
    """Caps concurrent model calls made from the sandbox."""
    observations: dict[str, asyncio.Task[vision.AerialObservations]] = field(
        default_factory=dict[str, asyncio.Task[vision.AerialObservations]]
    )


def http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=GEO_SERVICE_URL, timeout=30)


class Property(TypedDict):
    id: str
    address: str
    kind: str
    units: int
    lon: float
    lat: float


class FloodZone(TypedDict):
    zone: Zone
    distance_to_1_percent_zone_m: float


class Elevation(TypedDict):
    elevation_m: float
    water_elevation_m: float
    height_above_water_m: float


class DistanceToWater(TypedDict):
    distance_m: float
    water_name: str | None


class Observations(TypedDict):
    water_visible: bool
    on_the_bank: bool
    ground_cover: str
    notes: str


class DecisionDict(TypedDict):
    property_id: str
    outcome: str
    flood_excess: bool
    confidence: dict[str, float] | None
    decided_by: str


async def _get(ctx: RunContext[Deps], path: str, **params: Any) -> Any:
    response = await ctx.deps.http.get(path, params=params)
    response.raise_for_status()
    return response.json()


async def list_properties(ctx: RunContext[Deps]) -> list[Property]:
    """List every property in the portfolio."""
    return await _get(ctx, '/properties')


async def get_flood_zone(ctx: RunContext[Deps], property_id: str, buffer_m: float = 0) -> FloodZone:
    """Get the flood zone a property is in.

    Args:
        property_id: The property's id, e.g. 'P042'.
        buffer_m: Grow every flood zone by this many metres first, to test a what-if.
    """
    return await _get(ctx, f'/properties/{property_id}/flood-zone', buffer_m=buffer_m)


async def get_elevation(ctx: RunContext[Deps], property_id: str) -> Elevation:
    """Get the ground elevation at a property and its height above the nearest water."""
    return await _get(ctx, f'/properties/{property_id}/elevation')


async def distance_to_water(ctx: RunContext[Deps], property_id: str) -> DistanceToWater:
    """Get the distance from a property to the nearest river or brook."""
    return await _get(ctx, f'/properties/{property_id}/distance-to-water')


async def _observe(ctx: RunContext[Deps], property_id: str) -> vision.AerialObservations:
    async def run() -> vision.AerialObservations:
        response = await ctx.deps.http.get(f'/properties/{property_id}/aerial-image')
        response.raise_for_status()
        async with ctx.deps.model_limit:
            return await vision.inspect_image(response.content)

    # Cache the task, not the result, so two calls for the same property share one model request.
    if property_id not in ctx.deps.observations:
        ctx.deps.observations[property_id] = asyncio.create_task(run())
    return await ctx.deps.observations[property_id]


async def inspect_aerial_image(ctx: RunContext[Deps], property_id: str) -> Observations:
    """Look at the aerial photograph of a property and describe what bears on its flood risk.

    This runs a vision model, so it is slower than the other tools: use it for the properties that
    need a closer look, not for the whole portfolio.
    """
    observations = await _observe(ctx, property_id)
    return observations.model_dump()  # pyright: ignore[reportReturnType]


async def facts_for(
    ctx: RunContext[Deps], property_id: str, flood_zone_buffer_m: float = 0, use_aerial_image: bool = False
) -> Facts:
    prop, flood_zone, elevation, water = await asyncio.gather(
        _get(ctx, f'/properties/{property_id}'),
        get_flood_zone(ctx, property_id, flood_zone_buffer_m),
        get_elevation(ctx, property_id),
        distance_to_water(ctx, property_id),
    )
    notes = (await _observe(ctx, property_id)).summary() if use_aerial_image else None
    return Facts(
        property_id=property_id,
        address=prop['address'],
        kind=prop['kind'],
        zone=flood_zone['zone'],
        distance_to_1_percent_zone_m=flood_zone['distance_to_1_percent_zone_m'],
        height_above_water_m=elevation['height_above_water_m'],
        distance_to_water_m=water['distance_m'],
        water_name=water['water_name'],
        aerial_notes=notes,
    )


async def decide(
    ctx: RunContext[Deps], property_id: str, flood_zone_buffer_m: float = 0, use_aerial_image: bool = False
) -> DecisionDict:
    """Make the underwriting decision for a property: accept, refer or decline.

    It gathers the property's flood zone, elevation and distance to water itself, so there is no need
    to look those up first. Always use this for decisions; do not decide yourself.

    Args:
        property_id: The property's id, e.g. 'P042'.
        flood_zone_buffer_m: Grow every flood zone by this many metres first, to test a what-if.
        use_aerial_image: Also take the aerial photograph into account. Slower, so use it for referrals.
    """
    facts = await facts_for(ctx, property_id, flood_zone_buffer_m, use_aerial_image)
    async with ctx.deps.model_limit:
        result = await decision.decide(facts)
    return result.model_dump(mode='json')  # pyright: ignore[reportReturnType]


async def show_map(ctx: RunContext[Deps], title: str, groups: dict[str, list[str]]) -> str:
    """Show properties on the user's map, coloured by group.

    Args:
        title: A short title for the map, e.g. 'Decisions for the portfolio'.
        groups: Property ids by group label, e.g. {'decline': ['P001'], 'refer': ['P002', 'P003']}.
            The labels 'accept', 'refer' and 'decline' get green, amber and red; others get their own colours.
    """
    if ctx.deps.events is not None:
        await ctx.deps.events.put({'type': 'map', 'title': title, 'groups': groups})
    return f'Map updated with {sum(len(ids) for ids in groups.values())} properties.'


TOOLS = [list_properties, get_flood_zone, get_elevation, distance_to_water, inspect_aerial_image, decide, show_map]
