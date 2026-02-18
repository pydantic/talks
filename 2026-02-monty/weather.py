import asyncio
import json

import logfire
from httpx import AsyncClient
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets.code_execution import CodeExecutionToolset
from pydantic_ai.toolsets.function import FunctionToolset
from typing_extensions import TypedDict

logfire.configure(environment='weather')
logfire.instrument_pydantic_ai()


class LatLng(TypedDict):
    lat: float
    lng: float


weather_toolset: FunctionToolset[AsyncClient] = FunctionToolset()


@weather_toolset.tool
async def get_lat_lng(ctx: RunContext[AsyncClient], location_description: str) -> LatLng:
    """Get the latitude and longitude of a location."""
    r = await ctx.deps.get(
        'https://demo-endpoints.pydantic.workers.dev/latlng',
        params={'location': location_description},
    )
    r.raise_for_status()
    return json.loads(r.content)


@weather_toolset.tool
async def get_temp(ctx: RunContext[AsyncClient], lat: float, lng: float) -> float:
    """Get the temp at a location."""
    r = await ctx.deps.get(
        'https://demo-endpoints.pydantic.workers.dev/number',
        params={'min': 10, 'max': 30},
    )
    r.raise_for_status()
    return float(r.text)


@weather_toolset.tool
async def get_weather_description(ctx: RunContext[AsyncClient], lat: float, lng: float) -> str:
    """Get the weather description at a location."""
    r = await ctx.deps.get(
        'https://demo-endpoints.pydantic.workers.dev/weather',
        params={'lat': lat, 'lng': lng},
    )
    r.raise_for_status()
    return r.text


agent = Agent(
    'gateway/anthropic:claude-sonnet-4-5',
    # toolsets=[weather_toolset],
    toolsets=[CodeExecutionToolset(toolset=weather_toolset)],
    deps_type=AsyncClient,
)


async def main():
    async with AsyncClient() as client:
        await agent.run('Compare the weather of London, Paris, and Tokyo.', deps=client)


if __name__ == '__main__':
    asyncio.run(main())
