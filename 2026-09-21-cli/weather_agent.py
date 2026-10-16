"""Weather agent: Pydantic AI with tools, run through Monty codemode."""

from typing import TypedDict

from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode

weather_agent = Agent(
    'gateway/openai-responses:gpt-5.6-terra',
    capabilities=[CodeMode()],
    instructions='Combine tool calls into a single code execution call',
)


@weather_agent.tool_plain
def get_lat_long(location: str) -> tuple[float, float]:
    """Get the coordinates of a location.

    Returns: A tuple of `(latitude, longitude)`.
    """
    return 0.0, 0.0


class Weather(TypedDict):
    temp_f: int
    condition: str


@weather_agent.tool_plain
def get_weather(lat: float, lon: float) -> Weather:
    """Get current weather at a location."""
    return Weather(temp_f=72, condition='sunny')


@weather_agent.tool_plain
def convert_temp(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return round((fahrenheit - 32) * 5 / 9, 1)


if __name__ == '__main__':
    import logfire

    logfire.configure(service_name='weather-agent')
    logfire.instrument_pydantic_ai()
    logfire.instrument_monty()

    result = weather_agent.run_sync("What's the weather in Paris and London, in Celsius?")
    print(result.output)
