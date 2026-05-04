import logfire
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode
from typing_extensions import TypedDict

logfire.configure()
logfire.instrument_pydantic_ai()

weather_agent = Agent(
    'gateway/openai:gpt-5.4-mini',
    capabilities=[CodeMode()],
    instructions='Combine tool calls into a single code execution when possible',
)


@weather_agent.tool_plain
def get_lat_long(location: str) -> tuple[float, float]:
    """Get current weather for a city."""
    return 0.0, 0.0


class Weather(TypedDict):
    temp_f: int
    condition: str


@weather_agent.tool_plain
def get_weather(lat: float, lon: float) -> Weather:
    """Get current weather for a city."""
    return Weather(temp_f=72, condition='sunny')


@weather_agent.tool_plain
def convert_temp(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return round((fahrenheit - 32) * 5 / 9, 1)


result = weather_agent.run_sync("What's the weather in Paris and Tokyo, in Celsius?")
print(result.output)
