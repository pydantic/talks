import logfire
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode

logfire.configure(service_name='codemode-examples')
logfire.instrument_pydantic_ai()

weather_agent = Agent('gateway/anthropic:claude-sonnet-4-6', capabilities=[CodeMode()])


@weather_agent.tool_plain
def get_weather(city: str) -> dict[str, str | int]:
    """Get current weather for a city."""
    return {'city': city, 'temp_f': 72, 'condition': 'sunny'}


@weather_agent.tool_plain
def convert_temp(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return round((fahrenheit - 32) * 5 / 9, 1)


result = weather_agent.run_sync("What's the weather in Paris and Tokyo, in Celsius?")
print(result.output)
