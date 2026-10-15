"""Prompts are managed variables: fetch `weather-agent` from Logfire, render it, run the agent."""

import logfire
from pydantic import BaseModel

from weather_agent import weather_agent


class WeatherInputs(BaseModel):
    units: str


prompt = logfire.template_var(
    name='prompt__weather_agent',
    type=str,
    default='Combine tool calls into a single code execution call. Answer in {{units}}.',
    inputs_type=WeatherInputs,
)

if __name__ == '__main__':
    logfire.configure(service_name='weather-agent')
    logfire.instrument_pydantic_ai()
    logfire.instrument_monty()

    with prompt.get(WeatherInputs(units='celsius'), label='production') as resolved:
        result = weather_agent.run_sync(
            "What's the weather in Paris and London?",
            instructions=resolved.value,
        )
    print(result.output)
