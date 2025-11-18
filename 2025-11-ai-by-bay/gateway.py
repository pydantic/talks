from datetime import date
from typing import TypedDict

from pydantic import BaseModel
from pydantic_ai import Agent

Location = TypedDict("Location", {"lat": float, "lng": float})


class City(BaseModel):
    name: str
    founded: date
    location: Location


agent = Agent(
    "gateway/anthropic:claude-haiku-4-5",
    # 'gateway/openai:gpt-4.1',
    output_type=City,
    instructions="Extract information about the city",
)
result = agent.run_sync("London was founded in 50AD it's located at 51.5074, 0.1278")
print(repr(result.output))
