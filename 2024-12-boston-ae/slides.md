

# Slides

## whoami

**Samuel Colvin** — creator of Pydantic

Pydantic:
* Python library for data validation
* Created Pydantic in 2017 — long before Gen AI
* Became a company, backed by Sequoia in 2023 — released Logfire earlier this year
* Released Pydantic V2 last year, core rewritten in Rust
* downloaded >300M per month
* Used by all of FAANG, OpenAI, Anthropic etc.


Ubiquitous   •   Boring   •   Beloved
































---

## Pydantic

Solving dumb problems for smart people.

```py
from datetime import date
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    dob: date


user = User(id='123', name='Samuel Colvin', dob='1987-01-28')
#> User(id=123, name='Samuel Colvin', dob=date(1987, 1, 28))

user = User.model_validate_json('{"id: 123, "name": "Samuel Colvin", "dob": "1987-01-28"}')
#> User(id=123, name='Samuel Colvin', dob=date(1987, 1, 28))

print(User.model_json_schema())
s = {
    'properties': {
        'id': {'title': 'Id', 'type': 'integer'},
        'name': {'title': 'Name', 'type': 'string'},
        'dob': {'format': 'date', 'title': 'Dob', 'type': 'string'},
    },
    'required': ['id', 'name', 'dob'],
    'title': 'User',
    'type': 'object',
}
```










---

## Pydantic in genAI

Over the last couple of years, Pydantic has been adopted by almost every Python LLM library.

Why?

```py
from datetime import date
from pydantic import BaseModel
from openai import OpenAI

class User(BaseModel):
    """Definition of a user"""
    id: int
    name: str
    dob: date

response = OpenAI().chat.completions.create(
    model='gpt-4o',
    messages=[
        {'role': 'system', 'content': 'Extract information about the user'},
        {'role': 'user', 'content': 'The user with ID 123 is called Samuel, born on Jan 28th 87'}
    ],
    tools=[
        {
            'function': {
                'name': User.__name__,
                'description': User.__doc__,
                'parameters': User.model_json_schema(),
            },
            'type': 'function'
        }
    ]
)
user = User.model_validate_json(response.choices[0].message.tool_calls[0].function.arguments)
print(user)
```









---

## PydanticAI

That same example with PydanticAI — Agent Framework for production.

```py
from datetime import date
from pydantic_ai import Agent
from pydantic import BaseModel

class User(BaseModel):
    """Definition of a user"""
    id: int
    name: str
    dob: date

agent = Agent(
    'openai:gpt-4o',
    result_type=User,
    system_prompt='Extract information about the user',
)
result = agent.run_sync('The user with ID 123 is called Samuel, born on Jan 28th 87')
print(result.data)
```











---

## Dependency Injection & type safety

In the real applications, LLMs don't operate in isolation.

```py
from dataclasses import dataclass
from typing import Any
from httpx import AsyncClient
from pydantic_ai import Agent, RunContext


@dataclass
class Deps:
    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None


weather_agent = Agent('openai:gpt-4o', deps_type=Deps)


@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> dict[str, float]:
    params = {'q': location_description, 'api_key': ctx.deps.geo_api_key}
    r = await ctx.deps.client.get('https://geocode.maps.co/search', params=params)
    r.raise_for_status()
    data = r.json()
    return {'lat': data[0]['lat'], 'lng': data[0]['lon']}


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    params = {'apikey': ctx.deps.weather_api_key, 'location': f'{lat},{lng}'}
    r = await ctx.deps.client.get('https://api.tomorrow.io/v4/weather/realtime', params=params)
    r.raise_for_status()
    values = r.json()['data']['values']
    return {
        'temperature': f'{values["temperatureApparent"]:0.0f}°C',
        'description': values['weatherCode'],
    }


async def main(weather_api_key: str, geo_api_key: str):
    async with AsyncClient() as client:
        deps = Deps(client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key)
        result = await weather_agent.run('What is the weather like in London and in Wiltshire?', deps=deps)
        print(result.data)
```






---

## Observability

How LLMs behave is inherently non-deterministic and slow — we sorely need Observability into what agents are doing.

Logfire to the rescue, just add a few lines to our code:

```py
import logfire
logfire.configure()
```

















---

### Next steps: Agent handoff

Making it easier to build multi-agent systems.

```py
from dataclasses import dataclass
from datetime import datetime

from httpx import AsyncClient
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.tools import AgentTool

@dataclass
class Deps:
    http_client: AsyncClient

class Flight(BaseModel):
    departure_time: datetime
    arrival_time: datetime
    destination: str

search_agent = Agent(model='openai:gpt-4o', deps_type=Deps, result_type=list[Flight])


class DesiredFlight(BaseModel):
    ideal_flight_time: datetime
    destination: str


control_agent = Agent(
    model='openai:gpt-4o',
    tools=[
        AgentTool(
            name='find_flights',
            agent=Agent(model='openai:gpt-4o', deps_type=Deps, result_type=list[Flight]),
            input_type=DesiredFlight,
        ),
        AgentTool(
            name='select_best_flight',
            agent=Agent(model='openai:gpt-4o', deps_type=Deps, result_type=Flight),
            input_type=DesiredFlight,
        )
    ],
    deps_type=Deps
)

result = control_agent.run_sync('Find me a flight to Alaska on the 20th of December')
```










---

## Next steps: Model Context Protocol

Released by Anthropic last week.

> ""


```py
from pydantic_ai import Agent
from pydantic_ai.toolsets import SlackToolset, OpenAPIToolset


agent = Agent(
    'openai:gpt-4o',
    toolsets=[SlackToolset(api_key=...), OpenAPIToolset(url='https://api.example.com')]
)
...
```























---

## Thank you!

Some useful links:

* Pydantic: docs.pydantic.dev

* Logfire: pydantic.dev/logfire

* PydanticAI: ai.pydantic.dev
