



# Durable Agents: long running AI workflows in a flakey world

## AI by the Bay, November 2025

Slides at <https://github.com/pydantic/talks>

Samuel Colvin




















## what

**Durable Agents: long running AI workflows in a flakey world**

* Part of the boarder topic of Engineering in AI - reliable AI
* But won't make much sense without some background...
* What does Pydantic do?











## 1: Pydantic Validation

```py {title="pydantic-validation.py"}
from datetime import date
from pydantic import BaseModel


class City(BaseModel):
    name: str
    founded: date
    location: tuple[float, float]


city = City(name='London', founded='0050-01-01', location=['51.5074', b'0.1278'])
print(repr(city))
```















## 2: Pydantic AI

```py {title="pydantic-ai.py"}
from datetime import date
from typing import TypedDict
from pydantic_ai import Agent
from pydantic import BaseModel

class City(BaseModel):
    name: str
    founded: date
    location: TypedDict('Location', {'lat': float, 'lng': float})

agent = Agent(
    'openai:gpt-4.1',
    output_type=City,
    instructions='Extract information about the city',
)
result = agent.run_sync("London was founded in 50AD it's located at 51.5074, 0.1278")
print(repr(result.output))
```














## 3: Pydantic Logfire

```py {title="pydantic-logfire.py"}
from datetime import date
from typing import TypedDict
from pydantic_ai import Agent
from pydantic import BaseModel

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

class City(BaseModel):
    name: str
    founded: date
    location: TypedDict('Location', {'lat': float, 'lng': float})

agent = Agent(
    'openai:gpt-4.1',
    output_type=City,
    instructions='Extract information about the city',
)
result = agent.run_sync("London was founded in 50AD it's located at 51.5074, 0.1278")
logfire.info(f'{result.output=}')
```













## 4: Pydantic AI Gateway

```py {title="pydantic-ai-gateway.py"}
from datetime import date
from typing import TypedDict
from pydantic_ai import Agent
from pydantic import BaseModel

class City(BaseModel):
    name: str
    founded: date
    location: TypedDict('Location', {'lat': float, 'lng': float})

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-5',
    output_type=City,
    instructions='Extract information about the city',
)
result = agent.run_sync("London was founded in 50AD it's located at 51.5074, 0.1278")
print(repr(result.output))
```














## So the point is:

**AI is still Engineering!**

* Lots of people seem to argue otherwise. I think they're wrong.
* AI doesn't run in a vacuum, isolated from practical challenges of the cloud
* The advent of AI makes good engineering practices MORE important, not less.

* System design, type safety, observability, durable execution - this things matter more than ever














## What is durable execution?

* networks are unreliable, nodes die - things are unreliable

* durable execution should allow us to:
  * recover from intermittent network failurs
  * recover from the host dieing mid-way through a task
  * write procedural code that runs over hours, weeks or years

* Pydantic AI supports: **Temporal**, **DBOS** and **Prefect**

* Temporal relies on the separation of Worflows and Activities












## Let's look at an example

...















## Thank you

Find us:
* pydantic.dev
* github.com/pydantic
* x.com/pydantic
