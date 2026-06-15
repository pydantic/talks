




# Model to Market

## London, 15 June 2026

Slides at <https://github.com/pydantic/talks>




















## What he **** is Pydantic?

* **Pydantic Validator** - open source validation library with 1B downloads
* **Pydantic AI** - Agent Framework
* **Pydantic Logfire** - observability tool from AI
* AI Gateway, prompt management, agent optimization, AI SRE, Agent governance















## 1: Pydantic AI

Type-safe AI Agent orchestration.

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
    'anthropic:claude-sonnet-4-5',
    output_type=City,
    instructions='Extract information about the city',
)
result = agent.run_sync("London was founded in 50AD it's located at 51.5074, 0.1278")
print(repr(result.output))
```














## 2: Pydantic Logfire

Observability tool from AI to API.

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
    'anthropic:claude-sonnet-4-5',
    output_type=City,
    instructions='Extract information about the city',
)
result = agent.run_sync("London was founded in 50AD it's located at 51.5074, 0.1278")
logfire.info(f'{result.output=}')
```













## 3: Pydantic AI Gateway
AI model routing that simply works.

```py {title="pydantic-ai-gateway.py"}
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
    'gateway/anthropic:claude-sonnet-4-5',
    # 'gateway/openai:gpt-5-nano',
    output_type=City,
    instructions='Extract information about the city',
    retries=4,
)
result = agent.run_sync("London was founded in 50AD it's located at 51.5074, 0.1278")
print(repr(result.output))
```













## 4: Prompt management and optimization

Agent optimization with Pydantic Logfire agent optimization.

```py
from pathlib import Path
import logfire

variables = logfire.VariablesOptions()
logfire.configure(variables=variables)
logfire.instrument_pydantic_ai()

my_agent_instructions = logfire.var(
    'my_agent_instructions',
    default=default_value,
)

with my_agent_instructions.get() as resolve:
    result = agent.run_sync(
        '...',
        instructions=resolve.value,
    )
```









## Thank you

Find us:
* Slides: <https://github.com/pydantic/talks>
* <https://pydantic.dev>
* <https://pydantic.dev/links>
* <https://github.com/pydantic>
* <https://x.com/pydantic>
