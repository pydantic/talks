




# Py AI

## NYC, 21 November 2025

Slides at <https://github.com/pydantic/talks>

Samuel Colvin




















## what

**We do more than just validation.**


















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
    'gateway/openai:gpt-4.1',
    output_type=City,
    instructions='Extract information about the city',
)
result = agent.run_sync("London was founded in 50AD it's located at 51.5074, 0.1278")
print(repr(result.output))
```















## Let's look at an example

```bash
uv run uvicorn src.server:app
```















## Thank you

Find us:
* <https://pydantic.dev>
* <https://github.com/pydantic>
* <https://x.com/pydantic>
