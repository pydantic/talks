# Data Council, Oakland California, April 2025

Slides at <https://github.com/pydantic/talks>

## whoami

**Samuel Colvin** — creator of Pydantic

Pydantic:
* Python library for data validation
* Created Pydantic in 2017 — long before Gen AI
* Became a company, backed by Sequoia in 2023 — released Logfire earlier this year
* Released Pydantic V2 last year, core rewritten in Rust
* downloaded >300M per month
* Used by all of FAANG
* Used by virtually every GenAI Python library — both provider SDKs and Agent Frameworks


Boring   •   Ubiquitous
















## what

"An Opinionated Blueprint for the Future of GenAI Applications"

* A lot has changed since I wrote the CFP for this...
* A lot hasn't changed — people still want to build
  reliable, scalable applications, and that's still hard.

In this workshop, we'll use **PydanticAI** & **Pydantic Logfire** to demonstrate:
* How to build typesafe agents
* The power of MCP for agents
* The importance of tracing and observability for AI Applications
* How evals fit into the picture














## What is an "Agent"?

This has been widely covered, but still seems to be a subject of dispute, so I'll explain what I mean.

From [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

![Agent loop diagram](agent-loop.png)

From [How We Build Effective Agents: Barry Zhang, Anthropic](https://youtu.be/D7_ipDqhtwk?&t=358)

**Agents are models using tools in a loop**
```py
env = Environment()
tools = Tools(env)
system_prompt = "Goals, constraints, and how to act"

while True:
    action = llm.run(system_prompt + env.state)
    env.state = tools.run(action)
```














## Enough pseudo code, show me a real example

```py {title="simplest_agent.py"}
from datetime import date
from pydantic_ai import Agent
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    dob: date
    city: str


agent = Agent(
    'openai:gpt-4o',
    output_type=Person,
    instructions='Extract information about the person',
)
result = agent.run_sync('Samuel lives in London and was born on Jan 28th 87')
print(repr(result.output))
```

This doesn't look much like a loop, but what if validation fails...

```py title="agent_might_fail.py"
from datetime import date
from pydantic_ai import Agent
from pydantic import BaseModel, field_validator

class Person(BaseModel):
    """Definition of an historic person"""
    name: str
    dob: date
    city: str

    @field_validator('dob')
    def validate_dob(cls, v: date) -> date:
        if v >= date(1900, 1, 1):
            raise ValueError('The person must be born in the 19th century')
        return v


agent = Agent(
    'google-vertex:gemini-2.0-flash',
    output_type=Person,
    instructions='Extract information about the person',
)
result = agent.run_sync('Samuel lives in London and was born on Jan 28th 87')
print(repr(result.output))
```
