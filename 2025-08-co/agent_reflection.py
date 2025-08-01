from datetime import date

import logfire
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent

logfire.configure(service_name='agent-retry')
logfire.instrument_pydantic_ai()


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
    'openai:gpt-4.1',
    output_type=Person,
    instructions='Extract information about the person',
)
result = agent.run_sync("Samuel lived in London and was born on Jan 28th '87")
print(repr(result.output))
