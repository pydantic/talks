from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from datetime import datetime

from devtools import debug
from pydantic import AwareDatetime, BaseModel
from pydantic_ai import Agent, RunContext
from typing_extensions import TypedDict


class TimeRangeBuilderSuccess(BaseModel, use_attribute_docstrings=True):
    """Response when a time range could be successfully generated."""

    min_timestamp_with_offset: AwareDatetime
    """A datetime in ISO format with timezone offset."""
    max_timestamp_with_offset: AwareDatetime
    """A datetime in ISO format with timezone offset."""
    explanation: str | None
    """
    A brief explanation of the time range that was selected.

    For example, if a user only mentions a specific point in time, you might explain that you selected a 10 minute
    window around that time.
    """

    def __str__(self):
        lines = [
            'TimeRangeBuilderSuccess:',
            f'* min_timestamp_with_offset: {self.min_timestamp_with_offset:%A, %B %d, %Y %H:%M:%S %Z}',
            f'* max_timestamp_with_offset: {self.max_timestamp_with_offset:%A, %B %d, %Y %H:%M:%S %Z}',
        ]
        if self.explanation is not None:
            lines.append(f'* explanation: {self.explanation}')
        return '\n'.join(lines)


class TimeRangeBuilderError(BaseModel):
    """Response when a time range cannot not be generated."""

    error_message: str

    def __str__(self):
        return f'TimeRangeBuilderError:\n* {self.error_message}'


TimeRangeResponse = TimeRangeBuilderSuccess | TimeRangeBuilderError


class TimeRangeInputs(TypedDict):
    """The inputs for the time range inference agent."""

    prompt: str
    now: AwareDatetime


@dataclass
class TimeRangeDeps:
    now: datetime = field(default_factory=lambda: datetime.now().astimezone())


time_range_agent = Agent[TimeRangeDeps, TimeRangeResponse](
    'openai:gpt-4o',
    output_type=TimeRangeResponse,  # type: ignore  # we can't yet annotate something as receiving a TypeForm
    deps_type=TimeRangeDeps,
    system_prompt='Convert the user request into a structured time range.',
    retries=1,
    instrument=True,
)


@time_range_agent.tool
def get_current_time(ctx: RunContext[TimeRangeDeps]) -> str:
    """Get the user's current time and timezone in the format 'Friday, November 22, 2024 11:15:14 PST'."""
    return f"The user's current time is {ctx.deps.now:%A, %B %d, %Y %H:%M:%S %Z}."


async def infer_time_range(inputs: TimeRangeInputs) -> TimeRangeResponse:
    """Infer a time range from a user prompt."""
    deps = TimeRangeDeps(now=inputs['now'])
    return (await time_range_agent.run(inputs['prompt'], deps=deps)).output


if __name__ == '__main__':
    import asyncio

    response = asyncio.run(
        infer_time_range(
            {'prompt': '2pm yesterday', 'now': datetime.now().astimezone()}
        )
    )

    debug(response)
