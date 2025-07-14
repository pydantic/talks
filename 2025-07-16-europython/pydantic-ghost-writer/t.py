# type: ignore

logfire.configure(scrubbing=False, token='pylf_v1_local_Vk7V4XrgmsfdXrLXtw5f9vDQQTj8sSgY0xM6FFlpm8JD', advanced=logfire.AdvancedOptions(base_url='http://localhost:8000/'))



# Snippets:

## FastAPI + SQLAlchemy

from fastapi import FastAPI

app = FastAPI()

from sqlalchemy import select

class Attendee:
    id: int


@app.get('/attendees/{id}')
def get_attendee(id: int) -> Attendee:
    return select(Attendee).where(Attendee.id == id)


## Pydantic AI

from typing import Annotated
from annotated_types import Interval
from pydantic_ai import Agent

agent = Agent(
    'google-vertex:gemini-2.5-pro',
    output_type=Annotated[int, Interval(ge=1, le=10)],
    system_prompt='Give me a score between 1 and 10.',
)




## Our agent


agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    output_type=str,
    instructions='You are writing a blog post for the Pydantic blog.',
)


## Our agent, with more context

from dataclasses import dataclass
from pydantic_ai import RunContext

@dataclass
class AgentDeps:
    blog_author: str
    author_role: str
    reference_links: list[str]

agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    output_type=str,
    deps_type=AgentDeps,
    instructions='You are writing a blog post for the Pydantic blog.',
)


await agent.run('Write a blog post about ...', deps=AgentDeps(blog_author='Victorien', ...))


@agent.instructions
def add_author_info(ctx: RunContext[AgentDeps]) -> str:
    return f"""
    Author: {ctx.deps.blog_author}
    Author role: {ctx.deps.author_role}
    Reference links that you may query: {', '.join(ctx.deps.reference_links)}
    """

## Our agent, with more capabilities

import trafilatura
import requests
from pydantic import HttpUrl

@agent.tool
def extract_technical_content(url: HttpUrl) -> str:
    """Extract technical content from one of the reference links."""

    response = requests.get(str(url))

    return trafilatura.extract(
        response.text,
        output_format="html",
        favor_precision=True,
        include_formatting=True,
        include_tables=True,
    )



## Our reviewer agent


class Score(TypedDict):
    score: int
    reason: str


reviewer_agent = Agent(
    "anthropic:claude-3-5-sonnet-latest",
    output_type=Score,
    instructions="",
)

## Expose reviewer agent

writer_agent = agent


@writer_agent.tool
async def review_page_content(content: str) -> Score:
    """Review the content and return a score with feedback."""
    result = await reviewer_agent.run(f"Review this content:\n\n{content}")
    return result.output

