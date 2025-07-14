from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ghost_writer.agents.shared import get_guidelines, load_prompt


class Review(BaseModel):
    score: Annotated[int, Field(ge=0, le=10)]
    """A score between 1 and 10."""

    feedback: str
    """Feedback on what can be improved."""


reviewer_agent = Agent(
    'anthropic:claude-3-7-sonnet-latest',
    output_type=Review,
    tools=[get_guidelines],
    instructions=load_prompt(role='reviewer', content_type='blog_post'),
)


@reviewer_agent.tool_plain
async def get_writer_instructions() -> str:
    """Get the blog post writing instructions that the writer should follow."""
    return load_prompt(role='writer', content_type='blog_post')
