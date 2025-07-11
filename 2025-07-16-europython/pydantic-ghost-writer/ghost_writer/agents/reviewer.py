from typing import TypedDict

from pydantic_ai import Agent

from ghost_writer.agents.shared import get_guidelines, load_prompt


class Score(TypedDict):
    score: int
    reason: str


reviewer_agent = Agent(
    'anthropic:claude-3-7-sonnet-latest',
    output_type=Score,
    tools=[get_guidelines],
    instructions=load_prompt(role='reviewer', content_type='blog_post'),
)


@reviewer_agent.tool_plain
async def get_writer_instructions() -> str:
    """Get the blog post writing instructions that the writer should follow."""
    return load_prompt(role='writer', content_type='blog_post')
