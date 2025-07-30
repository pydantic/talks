from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ghost_writer.agents.shared import load_prompt


class Review(BaseModel):
    score: Annotated[int, Field(ge=0, le=10)]
    """A score between 1 and 10."""

    passed: bool
    """Boolean indicating if the content passes review (score >= 7)."""

    feedback: str
    """Feedback on what can be improved. Should contain specific examples."""


reviewer_agent = Agent(
    'anthropic:claude-3-7-sonnet-latest',
    output_type=Review,
    instructions=load_prompt(role='reviewer', content_type='blog_post', add_guidelines=True),
)
