"""The main agent, writing the content."""

import os
from dataclasses import dataclass

import httpx
import trafilatura
from pydantic import HttpUrl
from pydantic_ai import Agent, RunContext

from ghost_writer.agents.github import ask_user_approval, create_blog_pr
from ghost_writer.agents.reviewer import Review, reviewer_agent
from ghost_writer.agents.shared import load_prompt


@dataclass
class WriterAgentDeps:
    http_client: httpx.AsyncClient
    author: str
    author_role: str
    user_requirements: str
    opinions: str
    examples: str
    reference_links: list[str]


writer_agent = Agent(
    'anthropic:claude-4-sonnet-20250514',
    deps_type=WriterAgentDeps,
    # These tools are meant for the Github PR creation:
    tools=[create_blog_pr, ask_user_approval] if 'GITHUB_TOKEN' is os.environ else [],
    instructions=load_prompt(role='writer', content_type='blog_post', add_guidelines=True),
)


@writer_agent.instructions
def add_user_info(ctx: RunContext[WriterAgentDeps]) -> str:
    """Add user information and instructions to the prompt."""
    instructions = ''
    deps = ctx.deps
    if deps.author:
        instructions += f'Author: {deps.author}\n'
    if deps.author_role:
        instructions += f'Author role: {deps.author_role}\n'
    if deps.user_requirements:
        instructions += f'User requirements: {deps.user_requirements}\n'
    if deps.opinions:
        instructions += f'Opinions: {deps.opinions}\n'
    if deps.examples:
        instructions += f'Examples: {deps.examples}\n'
    if deps.reference_links:
        instructions += f"Reference links that you may query using the 'extract_technical_content' tool: {', '.join(deps.reference_links)}"

    return instructions


# Writer agent-specific tools:


@writer_agent.tool
async def extract_technical_content(ctx: RunContext[WriterAgentDeps], url: HttpUrl) -> str:
    """Extract technical content optimized for code snippets and documentation."""

    response = await ctx.deps.http_client.get(str(url))

    extracted_content = trafilatura.extract(
        response.text,
        output_format='markdown',
        favor_precision=True,
        include_formatting=True,
        include_tables=True,
    )

    return extracted_content or 'no content available'


@writer_agent.tool_plain
async def review_page_content(content: str) -> Review:
    """Review the content and return a score with feedback."""
    result = await reviewer_agent.run(f'Review this content:\n\n{content}')
    return result.output
