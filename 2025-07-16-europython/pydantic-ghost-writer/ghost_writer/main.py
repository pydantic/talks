import httpx
import logfire

from ghost_writer.agents.writer import WriterAgentDeps, writer_agent

writer_http_client = httpx.AsyncClient()
"""An HTTP client for the writer agent."""

# Configure Logfire for observability
logfire.configure(scrubbing=False, console=logfire.ConsoleOptions(show_project_link=False))
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(writer_http_client, capture_all=True)


# Main function to generate blog content
async def generate_blog_post(
    topic: str,
    author: str = '',
    author_role: str = '',
    user_requirements: str = '',
    opinions: str = '',
    examples: str = '',
    reference_links: list[str] = [],
) -> str:
    """Generate a blog post about the given topic."""

    deps = WriterAgentDeps(
        http_client=writer_http_client,
        author=author,
        author_role=author_role,
        user_requirements=user_requirements,
        opinions=opinions,
        examples=examples,
        reference_links=reference_links,
    )

    async with writer_http_client:
        response = await writer_agent.run(
            f'Write a blog post about {topic}.',
            deps=deps,
        )
    return response.output
