import asyncio
from datetime import date

import httpx
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.tools import RunContext

agent = Agent(
    deps_type=httpx.AsyncClient,
    instructions="""
You are an AI assistant that helps people find answers from documentation.

Use the provided tools to find answers, never guess at the answer from your own knowledge.
    """,
    toolsets=[
        # MCPServerStreamableHTTP('http://localhost:8001/mcp'),
        MCPServerStreamableHTTP('https://py-ai-mcp.fastmcp.app/mcp'),
    ],
)


@agent.instructions
def add_date() -> str:
    return f"Today's date is: {date.today()}"


@agent.tool
async def fetch(ctx: RunContext[httpx.AsyncClient], url: str) -> str:
    r = await ctx.deps.get(url)
    if r.status_code > 299:
        raise ModelRetry(f'Failed to fetch {url}: {r.status_code}')
    return r.text


async def main():
    async with httpx.AsyncClient(timeout=30) as client:
        result = await agent.run(
            'what event is happening today',
            deps=client,
            model='gateway/openai:gpt-4.1',
        )
        print(result.output)


if __name__ == '__main__':
    import logfire

    logfire.configure()
    logfire.instrument_pydantic_ai()
    logfire.instrument_mcp()
    asyncio.run(main())
