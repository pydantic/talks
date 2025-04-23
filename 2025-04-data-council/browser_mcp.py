import logfire
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

logfire.configure(scrubbing=False, service_name='browse')
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

browser_mcp = MCPServerStdio('npx', args=['-Y', '@playwright/mcp@latest'])

agent = Agent(
    'anthropic:claude-3-7-sonnet-latest',
    mcp_servers=[browser_mcp],
)


async def main():
    async with agent.run_mcp_servers():
        result = await agent.run(
            'get the most recent blog post from pydantic.dev '
            'which should contain multiple announcements, '
            'summaries those annoucements as a list.'
        )
    print(result.output)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
