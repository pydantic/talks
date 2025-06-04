from datetime import date

import logfire
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

logfire.configure(service_name='mcp-client')

logfire.instrument_pydantic_ai()
logfire.instrument_mcp()

server = MCPServerStdio(command='uv', args=['run', 'pypi_mcp_server.py'])
libs_agent = Agent(
    'openai:gpt-4o',
    mcp_servers=[server],
    instructions='your job is to help the user research software libraries and packages using the tools provided',
)


@libs_agent.system_prompt
def add_date():
    return f'Today is {date.today():%Y-%m-%d}'


async def main():
    async with libs_agent.run_mcp_servers():
        result = await libs_agent.run('How many times has pydantic been downloaded this year')
    print(result.output)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
