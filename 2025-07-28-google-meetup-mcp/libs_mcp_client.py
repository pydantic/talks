import os
from datetime import date

import logfire
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP

logfire.configure(service_name='mcp-client')

logfire.instrument_pydantic_ai()
logfire.instrument_mcp()


pypi_server = MCPServerStdio(
    command='uv',
    args=['run', 'pypi_mcp_server.py'],
    env={'LOGFIRE_CONSOLE_SHOW_PROJECT_LINK': 'false'},
)
github_mcp_pat = os.environ['GITHUB_MCP_PAT']
github_server = MCPServerStreamableHTTP(
    url='https://api.githubcopilot.com/mcp/',
    headers={'authorization': f'Bearer {github_mcp_pat}'},
)
libs_agent = Agent(
    'anthropic:claude-sonnet-4-0',
    toolsets=[pypi_server, github_server],
    instructions='your job is to help the user research software libraries and packages using the tools provided. Reply concisely.',
)
# set the model to use on the mcp connection
libs_agent.set_mcp_sampling_model()


@libs_agent.instructions
def add_date():
    return f'Today is {date.today():%Y-%m-%d}'


async def main():
    async with libs_agent:
        result = await libs_agent.run('How many times has pydantic been downloaded this year?')
    print(result.output)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
