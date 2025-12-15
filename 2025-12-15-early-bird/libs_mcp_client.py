from datetime import date

import logfire
from mcp.types import LoggingMessageNotificationParams
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

logfire.configure(service_name='mcp-client', console=False)

logfire.instrument_pydantic_ai()
logfire.instrument_mcp()


async def log_handler(params: LoggingMessageNotificationParams):
    print(f'{params.level}: {params.data}')


server = MCPServerStdio(command='uv', args=['run', 'pypi_mcp_server.py'], log_handler=log_handler)
libs_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-5',
    toolsets=[server],
    instructions='your job is to help the user research software libraries and packages using the tools provided',
)
libs_agent.set_mcp_sampling_model()


@libs_agent.instructions
def add_date():
    return f"Today's date is {date.today():%Y-%m-%d}"


async def main():
    async with libs_agent:
        result = await libs_agent.run('How many times has pydantic been downloaded this year')
    print(result.output)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
