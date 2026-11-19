import asyncio
import os
import sys
from pathlib import Path

import logfire
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode
from pydantic_evals.online import wait_for_evaluations

from .evals_online import online_evaluation
from .policy import GUIDE
from .tools import TOOLS, Deps, http_client

# Evaluate every run in the background and attach the results to its trace.
ONLINE_EVALS = os.getenv('FLOOD_ONLINE_EVALS') == '1'

code_mode = CodeMode(
    # The model's code runs in Logfire Agent Sandboxes, not on this machine.
    monty_sandbox=logfire.agent_sandboxes,
    # A portfolio question is over a thousand tool calls from one run of code.
    max_tool_calls=5000,
)

agent = Agent(
    'gateway/anthropic:claude-sonnet-5-5',
    name='flood_risk_agent',
    deps_type=Deps,
    instructions=(Path(__file__).parent / 'instructions.md').read_text().replace('{guide}', GUIDE),
    retries=3,
    tools=TOOLS,
    capabilities=[code_mode, online_evaluation()] if ONLINE_EVALS else [code_mode],
)


async def main(question: str) -> None:
    logfire.configure(service_name='flood_risk_example')
    logfire.instrument_pydantic_ai()

    async with http_client() as http:
        result = await agent.run(question, deps=Deps(http=http))
    print(result.output)
    # Online evaluators run in the background, so let any still in flight finish before exiting.
    await wait_for_evaluations()


if __name__ == '__main__':
    # `make agent`, or `make agent Q="Which street has the most referrals?"`
    asyncio.run(main(' '.join(sys.argv[1:]) or 'Should we insure 34 State St?'))
