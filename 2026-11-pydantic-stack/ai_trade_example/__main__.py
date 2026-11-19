import argparse
import asyncio

import logfire
from pydantic_evals.online import wait_for_evaluations

from .agent import build_agent
from .repl import run_repl

parser = argparse.ArgumentParser(prog='ai-trade')
parser.add_argument('--sandbox', action='store_true', help='run code in Logfire Agent Sandboxes')
parser.add_argument('--online-evals', action='store_true', help='evaluate every run in the background')
args = parser.parse_args()

logfire.configure(service_name='ai_trade_example')
logfire.instrument_pydantic_ai()
logfire.instrument_monty()


async def main() -> None:
    agent = build_agent(sandbox=args.sandbox, online_evals=args.online_evals)
    await run_repl(agent, prog_name='ai-trade')
    # Evaluators run in the background, so let any still in flight finish before exiting.
    await wait_for_evaluations()


asyncio.run(main())
