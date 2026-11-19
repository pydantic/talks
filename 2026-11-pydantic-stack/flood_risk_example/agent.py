from pathlib import Path

import logfire
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode

from .evals_online import online_evaluation
from .tools import TOOLS, Deps

INSTRUCTIONS = (Path(__file__).parent / 'instructions.md').read_text()


def build_agent(*, sandbox: bool = False, online_evals: bool = False) -> Agent[Deps, str]:
    """Build the underwriter's assistant.

    Args:
        sandbox: Run the model's code in Logfire Agent Sandboxes instead of a local Monty process.
        online_evals: Evaluate every run in the background and attach the results to its trace.
    """
    code_mode = CodeMode(
        monty_sandbox=logfire.agent_sandboxes if sandbox else None,
        # A portfolio question is over a thousand tool calls from one run of code.
        max_tool_calls=5000,
    )
    return Agent(
        'gateway/anthropic:claude-sonnet-5-5',
        name='flood_risk_agent',
        deps_type=Deps,
        instructions=INSTRUCTIONS,
        retries=3,
        tools=TOOLS,
        capabilities=[code_mode, online_evaluation()] if online_evals else [code_mode],
    )
