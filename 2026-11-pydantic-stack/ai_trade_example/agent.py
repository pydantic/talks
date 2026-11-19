from pathlib import Path

import logfire
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode
from pydantic_monty import MountDir

from .charts import draw_chart
from .database import Database
from .evals_online import online_evaluation

db = Database()

THIS_DIR = Path(__file__).parent
INSTRUCTIONS = (THIS_DIR / 'instructions.md').read_text()


def build_agent(*, sandbox: bool = False, online_evals: bool = False) -> Agent[None, str]:
    """Build the AI trade agent.

    Args:
        sandbox: Run the model's code in Logfire Agent Sandboxes instead of a local Monty process.
        online_evals: Evaluate every run in the background and attach the results to its trace.
    """
    code_mode = CodeMode(
        # Background press coverage the agent can read with pathlib; see docs/README.md.
        mount=MountDir(host_path=THIS_DIR / 'docs', virtual_path='/docs', mode='read-only'),
        monty_sandbox=logfire.agent_sandboxes if sandbox else None,
    )
    return Agent(
        'gateway/openai:gpt-5.6-terra',
        name='monty_ai_trade_agent',
        instructions=INSTRUCTIONS,
        retries=3,
        tools=[
            db.query,
            db.list_tables,
            db.describe_table,
            db.table_count,
            draw_chart,
        ],
        model_settings={'max_tokens': 16384},
        capabilities=[code_mode, online_evaluation()] if online_evals else [code_mode],
    )
