from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.code_mode import CodeModeMount
from pydantic_monty import MountDir

from .database import Database

db = Database()

# Persona/style notes followed by the full schema reference (one self-contained file).
THIS_DIR = Path(__file__).parent
INSTRUCTIONS = (THIS_DIR / 'instructions.md').read_text()

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    name='monty_world_cup_agent',
    instructions=INSTRUCTIONS,
    retries=3,
    tools=[
        db.query,
        db.list_tables,
        db.describe_table,
        db.table_count,
    ],
    model_settings={'max_tokens': 16384},
    # capabilities=[
    #     CodeMode(
    #         max_retries=30,
    #         mount=MountDir('/output', (THIS_DIR / '..' / 'agent_output').resolve(), mode='read-write'),
    #     )
    # ],
)
