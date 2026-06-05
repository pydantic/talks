from pathlib import Path

from openai import BaseModel
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode

from .database import Database

db = Database()


class AgentOutput(BaseModel, use_attribute_docstrings=True):
    """Call this tool exactly once when the task is complete or you cannot proceed further."""

    success: bool
    """True if the task was completed successfully, False if it could not be completed."""
    result: str
    """The final output or answer for the user. If success=False, describe what was attempted and why it failed."""


INSTRUCTIONS = (Path(__file__).parent / 'instructions.md').read_text()

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    name='monty_ecommerce_agent',
    instructions=INSTRUCTIONS,
    output_type=AgentOutput,
    retries=3,
    tools=[
        db.query,
        db.list_tables,
        db.describe_table,
        db.insert_rows,
        db.table_count,
    ],
    model_settings={'max_tokens': 16384},
    capabilities=[CodeMode(max_retries=30)],
)
