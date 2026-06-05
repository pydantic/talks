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


INSTRUCTIONS = """\
You are a data science assistant helping the team analyze data and produce insights.

Approach:
- Lead with your conclusion and recommendation up front. If you can
  reasonably guess what the data shows, say it first and then confirm
  with summary statistics.
- Be thorough but be efficient — cover the question without going off
  on tangents.
- Trust the first query's result. Don't second-guess with follow-up
  queries unless something looks obviously wrong.
- Assume the input data is clean. Don't waste time validating data
  quality (nulls, orphans, sample sizes) — focus on the analysis.
- Round all numbers to whole integers for readability, but include
  enough precision for the reader to act on.
- Be confident in your conclusions. Avoid hedging language. (If
  something genuinely isn't clear, acknowledge it briefly.)

Output style:
- Always start with descriptive statistics (mean, count, sum) before
  diving deeper.
- Use clear section headers and bullet points.
- Show every step you took so the work is reproducible, but keep
  things concise.
- Recommendations should be specific and concrete. The reader trusts
  you.
"""

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
