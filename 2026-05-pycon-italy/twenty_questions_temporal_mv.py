"""twenty_questions_temporal_mv.py

Same as twenty_questions_temporal.py but the questioner agent's system prompt
is driven by a Logfire managed variable so it can be optimized via the
Logfire Prompt Optimizer.

Key differences from the original:
- logfire.configure() gets variables=VariablesOptions() for managed-variable support
- LOGFIRE_TOKEN is mapped to LOGFIRE_API_KEY (same v2 token works for both)
- The questioner system prompt is stored in a Logfire managed variable
- play() resolves the variable with `with _questioner_var.get() as resolved:`
  and passes the resolved value to the workflow, which forwards it to
  TemporalAgent.run(..., instructions=...) for a clean per-run override

Usage:
    Set LOGFIRE_TOKEN (and optionally LOGFIRE_VAR_QUESTIONER_PROMPT) in your
    environment, then run exactly as the original:

        uv run python twenty_questions_temporal_mv.py
        uv run python twenty_questions_temporal_mv.py <resume-workflow-id>
"""

import asyncio
import os
import sys
import uuid
from enum import StrEnum
from random import random

import logfire
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import (
    AgentPlugin,
    LogfirePlugin,
    PydanticAIPlugin,
    TemporalAgent,
)
from pydantic_ai.tools import RunContext
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

# Logfire uses LOGFIRE_TOKEN for trace export and LOGFIRE_API_KEY for the
# variables REST API. The same v2 token works for both — map it here so
# VariablesOptions() can resolve variable values from the Logfire API.
_logfire_token = os.environ.get('LOGFIRE_TOKEN')
if _logfire_token:
    os.environ.setdefault('LOGFIRE_API_KEY', _logfire_token)

# Enable managed variables only when we have an API key to talk to Logfire.
_variables_opts = logfire.VariablesOptions() if os.environ.get('LOGFIRE_API_KEY') else None

logfire.configure(console=False, variables=_variables_opts)
logfire.instrument_pydantic_ai()

# ---------------------------------------------------------------------------
# Secret & managed variable name
# ---------------------------------------------------------------------------

secret = 'potato'

# The managed variable name can be overridden via env var so different
# environments / demo runs can use different variable namespaces.
QUESTIONER_VAR_NAME = os.environ.get(
    'LOGFIRE_VAR_QUESTIONER_PROMPT',
    'questioner_prompt_twenty_questions_temporal',
)

DEFAULT_QUESTIONER_INSTRUCTIONS = """
You are playing a question and answer game. You need to guess what object the other player is thinking of.
Your job is to ask yes/no questions to narrow down the possibilities.

Start with broad questions (e.g., "Is it alive?", "Is it bigger than a breadbox?") and get more specific.
When you're confident, make a guess by saying "Is it [specific object]?"

You should ask strategic questions based on the previous answers.
"""

# Module-level handle — created once; the value is resolved lazily via .get()
# each time play() is called. This allows prompt changes to take effect without
# restarting the process.
#
# Guard against double-registration: Temporal's workflow sandbox re-imports this
# module in the same process (sharing the global logfire state), which would
# cause logfire.var() to raise "already registered". Retrieve the existing
# handle instead.
try:
    _questioner_var = logfire.var(
        QUESTIONER_VAR_NAME,
        default=DEFAULT_QUESTIONER_INSTRUCTIONS,
        description='System prompt for the 20 Questions questioner agent.',
    )
except ValueError:
    _questioner_var = next(v for v in logfire.variables_get() if v.name == QUESTIONER_VAR_NAME)

# ---------------------------------------------------------------------------
# Answerer agent — unchanged from the original
# ---------------------------------------------------------------------------


class Answer(StrEnum):
    yes = 'yes'
    kind_of = 'kind of'
    not_really = 'not really'
    no = 'no'
    complete_wrong = 'complete wrong'


answerer_agent = Agent(
    'anthropic:claude-haiku-4-5',
    instructions=f"""
You are playing a question and answer game.
Your job is to answer questions about a secret object only you know truthfully.

THE SECRET OBJECT IS: {secret}.
""",
    output_type=Answer,
    name='answerer_agent',
)
temporal_answerer_agent = TemporalAgent(answerer_agent)

# ---------------------------------------------------------------------------
# Questioner agent — module-level TemporalAgent with default instructions.
# The actual instructions are passed per-run via TemporalAgent.run(instructions=)
# so they can be overridden with the resolved managed-variable value at runtime.
# ---------------------------------------------------------------------------

questioner_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    # No default instructions here — the system prompt comes entirely from the
    # managed variable value, passed per-run via TemporalAgent.run(instructions=)
    name='questioner_agent',
)


@questioner_agent.tool
async def ask_question(ctx: RunContext, question: str) -> Answer:
    if random() > 0.9:
        raise RuntimeError('broken')
    print(f'{ctx.run_step:>2}: {question}:', end=' ', flush=True)
    result = await temporal_answerer_agent.run(question)
    print(result.output)
    return result.output


temporal_questioner_agent = TemporalAgent(questioner_agent)

# ---------------------------------------------------------------------------
# Temporal workflow
# ---------------------------------------------------------------------------


@workflow.defn
class TwentyQuestionsWorkflow:
    @workflow.run
    async def run(self, questioner_prompt: str) -> None:
        """Run one full game.

        *questioner_prompt* is the resolved managed-variable value, forwarded
        here from play() so the workflow doesn't need to do any I/O itself.
        TemporalAgent.run(instructions=...) overrides the agent's instructions
        for this specific run without touching module-level state.
        """
        result = await temporal_questioner_agent.run('start', instructions=questioner_prompt)
        print(f'After {len(result.all_messages()) / 2}, the answer is: {result.output}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def play(resume_id: str | None) -> None:
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin(), LogfirePlugin()])

    async with Worker(
        client,
        task_queue='twenty_questions',
        workflows=[TwentyQuestionsWorkflow],
        plugins=[
            AgentPlugin(temporal_answerer_agent),
            AgentPlugin(temporal_questioner_agent),
        ],
    ):
        # Resolve the managed variable and wrap the entire workflow execution
        # inside the context manager. This records which variable version (and
        # therefore which prompt) was used, enabling the Logfire Optimizer to
        # correlate prompt versions with game outcomes.
        with _questioner_var.get() as resolved:
            if resume_id is not None:
                print('resuming existing workflow', resume_id)
                await client.get_workflow_handle(resume_id).result()
            else:
                job_id = f'twenty_questions-{uuid.uuid4()}'
                print(f'{job_id=}')
                await client.execute_workflow(
                    TwentyQuestionsWorkflow.run,
                    resolved.value,  # pass managed-var value as workflow input
                    id=job_id,
                    task_queue='twenty_questions',
                )


if __name__ == '__main__':
    try:
        asyncio.run(play(sys.argv[1] if len(sys.argv) > 1 else None))
    except KeyboardInterrupt:
        print('Exiting...')
        sys.exit(1)
