import asyncio
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

logfire.configure(console=False)
logfire.instrument_pydantic_ai()

secret = "potato"


class Answer(StrEnum):
    yes = "yes"
    kind_of = "kind of"
    not_really = "not really"
    no = "no"
    complete_wrong = "complete wrong"


answerer_agent = Agent(
    "anthropic:claude-haiku-4-5",
    instructions=f"""
You are playing a question and answer game.
Your job is to answer yes/no questions about the secret object truthfully.
ALWAYS answer with only true for 'yes' and false for 'no'.

THE SECRET OBJECT IS: {secret}.
""",
    output_type=Answer,
    name="answerer_agent",
)
temporal_answerer_agent = TemporalAgent(answerer_agent)


# Agent that asks questions to guess the object
questioner_agent = Agent(
    "anthropic:claude-sonnet-4-5",
    instructions="""
You are playing a question and answer game. You need to guess what object the other player is thinking of.
Your job is to ask yes/no questions to narrow down the possibilities.

Start with broad questions (e.g., "Is it alive?", "Is it bigger than a breadbox?") and get more specific.
When you're confident, make a guess by saying "Is it [specific object]?"

You should ask strategic questions based on the previous answers.
""",
    name="questioner_agent",
)


@questioner_agent.tool
async def ask_question(ctx: RunContext, question: str) -> Answer:
    if random() > 0.9:
        raise RuntimeError("broken")
    print(f"{ctx.run_step:>2}: {question}:", end=" ", flush=True)
    result = await temporal_answerer_agent.run(question)
    print(result.output)
    return result.output


temporal_questioner_agent = TemporalAgent(questioner_agent)


@workflow.defn
class TwentyQuestionsWorkflow:
    @workflow.run
    async def run(self) -> None:
        result = await temporal_questioner_agent.run("start")
        print(f"After {len(result.all_messages()) / 2}, the answer is: {result.output}")


async def play(resume_id: str | None):
    client = await Client.connect(
        "localhost:7233", plugins=[PydanticAIPlugin(), LogfirePlugin()]
    )

    async with Worker(
        client,
        task_queue="twenty_questions",
        workflows=[TwentyQuestionsWorkflow],
        plugins=[
            AgentPlugin(temporal_answerer_agent),
            AgentPlugin(temporal_questioner_agent),
        ],
    ):
        if resume_id is not None:
            print("resuming existing workflow", resume_id)
            await client.get_workflow_handle(resume_id).result()  # type: ignore[ReportUnknownMemberType]
        else:
            await client.execute_workflow(  # type: ignore[ReportUnknownMemberType]
                TwentyQuestionsWorkflow.run,
                id=f"twenty_questions-{uuid.uuid4()}",
                task_queue="twenty_questions",
            )


if __name__ == "__main__":
    try:
        asyncio.run(play(sys.argv[1] if len(sys.argv) > 1 else None))
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(1)
