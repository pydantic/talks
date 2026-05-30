import asyncio
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, UsageLimitExceeded, UsageLimits
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from typing_extensions import NotRequired

# Enable managed variables only when we have an API key to talk to Logfire.
variables_opts = logfire.VariablesOptions() if os.environ.get('LOGFIRE_API_KEY') else None

logfire.configure(console=False, variables=variables_opts)
logfire.instrument_pydantic_ai()


class Answer(StrEnum):
    yes = 'yes'
    kind_of = 'kind of'
    not_really = 'not really'
    no = 'no'
    complete_wrong = 'complete wrong'


answerer_agent = Agent(
    'anthropic:claude-haiku-4-5',
    deps_type=str,
    instructions="""
You are playing a question and answer game.
Your job is to answer questions about a secret object only you know truthfully.
""",
    output_type=Answer,
)


@answerer_agent.instructions
def add_answer(ctx: RunContext[str]) -> str:
    return f'THE SECRET OBJECT IS: "{ctx.deps}".'


@dataclass
class GameState:
    answer: str
    questions: int = 0


class GameResult(BaseModel, use_attribute_docstrings=True):
    answer: str
    """The exact object the other player is thinking of."""
    explanation: str
    """The explanation for the object the player is thinking of."""


questioner_prompt = logfire.var(
    'twenty_questions_prompt',
    default="""
You are playing a question and answer game. You need to guess what object the other player is thinking of.
Your job is to ask quantitative questions to narrow down the possibilities.

Start with broad questions (e.g., "Is it alive?", "Is it bigger than a breadbox?") and get more specific.
When you're confident, make a guess by saying "Is it [specific object]?"

You should ask strategic questions based on the previous answers.
""",
    description='System prompt for the 20 Questions questioner agent.',
)
# Agent that asks questions to guess the object
questioner_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    deps_type=GameState,
    output_type=GameResult,
)


@questioner_agent.tool
async def ask_question(ctx: RunContext[GameState], question: str) -> Answer:
    result = await answerer_agent.run(question, deps=ctx.deps.answer)
    ctx.deps.questions += 1
    print(f'{ctx.run_step:>2}: {question}: {result.output}')
    return result.output


class PlayResult(TypedDict):
    questions: float
    success: bool
    answer: NotRequired[str]
    explanation: NotRequired[str]


@dataclass
class QuestionCount(Evaluator[str, PlayResult]):
    async def evaluate(self, ctx: EvaluatorContext[str, PlayResult]) -> float:
        return ctx.output['questions']


@dataclass
class QnASuccess(Evaluator[str, PlayResult]):
    async def evaluate(self, ctx: EvaluatorContext[str, PlayResult]) -> bool:
        return ctx.output['success']


dataset: Dataset[str, PlayResult] = Dataset(
    cases=[
        Case(name='Potato', inputs='potato'),
        Case(name='Man', inputs='man'),
        Case(name='Woman', inputs='woman'),
        Case(name='Child', inputs='child'),
        Case(name='Bicycle', inputs='bicycle'),
        Case(name='House', inputs='house'),
        Case(name='Spoon', inputs='spoon'),
        Case(name='Toothbrush', inputs='toothbrush'),
        Case(name='Umbrella', inputs='umbrella'),
        Case(name='Pillow', inputs='pillow'),
        Case(name='Kettle', inputs='kettle'),
    ],
    evaluators=[QuestionCount(), QnASuccess()],
    name='20 Questions',
)


async def play(answer: str) -> PlayResult:
    state = GameState(answer=answer)
    with questioner_prompt.get() as resolve:
        try:
            result = await questioner_agent.run(
                'start',
                deps=state,
                usage_limits=UsageLimits(request_limit=50),
                instructions=resolve.value,
            )
        except UsageLimitExceeded:
            logfire.exception('usage limit exceeded')
            return {'questions': 50, 'success': False}
        else:
            print(f'Finished after {state.questions} questions: {result.output}')
            return {
                'questions': state.questions,
                'success': result.output.answer.strip().lower() == answer.strip().lower(),
                'answer': result.output.answer,
                'explanation': result.output.explanation,
            }


async def run_evals():
    report = await dataset.evaluate(play, name=f'Twenty Questions', repeat=3)
    report.print(include_input=False, include_output=False)


if __name__ == '__main__':
    asyncio.run(run_evals())
