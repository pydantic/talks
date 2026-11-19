"""Offline evals: a small dataset of questions with known answers, run against the agent before a change ships.

Run with `make evals` (or `uv run -m ai_trade_example.evals_offline`). The report prints here and the
experiment is recorded in Logfire.
"""

import asyncio
import re
from dataclasses import dataclass

import logfire
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge

from .agent import build_agent
from .evals_online import JUDGE_MODEL

agent = build_agent()


async def answer(question: str) -> str:
    result = await agent.run(question)
    return result.output


@dataclass
class MentionsFigures(Evaluator[str, str, list[float]]):
    """Checks that every expected percentage (the case metadata) appears in the answer.

    Signs are ignored, since "fell 47%" and "-47%" are the same answer.
    """

    tolerance: float = 1.5

    def evaluate(self, ctx: EvaluatorContext[str, str, list[float]]) -> EvaluationReason:
        found = [float(n) for n in re.findall(r'(\d+(?:\.\d+)?)\s?%', ctx.output)]
        missing = [
            expected
            for expected in ctx.metadata or []
            if not any(abs(expected - value) <= self.tolerance for value in found)
        ]
        if missing:
            return EvaluationReason(value=False, reason=f'Missing {missing}, answer has {found}')
        return EvaluationReason(value=True)


dataset = Dataset[str, str, list[float]](
    name='ai_trade',
    cases=[
        Case(
            name='h1_return_vs_leverage',
            inputs=(
                'The press says the fund made 439% in the first half of 2026. Does the long book in its '
                'March filing, at the leverage the press reported, account for that?'
            ),
            expected_output='Yes. The March longs gained about 110.8% by 30 June, which at the reported 4x is about 443%.',
            metadata=[110.8, 443],
        ),
        Case(
            name='sandisk_vs_sk_hynix',
            inputs='Which hurt the fund more in July, Sandisk or SK Hynix?',
            expected_output='Sandisk: it fell about 47% in July, against about 30% for SK Hynix in USD terms.',
            metadata=[47, 30],
        ),
        Case(
            name='adobe_short',
            inputs='Was the Adobe short a hedge in July?',
            expected_output='No. Adobe rose about 22% in July while the long book fell about 33%.',
            metadata=[22, 33],
        ),
        Case(
            name='july_returns_by_theme',
            inputs='Chart July returns by AI theme.',
            expected_output='Neoclouds and memory fell 25 to 35% in July, while software rose about 21%.',
            metadata=[21],
        ),
    ],
    evaluators=[
        MentionsFigures(),
        LLMJudge(
            rubric='The answer reaches the same conclusion as the expected output.',
            model=JUDGE_MODEL,
            include_input=True,
            include_expected_output=True,
        ),
    ],
)


async def main() -> None:
    # console=False keeps the span log out of the terminal so the report is readable.
    logfire.configure(service_name='ai_trade_example', console=False)
    logfire.instrument_pydantic_ai()
    logfire.instrument_monty()

    report = await dataset.evaluate(answer, max_concurrency=4)
    report.print(include_input=False, include_output=False, include_reasons=True)


if __name__ == '__main__':
    asyncio.run(main())
