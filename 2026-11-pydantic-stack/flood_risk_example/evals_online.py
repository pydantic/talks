"""Online evals: evaluators that run in this process after each agent run, on live traffic.

There is no expected answer to compare against here, so the checks are about how the agent behaved.
"""

import os
import re
from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge
from pydantic_evals.online import OnlineEvalConfig
from pydantic_evals.online_capability import OnlineEvaluation

# A decision model returns a pass or fail with no written reason, and is cheap enough to judge every run.
JUDGE_MODEL = os.getenv('JUDGE_MODEL', 'typesafe:jev-latest')


@dataclass
class RanCode(Evaluator):
    """Passes if the agent ran code at least once, and at most `max_runs` times."""

    max_runs: int = 4

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        runs = ctx.span_tree.find({'has_attributes': {'gen_ai.tool.name': 'run_code'}})
        if not runs:
            return EvaluationReason(value=False, reason='Answered without running any code')
        if len(runs) > self.max_runs:
            return EvaluationReason(value=False, reason=f'{len(runs)} code runs, expected at most {self.max_runs}')
        return EvaluationReason(value=True, reason=f'code runs: {len(runs)}')


@dataclass
class DecisionsFromTool(Evaluator):
    """Passes unless the answer gives an underwriting decision that no `decide` call backs up."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        decisions = ctx.span_tree.find({'has_attributes': {'gen_ai.tool.name': 'decide'}})
        gives_decision = re.search(r'\b(accept|refer|decline)(ed|red|s|d)?\b', str(ctx.output), re.IGNORECASE)
        if gives_decision and not decisions:
            return EvaluationReason(value=False, reason='The answer gives a decision but `decide` was never called')
        return EvaluationReason(value=True, reason=f'decide calls: {len(decisions)}')


def online_evaluation() -> OnlineEvaluation:
    return OnlineEvaluation(
        evaluators=[
            LLMJudge(
                rubric='The answer addresses the question that was asked and gives specific properties or counts.',
                model=JUDGE_MODEL,
                include_input=True,
            ),
            RanCode(),
            DecisionsFromTool(),
        ],
        config=OnlineEvalConfig(default_sample_rate=1.0),
    )
