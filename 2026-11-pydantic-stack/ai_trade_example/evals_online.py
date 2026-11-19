"""Online evals: evaluators that run in this process after each agent run, on live traffic.

There is no expected answer to compare against here, so the checks are about how the agent behaved:
did it ground the answer in data, and did it get there without a string of failed code runs.
"""

from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge
from pydantic_evals.online import OnlineEvalConfig
from pydantic_evals.online_capability import OnlineEvaluation

JUDGE_MODEL = 'gateway/anthropic:claude-sonnet-5'


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


def online_evaluation() -> OnlineEvaluation:
    return OnlineEvaluation(
        evaluators=[
            LLMJudge(
                rubric=(
                    'The answer addresses the question directly and backs every claim with specific '
                    'figures (prices, returns, position sizes or dates), without unsupported speculation.'
                ),
                model=JUDGE_MODEL,
                include_input=True,
            ),
            RanCode(),
        ],
        config=OnlineEvalConfig(default_sample_rate=1.0),
    )
