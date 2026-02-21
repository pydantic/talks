from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Any

import logfire
import pydantic_core
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from main import get_prices, prices_agent
from sub_agent import ModelInfo

logfire.configure(environment='llm-prices-evals', console=False)
logfire.instrument_pydantic_ai()

reference_models: dict[str, ModelInfo] = {
    k: ModelInfo.model_validate(v)
    for k, v in pydantic_core.from_json(Path('anthropic_prices.json').read_bytes()).items()
}


class ModelCount(Evaluator[Any, dict[str, ModelInfo]]):
    """Number of models returned."""

    def evaluate(self, ctx: EvaluatorContext[Any, dict[str, ModelInfo]]) -> float:
        return float(len(ctx.output))


class ModelMatch(Evaluator[Any, dict[str, ModelInfo]]):
    """Count of model IDs present in both output and reference."""

    def evaluate(self, ctx: EvaluatorContext[Any, dict[str, ModelInfo]]) -> float:
        assert ctx.expected_output
        matching = set(ctx.output.keys()) & set(ctx.expected_output.keys())
        return float(len(matching))


class FieldCorrectness(Evaluator[Any, dict[str, ModelInfo]]):
    """Total correct fields across all ID-matched models (5 fields each)."""

    def evaluate(self, ctx: EvaluatorContext[Any, dict[str, ModelInfo]]) -> float:
        total = 0.0
        assert ctx.expected_output
        matching_ids = set(ctx.output.keys()) & set(ctx.expected_output.keys())
        for model_id in matching_ids:
            out = ctx.output[model_id]
            exp = ctx.expected_output[model_id]
            for field_name in ('name', 'description', 'input_mtok', 'output_mtok', 'attributes'):
                if getattr(out, field_name) == getattr(exp, field_name):
                    total += 1.0
        return total


dataset: Dataset[str, dict[str, ModelInfo]] = Dataset(
    cases=[
        Case(
            name='Anthropic',
            inputs='anthropic',
            expected_output=reference_models,
        ),
        Case(
            name='Groq',
            inputs='groq',
            expected_output=reference_models,
        ),
    ],
    evaluators=[ModelCount(), ModelMatch(), FieldCorrectness()],
    name='llm-api-prices',
)


async def run_eval(model: str):
    with prices_agent.override(model=model):
        report = await dataset.evaluate(get_prices, name=model)
        report.print(include_input=False, include_output=False)


async def run_evals():
    models = [
        'gateway/anthropic:claude-sonnet-4-6',
        'gateway/anthropic:claude-opus-4-6',
        'gateway/openai:gpt-5.2',
        'gateway/openai:gpt-5-mini',
    ]
    await asyncio.gather(*[run_eval(model) for model in models])


async def run_reuse_evals():
    report_false, report_true = await asyncio.gather(
        dataset.evaluate(partial(get_prices, allow_code_reuse=False), name='allow-reuse-false'),
        dataset.evaluate(partial(get_prices, allow_code_reuse=True), name='allow-reuse-true'),
    )
    report_false.print(include_input=False, include_output=False)
    report_true.print(include_input=False, include_output=False)


if __name__ == '__main__':
    # asyncio.run(run_evals())
    asyncio.run(run_reuse_evals())
