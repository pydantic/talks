"""Offline evals: datasets with known answers, run before a change ships.

Two datasets, both built from the geo risk service, so `make serve` must be running:

- `decisions`: 60 borderline properties whose correct outcome follows from the underwriting guide, run through the
  decision agent on the decision model and again on a language model, to compare the two.
- `portfolio`: questions about the whole portfolio with answers computed here from the same data, run
  through the main agent.

Run with `make evals` (decisions) or `make evals-portfolio`. Reports print here and each run is an
experiment in Logfire.
"""

import asyncio
import re
import sys
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx2
import logfire
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge
from rich.console import Console
from rich.table import Table

from . import decision
from .agent import agent
from .policy import Facts, OutcomeName, expected_flood_excess, expected_outcome, margin
from .tools import Deps, http_client

CASES_PER_OUTCOME = 20
TASK_TIMEOUT_S = 120


class Expected(BaseModel):
    outcome: OutcomeName
    flood_excess: bool


async def all_facts(http: httpx2.AsyncClient, buffer_m: float = 0) -> list[Facts]:
    """The facts for every property, straight from the geo risk service."""

    async def get(path: str, **params: Any) -> Any:
        response = await http.get(path, params=params)
        response.raise_for_status()
        return response.json()

    async def facts(prop: dict[str, Any]) -> Facts:
        base = f'/properties/{prop["id"]}'
        zone, elevation, water = await asyncio.gather(
            get(f'{base}/flood-zone', buffer_m=buffer_m), get(f'{base}/elevation'), get(f'{base}/distance-to-water')
        )
        return Facts(
            property_id=prop['id'],
            address=prop['address'],
            kind=prop['kind'],
            units=prop['units'],
            zone=zone['zone'],
            distance_to_1_percent_zone_m=zone['distance_to_1_percent_zone_m'],
            elevation_m=elevation['elevation_m'],
            water_elevation_m=elevation['water_elevation_m'],
            distance_to_water_m=water['distance_m'],
            water_name=water['water_name'],
        )

    return await asyncio.gather(*[facts(p) for p in await get('/properties')])


# Decisions


@dataclass
class OutcomeCorrect(Evaluator[Facts, decision.DecisionResult, None]):
    def evaluate(self, ctx: EvaluatorContext[Facts, decision.DecisionResult, None]) -> EvaluationReason:
        assert isinstance(ctx.expected_output, Expected)
        expected, actual = ctx.expected_output.outcome, ctx.output.outcome.value
        return EvaluationReason(
            value=actual == expected, reason=None if actual == expected else f'{actual}, not {expected}'
        )


@dataclass
class FloodExcessCorrect(Evaluator[Facts, decision.DecisionResult, None]):
    def evaluate(self, ctx: EvaluatorContext[Facts, decision.DecisionResult, None]) -> bool:
        assert isinstance(ctx.expected_output, Expected)
        return ctx.output.flood_excess == ctx.expected_output.flood_excess


@dataclass
class DecidedBy(Evaluator[Facts, decision.DecisionResult, None]):
    """Records which model answered, which shows how often the decision model handed off."""

    def evaluate(self, ctx: EvaluatorContext[Facts, decision.DecisionResult, None]) -> dict[str, str]:
        return {'decided_by': ctx.output.decided_by}


def decisions_dataset(facts: list[Facts]) -> Dataset[Facts, Any, None]:
    cases: list[Case[Facts, Any, None]] = []
    for outcome in ('accept', 'refer', 'decline'):
        # The cases closest to one of the guide's thresholds: the ones a careless reader gets wrong.
        matching = sorted(
            (f for f in facts if expected_outcome(f) == outcome), key=lambda f: (margin(f), f.property_id)
        )
        for f in sorted(matching[:CASES_PER_OUTCOME], key=lambda f: f.property_id):
            expected = Expected(outcome=outcome, flood_excess=expected_flood_excess(f))
            cases.append(Case(name=f'{f.property_id} {f.address}', inputs=f, expected_output=expected))
    return Dataset(
        name='flood_decisions', cases=cases, evaluators=[OutcomeCorrect(), FloodExcessCorrect(), DecidedBy()]
    )


async def evaluate_decisions(http: httpx2.AsyncClient, model_names: list[str]) -> None:
    """Run the decisions dataset once per model, then print the models side by side."""
    dataset = decisions_dataset(await all_facts(http))
    summary = Table(title='Decisions by model')
    for column in 'Model', 'Outcome', 'Flood excess', 'Decided by', 'Avg time', 'Total cost':
        summary.add_column(column, justify='left' if column in ('Model', 'Decided by') else 'right')
    for model_name in model_names:
        # The timeout stops one hung model request from holding up the whole run.
        report = await dataset.evaluate(
            decide_with(model_name), name=model_name, max_concurrency=10, task_timeout=TASK_TIMEOUT_S
        )
        report.print(include_input=False, include_output=False, include_reasons=True)
        cases, total = report.cases, len(report.cases) + len(report.failures)
        decided_by = Counter(case.labels['decided_by'].value for case in cases)
        summary.add_row(
            model_name,
            f'{sum(case.assertions["OutcomeCorrect"].value for case in cases)}/{total}',
            f'{sum(case.assertions["FloodExcessCorrect"].value for case in cases)}/{total}',
            ', '.join(f'{name} {n}' for name, n in decided_by.most_common()),
            f'{sum(case.task_duration for case in cases) / max(len(cases), 1) * 1000:.0f} ms',
            f'${sum(case.metrics.get("cost", 0) for case in cases):.4f}',
        )
    Console().print(summary)


def decide_with(model_name: str) -> Callable[[Facts], Awaitable[decision.DecisionResult]]:
    model = decision.decision_model(model_name)

    async def task(facts: Facts) -> decision.DecisionResult:
        return await decision.decide(facts, model)

    return task


# Portfolio questions


@dataclass
class MentionsNumbers(Evaluator[str, str, list[float]]):
    """Checks that every expected number (the case metadata) appears in the answer."""

    tolerance: float = 0.06

    def evaluate(self, ctx: EvaluatorContext[str, str, list[float]]) -> EvaluationReason:
        found = [float(n.replace(',', '')) for n in re.findall(r'\d[\d,]*(?:\.\d+)?', ctx.output)]
        missing = [n for n in ctx.metadata or [] if not any(abs(n - value) <= self.tolerance for value in found)]
        return EvaluationReason(value=not missing, reason=f'Missing {missing}' if missing else None)


async def portfolio_dataset(http: httpx2.AsyncClient) -> Dataset[str, str, list[float]]:
    now, grown = await asyncio.gather(all_facts(http), all_facts(http, buffer_m=50))
    high_risk = ('floodway', '1_percent')
    in_one_percent = sum(f.zone == '1_percent' for f in now)
    added = sum(f.zone in high_risk for f in grown) - sum(f.zone in high_risk for f in now)
    houses = [f.height_above_water_m for f in now if f.kind == 'house']
    house_height = round(sum(houses) / len(houses), 1)
    closest = min(now, key=lambda f: f.distance_to_water_m)
    return Dataset(
        name='flood_portfolio',
        cases=[
            Case(
                name='count_in_1_percent_zone',
                inputs='How many properties are in the 1% annual chance flood zone, not counting the floodway?',
                expected_output=f'{in_one_percent} properties.',
                metadata=[in_one_percent],
            ),
            Case(
                name='zones_grow_50m',
                inputs='How many more properties would be in the floodway or the 1% zone if the flood zones grew by 50 m?',
                expected_output=f'{added} more properties.',
                metadata=[added],
            ),
            Case(
                name='average_house_height',
                inputs='What is the average height above the nearest water of the houses, to one decimal place?',
                expected_output=f'{house_height} m.',
                metadata=[house_height],
            ),
            Case(
                name='closest_to_water',
                inputs='Which property is closest to water?',
                expected_output=f'{closest.address} ({closest.property_id}), {closest.distance_to_water_m:.0f} m away.',
                metadata=[],
            ),
        ],
        evaluators=[
            MentionsNumbers(),
            LLMJudge(
                rubric='The answer gives the same property or number as the expected output.',
                model=decision.LANGUAGE_MODEL,
                include_input=True,
                include_expected_output=True,
            ),
        ],
    )


async def evaluate_portfolio(http: httpx2.AsyncClient) -> None:
    dataset = await portfolio_dataset(http)

    async def task(question: str) -> str:
        result = await agent.run(question, deps=Deps(http=http))
        return result.output

    report = await dataset.evaluate(task, max_concurrency=4)
    report.print(include_input=False, include_output=False, include_reasons=True)


async def main() -> None:
    # console=False keeps the span log out of the terminal so the reports are readable.
    logfire.configure(service_name='flood_risk_example', console=False)
    logfire.instrument_pydantic_ai()

    which = sys.argv[1] if len(sys.argv) > 1 else 'decisions'
    async with http_client() as http:
        if which in ('decisions', 'all'):
            # Models to compare can be given on the command line, e.g. `make evals MODELS="..."`.
            models = sys.argv[2:] or list(dict.fromkeys([decision.DECISION_MODEL, decision.LANGUAGE_MODEL]))
            await evaluate_decisions(http, models)
        if which in ('portfolio', 'all'):
            await evaluate_portfolio(http)


if __name__ == '__main__':
    asyncio.run(main())
