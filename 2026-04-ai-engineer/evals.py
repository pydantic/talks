"""Evaluation dataset and evaluators for political relations extraction.

This module defines:
- RelationsCaseMetadata: Metadata for each test case
- RelationsAccuracyEvaluator: Evaluates extraction accuracy
- relations_dataset: The evaluation dataset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from task import MP, PoliticalRelation, TaskInput


@dataclass
class RelationsCaseMetadata:
    """Metadata for political relations extraction test cases."""

    difficulty: str  # 'easy', 'medium', 'hard'
    description: str


@dataclass
class RelationsAccuracyEvaluator(Evaluator[TaskInput, list[PoliticalRelation], RelationsCaseMetadata]):
    """Evaluates how accurately political relations were extracted.

    Compares extracted relations against expected ones by matching on name.
    Returns a score between 0.0 and 1.0.
    """

    def evaluate(
        self, ctx: EvaluatorContext[TaskInput, list[PoliticalRelation], RelationsCaseMetadata]
    ) -> dict[str, Any]:
        if ctx.expected_output is None:
            return {'accuracy': 1.0}

        expected = ctx.expected_output
        output = ctx.output

        if not expected and not output:
            return {'accuracy': 1.0, 'expected_count': 0, 'output_count': 0, 'matched': 0}

        if not expected:
            # Expected no relations but got some — penalise
            return {'accuracy': 0.0, 'expected_count': 0, 'output_count': len(output), 'matched': 0}

        matched = 0
        for exp in expected:
            exp_name = exp.name.lower().strip()
            for out in output:
                if exp_name in out.name.lower().strip() or out.name.lower().strip() in exp_name:
                    matched += 1
                    break

        # Score: fraction of expected relations found, penalised by false positives
        precision = matched / len(output) if output else 0.0
        recall = matched / len(expected)
        if precision + recall == 0:
            accuracy = 0.0
        else:
            accuracy = 2 * precision * recall / (precision + recall)  # F1

        return {
            'accuracy': accuracy,
            'expected_count': len(expected),
            'output_count': len(output),
            'matched': matched,
        }


def _mp(id: int, name: str, party: str) -> MP:
    return MP(id=id, name=name, url='', raw_party=party)


relations_cases: list[Case[TaskInput, list[PoliticalRelation], RelationsCaseMetadata]] = [
    # MP with multiple well-known political relations
    Case(
        name='stephen_kinnock',
        inputs=TaskInput(mp=_mp(2, 'Stephen Kinnock', 'Labour')),
        expected_output=[
            PoliticalRelation(
                name='Neil Kinnock',
                role='Former Leader of the Labour Party; Member of Parliament',
                relation='father',
                party='Labour',
            ),
            PoliticalRelation(
                name='Glenys Kinnock',
                role='Member of the European Parliament',
                relation='mother',
                party='Labour',
            ),
            PoliticalRelation(
                name='Helle Thorning-Schmidt',
                role='Prime Minister of Denmark',
                relation='wife',
            ),
        ],
        metadata=RelationsCaseMetadata(
            difficulty='easy',
            description='MP with multiple prominent political family members',
        ),
    ),
    # MP with one relation
    Case(
        name='kirsty_blackman',
        inputs=TaskInput(mp=_mp(3, 'Kirsty Blackman', 'Scottish National Party')),
        expected_output=[
            PoliticalRelation(
                name='John West',
                role='Aberdeen City Councillor',
                relation='brother',
            ),
        ],
        metadata=RelationsCaseMetadata(
            difficulty='medium',
            description='MP with one less prominent political relation',
        ),
    ),
    # MP with one relation
    Case(
        name='stephen_flynn',
        inputs=TaskInput(mp=_mp(4, 'Stephen Flynn', 'Scottish National Party')),
        expected_output=[
            PoliticalRelation(
                name='Mark Flynn',
                role='Leader of Dundee City Council',
                relation='father',
            ),
        ],
        metadata=RelationsCaseMetadata(
            difficulty='medium',
            description='MP whose father is a council leader',
        ),
    ),
    # MPs with no political relations
    Case(
        name='seamus_logan',
        inputs=TaskInput(mp=_mp(5, 'Seamus Logan', 'Scottish National Party')),
        expected_output=[],
        metadata=RelationsCaseMetadata(
            difficulty='easy',
            description='MP with no political relations',
        ),
    ),
    Case(
        name='alex_baker',
        inputs=TaskInput(mp=_mp(7, 'Alex Baker', 'Labour')),
        expected_output=[],
        metadata=RelationsCaseMetadata(
            difficulty='easy',
            description='MP with no political relations',
        ),
    ),
    Case(
        name='wendy_morton',
        inputs=TaskInput(mp=_mp(8, 'Wendy Morton', 'Conservative')),
        expected_output=[],
        metadata=RelationsCaseMetadata(
            difficulty='easy',
            description='Conservative MP with no political relations',
        ),
    ),
    Case(
        name='connor_rand',
        inputs=TaskInput(mp=_mp(10, 'Connor Rand', 'Labour')),
        expected_output=[],
        metadata=RelationsCaseMetadata(
            difficulty='easy',
            description='MP with no political relations',
        ),
    ),
    Case(
        name='mark_tami',
        inputs=TaskInput(mp=_mp(11, 'Mark Tami', 'Labour')),
        expected_output=[],
        metadata=RelationsCaseMetadata(
            difficulty='easy',
            description='MP with no political relations',
        ),
    ),
]

relations_dataset: Dataset[TaskInput, list[PoliticalRelation], RelationsCaseMetadata] = Dataset(
    name='political_relations_extraction',
    cases=relations_cases,
    evaluators=[RelationsAccuracyEvaluator()],
)
