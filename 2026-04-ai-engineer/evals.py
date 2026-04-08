"""Evaluation dataset construction and scoring for political relations extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from cases import DEFAULT_CASES_PATH, SplitFilter, load_case_records
from task import PoliticalRelation, RelationFocus, TaskInput, classify_relation_scope, relation_in_focus


@dataclass
class RelationsCaseMetadata:
    """Metadata for political relations extraction test cases."""

    split: Literal['train', 'val', 'test']
    focus: RelationFocus
    description: str
    expected_count: int
    raw_expected_count: int


def normalize_text(value: str) -> str:
    """Normalize free text for matching."""
    value = value.casefold().replace('–', '-').replace('—', '-')
    value = re.sub(r'[^a-z0-9]+', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def normalize_name(value: str) -> str:
    """Normalize names while dropping common honorifics."""
    value = normalize_text(value)
    value = re.sub(
        r'\b(rt hon|hon|sir|dame|lord|lady|baroness|baron|dr|the)\b',
        ' ',
        value,
    )
    return re.sub(r'\s+', ' ', value).strip()


def token_set(value: str) -> set[str]:
    """Tokenize text for loose overlap scoring."""
    stopwords = {
        'and',
        'the',
        'of',
        'for',
        'to',
        'a',
        'an',
        'member',
        'members',
        'party',
        'former',
        'served',
        'serving',
        'leader',
    }
    return {token for token in normalize_text(value).split() if len(token) > 2 and token not in stopwords}


def names_match(expected: str, actual: str) -> bool:
    """Check whether two person names are close enough to be the same person."""
    expected_name = normalize_name(expected)
    actual_name = normalize_name(actual)
    if not expected_name or not actual_name:
        return False
    if expected_name == actual_name:
        return True
    shorter_length = min(len(expected_name), len(actual_name))
    return shorter_length >= 5 and (expected_name in actual_name or actual_name in expected_name)


def relation_match_score(expected: str, actual: str) -> float:
    """Score how closely two relation labels align."""
    expected_relation = normalize_text(expected)
    actual_relation = normalize_text(actual)
    if not expected_relation or not actual_relation:
        return 0.0
    if expected_relation == actual_relation:
        return 1.0
    if expected_relation in actual_relation or actual_relation in expected_relation:
        return 0.85
    if classify_relation_scope(expected) == classify_relation_scope(actual):
        return 0.45
    return 0.0


def role_match_score(expected: str, actual: str) -> float:
    """Score how closely two role descriptions align."""
    expected_role = normalize_text(expected)
    actual_role = normalize_text(actual)
    if not expected_role or not actual_role:
        return 0.0
    if expected_role == actual_role:
        return 1.0
    if expected_role in actual_role or actual_role in expected_role:
        return 0.85
    expected_tokens = token_set(expected)
    actual_tokens = token_set(actual)
    if not expected_tokens or not actual_tokens:
        return 0.0
    overlap = len(expected_tokens & actual_tokens) / max(len(expected_tokens), len(actual_tokens))
    if overlap >= 0.6:
        return 0.7
    if overlap >= 0.3:
        return 0.45
    if overlap > 0:
        return 0.2
    return 0.0


def party_match_score(expected: str | None, actual: str | None) -> float:
    """Score party agreement, treating missing expected parties as neutral."""
    if expected is None:
        return 1.0
    if actual is None:
        return 0.0
    return 1.0 if normalize_text(expected) == normalize_text(actual) else 0.0


@dataclass
class MatchDetails:
    """Detailed match information for one expected/output pair."""

    total: float
    relation_score: float
    role_score: float
    party_score: float


def score_pair(expected: PoliticalRelation, actual: PoliticalRelation) -> MatchDetails:
    """Produce a weighted score for an expected/output relation pair."""
    if not names_match(expected.name, actual.name):
        return MatchDetails(total=0.0, relation_score=0.0, role_score=0.0, party_score=0.0)

    relation_score = relation_match_score(expected.relation, actual.relation)
    role_score = role_match_score(expected.role, actual.role)
    party_score = party_match_score(expected.party, actual.party)
    total = 0.5 + 0.3 * relation_score + 0.15 * role_score + 0.05 * party_score
    return MatchDetails(
        total=total,
        relation_score=relation_score,
        role_score=role_score,
        party_score=party_score,
    )


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

        remaining_outputs = list(output)
        matched_score = 0.0
        matched_pairs = 0
        relation_matches = 0
        role_matches = 0
        party_matches = 0

        for exp in expected:
            best_index = -1
            best_match = MatchDetails(total=0.0, relation_score=0.0, role_score=0.0, party_score=0.0)
            for index, out in enumerate(remaining_outputs):
                candidate = score_pair(exp, out)
                if candidate.total > best_match.total:
                    best_index = index
                    best_match = candidate

            if best_index >= 0 and best_match.total > 0:
                matched_score += best_match.total
                matched_pairs += 1
                relation_matches += int(best_match.relation_score >= 0.85)
                role_matches += int(best_match.role_score >= 0.45)
                party_matches += int(best_match.party_score == 1.0 and exp.party is not None)
                remaining_outputs.pop(best_index)

        precision = matched_score / len(output) if output else 0.0
        recall = matched_score / len(expected)
        if precision + recall == 0:
            accuracy = 0.0
        else:
            accuracy = 2 * precision * recall / (precision + recall)

        off_focus_output_count = 0
        if ctx.metadata is not None and ctx.metadata.focus == 'ancestors':
            off_focus_output_count = sum(
                1 for relation in output if not relation_in_focus(relation.relation, ctx.metadata.focus)
            )
        focus_compliance = 1.0 if not output else (len(output) - off_focus_output_count) / len(output)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'expected_count': len(expected),
            'output_count': len(output),
            'matched_pairs': matched_pairs,
            'matched_score': matched_score,
            'relation_matches': relation_matches,
            'role_matches': role_matches,
            'party_matches': party_matches,
            'focus_compliance': focus_compliance,
            'off_focus_output_count': off_focus_output_count,
        }


def load_relations_dataset(
    *,
    cases_file: str | None = None,
    split: SplitFilter = 'all',
    focus: RelationFocus = 'all',
    max_cases: int | None = None,
) -> Dataset[TaskInput, list[PoliticalRelation], RelationsCaseMetadata]:
    """Load a dataset from the persisted golden cases file."""
    records = load_case_records(
        DEFAULT_CASES_PATH if cases_file is None else Path(cases_file),
        split=split,
        max_cases=max_cases,
    )

    cases: list[Case[TaskInput, list[PoliticalRelation], RelationsCaseMetadata]] = []
    for record in records:
        expected_output = [
            relation.model_copy(deep=True)
            for relation in record.expected_output
            if relation_in_focus(relation.relation, focus)
        ]
        cases.append(
            Case(
                name=record.name,
                inputs=TaskInput(mp=record.mp),
                expected_output=expected_output,
                metadata=RelationsCaseMetadata(
                    split=record.split,
                    focus=focus,
                    description=f'{record.mp.name} ({record.mp.party})',
                    expected_count=len(expected_output),
                    raw_expected_count=len(record.expected_output),
                ),
            )
        )

    return Dataset(
        name=f'political_relations_extraction_{focus}_{split}',
        cases=cases,
        evaluators=[RelationsAccuracyEvaluator()],
    )
