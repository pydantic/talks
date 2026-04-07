"""Run prompt optimization for political relations extraction.

Usage:
    uv run -m main eval
    uv run -m main eval --expert
    uv run -m main compare
    uv run -m main optimize --max-calls 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import logfire
from gepa.api import optimize  # pyright: ignore[reportUnknownVariableType]

from adapter import create_adapter
from evals import relations_dataset
from task import extract_relations, relations_agent

# Configure logfire for observability
logfire.configure(
    send_to_logfire='if-token-present',
    environment='development',
    service_name='prompt-optimization-example',
)
logfire.instrument_pydantic_ai()


INITIAL_INSTRUCTIONS = """\
Your role is to inspect the contents the politician's wikipedia page and extract information
about any family members who were either a member of parliament a local councilor, or otherwise a politician.
"""

EXPERT_INSTRUCTIONS = """\
Extract information about political family members from a UK MP's Wikipedia page.

Guidelines:
1. RELATIONS: Only include family members who held political roles (MPs, councillors, MEPs, \
government ministers, party leaders, etc.)
2. RELATION TYPE: Use the exact relationship to the MP (father, mother, wife, husband, brother, \
sister, uncle, aunt, grandparent etc.)
3. ROLE: Describe their most notable political role(s) concisely.
4. PARTY: Include party affiliation if mentioned.
5. If no family members with political roles are found, return an empty list.

Important:
- Do NOT include the MP themselves, only their family members.
- Do NOT include non-political family members.
- Focus on clearly stated relationships, not speculation.
"""


def run_evaluation(instructions: str = INITIAL_INSTRUCTIONS) -> None:
    """Run evaluation with the given instructions and print results."""
    print(f'\nRunning evaluation with instructions:\n{instructions[:100]}...')
    print('-' * 60)

    async def evaluate():
        with relations_agent.override(instructions=instructions):
            report = await relations_dataset.evaluate(
                extract_relations,
                max_concurrency=5,
                progress=True,
            )
        return report

    report = asyncio.run(evaluate())

    print('\nEvaluation Results:')
    print('=' * 60)

    total_score = 0.0
    for case_report in report.cases:
        case_name = case_report.name if hasattr(case_report, 'name') else 'unknown'
        scores = case_report.scores if hasattr(case_report, 'scores') else {}
        accuracy_result = scores.get('accuracy')
        accuracy = float(accuracy_result.value) if accuracy_result else 0.0
        total_score += accuracy
        print(f'  {case_name}: accuracy={accuracy:.2f}')

    avg_score = total_score / len(report.cases) if report.cases else 0.0
    print('-' * 60)
    print(f'Average accuracy: {avg_score:.2%}')


def run_optimization(
    max_metric_calls: int = 50,
    output_file: str | None = None,
) -> str:
    """Run GEPA optimization to improve instructions."""
    print('\nStarting prompt optimization...')
    print(f'Max metric calls: {max_metric_calls}')
    print('-' * 60)

    adapter = create_adapter(
        dataset=relations_dataset,
        task=extract_relations,
        agent=relations_agent,
        proposer_model='openai:gpt-4o',
        max_concurrency=5,
    )

    seed_candidate = {'instructions': json.dumps(INITIAL_INSTRUCTIONS)}

    result = optimize(  # pyright: ignore[reportUnknownVariableType]
        seed_candidate=seed_candidate,
        trainset=relations_dataset.cases,
        valset=relations_dataset.cases,
        adapter=adapter,
        max_metric_calls=max_metric_calls,
        display_progress_bar=True,
    )

    assert isinstance(result.best_candidate, dict)
    best_instructions = json.loads(result.best_candidate['instructions'])

    print('\n' + '=' * 60)
    print('Optimization Complete!')
    print('=' * 60)
    print(f'\nBest validation score: {result.val_aggregate_scores[result.best_idx]:.2%}')
    print(f'\nOptimized Instructions:\n{best_instructions}')

    if output_file:
        Path(output_file).write_text(best_instructions)
        print(f'\nSaved to: {output_file}')

    return best_instructions


def compare_instructions() -> None:
    """Compare initial vs expert instructions."""
    print('\n' + '=' * 60)
    print('Comparison: Initial vs Expert Instructions')
    print('=' * 60)

    print('\n1. Evaluating INITIAL instructions:')
    run_evaluation(INITIAL_INSTRUCTIONS)

    print('\n2. Evaluating EXPERT instructions:')
    run_evaluation(EXPERT_INSTRUCTIONS)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Prompt optimization for political relations extraction')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    eval_parser = subparsers.add_parser('eval', help='Run evaluation only')
    eval_parser.add_argument('--expert', action='store_true', help='Use expert instructions')

    opt_parser = subparsers.add_parser('optimize', help='Run optimization')
    opt_parser.add_argument('--max-calls', type=int, default=50, help='Maximum metric calls')
    opt_parser.add_argument('--output', type=str, help='File to save optimized instructions')

    subparsers.add_parser('compare', help='Compare initial vs expert instructions')

    args = parser.parse_args()

    if args.command == 'eval':
        instructions = EXPERT_INSTRUCTIONS if args.expert else INITIAL_INSTRUCTIONS
        run_evaluation(instructions)
    elif args.command == 'optimize':
        run_optimization(
            max_metric_calls=args.max_calls,
            output_file=args.output,
        )
    elif args.command == 'compare':
        compare_instructions()
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
