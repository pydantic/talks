"""Run prompt optimization for contact extraction.

This script demonstrates using GEPA with pydantic-ai and pydantic-evals
to automatically optimize agent instructions.

Usage:
    # Run a quick evaluation with current instructions
    uv run python pai-gepa-prompt-optimization/run_optimization.py eval

    # Run optimization
    uv run python pai-gepa-prompt-optimization/run_optimization.py optimize

    # Run optimization with custom settings
    uv run python pai-gepa-prompt-optimization/run_optimization.py optimize --max-calls 100
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
from evals import contact_dataset
from task import contact_agent, extract_contact_info

# Configure logfire for observability
logfire.configure(
    send_to_logfire='if-token-present',
    environment='development',
    service_name='prompt-optimization-example',
)
logfire.instrument_pydantic_ai()


# Initial weak instructions (what we're trying to improve)
INITIAL_INSTRUCTIONS = 'Extract contact information from the provided text.'

# Expert instructions (for comparison - what good instructions look like)
EXPERT_INSTRUCTIONS = """Extract contact information from the provided text with high precision.

Guidelines:
1. NAME: Look for full names (first + last). Ignore titles like Dr., Mr., etc. in the extracted name.
2. EMAIL: Extract any valid email address format (user@domain.tld).
3. PHONE: Extract phone numbers in any format (with or without country codes, parentheses, dashes).
4. COMPANY: Look for organization names, often near titles or after "at" or "from".
5. TITLE: Extract job titles/roles, usually appearing near names or before company names.

Important:
- If multiple contacts appear, focus on the PRIMARY contact being introduced or highlighted.
- If information is missing, leave the field as null rather than guessing.
- Normalize phone numbers by preserving the original format.
- For names, extract just the name without titles or credentials (Ph.D., Jr., etc.)."""


def run_evaluation(instructions: str = INITIAL_INSTRUCTIONS) -> None:
    """Run evaluation with the given instructions and print results."""
    print(f'\nRunning evaluation with instructions:\n{instructions[:100]}...')
    print('-' * 60)

    async def evaluate():
        with contact_agent.override(instructions=instructions):
            report = await contact_dataset.evaluate(
                extract_contact_info,
                max_concurrency=5,
                progress=True,
            )
        return report

    report = asyncio.run(evaluate())

    # Print summary
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
    """Run GEPA optimization to improve instructions.

    Args:
        max_metric_calls: Maximum number of evaluation calls
        output_file: Optional file to save optimized instructions

    Returns:
        The optimized instructions string
    """
    print('\nStarting prompt optimization...')
    print(f'Max metric calls: {max_metric_calls}')
    print('-' * 60)

    # Create the adapter
    adapter = create_adapter(
        dataset=contact_dataset,
        task=extract_contact_info,
        agent=contact_agent,
        proposer_model='openai:gpt-4o',
        max_concurrency=5,
    )

    # Create seed candidate
    seed_candidate = {'instructions': json.dumps(INITIAL_INSTRUCTIONS)}

    # Run optimization
    result = optimize(  # pyright: ignore[reportUnknownVariableType]
        seed_candidate=seed_candidate,
        trainset=contact_dataset.cases,
        valset=contact_dataset.cases,  # Using same set for this example
        adapter=adapter,
        max_metric_calls=max_metric_calls,
        display_progress_bar=True,
    )

    assert isinstance(result.best_candidate, dict)
    # Extract best instructions
    best_instructions = json.loads(result.best_candidate['instructions'])

    print('\n' + '=' * 60)
    print('Optimization Complete!')
    print('=' * 60)
    print(f'\nBest validation score: {result.val_aggregate_scores[result.best_idx]:.2%}')
    print(f'\nOptimized Instructions:\n{best_instructions}')

    # Save if requested
    if output_file:
        Path(output_file).write_text(best_instructions)
        print(f'\nSaved to: {output_file}')

    return best_instructions


def compare_instructions() -> None:
    """Compare initial, expert, and optimized instructions."""
    print('\n' + '=' * 60)
    print('Comparison: Initial vs Expert Instructions')
    print('=' * 60)

    print('\n1. Evaluating INITIAL instructions:')
    run_evaluation(INITIAL_INSTRUCTIONS)

    print('\n2. Evaluating EXPERT instructions:')
    run_evaluation(EXPERT_INSTRUCTIONS)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Prompt optimization example using GEPA with pydantic-ai')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Run evaluation only')
    eval_parser.add_argument('--expert', action='store_true', help='Use expert instructions')

    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Run optimization')
    opt_parser.add_argument('--max-calls', type=int, default=50, help='Maximum metric calls')
    opt_parser.add_argument('--output', type=str, help='File to save optimized instructions')

    # Compare command
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
