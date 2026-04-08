"""Run prompt optimization for political relations extraction.

Usage:
    uv run -m main generate-cases --limit 100
    uv run -m main eval --split test
    uv run -m main compare --split test
    uv run -m main optimize --train-split train --val-split val --max-calls 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast

import logfire
from gepa.api import optimize  # pyright: ignore[reportUnknownVariableType]

from adapter import create_adapter
from cases import DEFAULT_CASES_PATH, SplitFilter
from evals import load_relations_dataset
from task import (
    DEFAULT_TASK_MODEL,
    InstructionStyle,
    extract_relations,
    get_instructions,
    relations_agent,
)

# Configure logfire for observability
logfire.configure(
    send_to_logfire='if-token-present',
    environment='development',
    service_name='prompt-optimization-example',
    console=False,
    scrubbing=False,
)
logfire.instrument_pydantic_ai()
logfire.instrument_print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Prompt optimization for political relations extraction')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    eval_parser = subparsers.add_parser('eval', help='Run evaluation only')
    eval_parser.add_argument('--cases-file', type=str, default=str(DEFAULT_CASES_PATH), help='Golden cases JSON path')
    eval_parser.add_argument('--split', choices=['all', 'train', 'val', 'test'], default='all')
    eval_parser.add_argument('--prompt-style', choices=['initial', 'expert'], default='initial')
    eval_parser.add_argument('--instructions-file', type=str, help='Evaluate with custom instructions from a file')
    eval_parser.add_argument('--model', type=str, default=DEFAULT_TASK_MODEL, help='Model used for evaluation')
    eval_parser.add_argument('--max-cases', type=int, help='Limit the number of evaluation cases')

    opt_parser = subparsers.add_parser('optimize', help='Run optimization')
    opt_parser.add_argument('--cases-file', type=str, default=str(DEFAULT_CASES_PATH), help='Golden cases JSON path')
    opt_parser.add_argument('--train-split', choices=['all', 'train', 'val', 'test'], default='train')
    opt_parser.add_argument('--val-split', choices=['all', 'train', 'val', 'test'], default='val')
    opt_parser.add_argument('--task-model', type=str, default=DEFAULT_TASK_MODEL, help='Model being optimized')
    opt_parser.add_argument('--prompt-style', choices=['initial', 'expert'], default='initial')
    opt_parser.add_argument('--seed-instructions-file', type=str, help='Seed prompt file for optimization')
    opt_parser.add_argument('--max-calls', type=int, default=400, help='Maximum metric calls')
    opt_parser.add_argument('--output', type=str, help='File to save optimized instructions')
    opt_parser.add_argument('--max-train-cases', type=int, help='Limit training cases')
    opt_parser.add_argument('--max-val-cases', type=int, help='Limit validation cases')

    subparsers.add_parser('compare', help='Compare initial vs expert instructions')
    compare_parser = subparsers.choices['compare']
    compare_parser.add_argument(
        '--cases-file', type=str, default=str(DEFAULT_CASES_PATH), help='Golden cases JSON path'
    )
    compare_parser.add_argument('--split', choices=['all', 'train', 'val', 'test'], default='test')
    compare_parser.add_argument('--model', type=str, default=DEFAULT_TASK_MODEL, help='Model used for both runs')
    compare_parser.add_argument('--max-cases', type=int, help='Limit the number of comparison cases')

    args = parser.parse_args()

    if args.command == 'eval':
        run_evaluation(
            cases_file=args.cases_file,
            split=cast(SplitFilter, args.split),
            prompt_style=cast(InstructionStyle, args.prompt_style),
            instructions_file=args.instructions_file,
            model=args.model,
            max_cases=args.max_cases,
        )
    elif args.command == 'optimize':
        run_optimization(
            cases_file=args.cases_file,
            train_split=cast(SplitFilter, args.train_split),
            val_split=cast(SplitFilter, args.val_split),
            task_model=args.task_model,
            prompt_style=cast(InstructionStyle, args.prompt_style),
            seed_instructions_file=args.seed_instructions_file,
            max_metric_calls=args.max_calls,
            output_file=args.output,
            max_train_cases=args.max_train_cases,
            max_val_cases=args.max_val_cases,
        )
    elif args.command == 'compare':
        compare_instructions(
            cases_file=args.cases_file,
            split=cast(SplitFilter, args.split),
            model=args.model,
            max_cases=args.max_cases,
        )
    else:
        parser.print_help()
        return 1

    return 0


def run_evaluation(
    *,
    cases_file: str,
    split: 'SplitFilter',
    prompt_style: InstructionStyle,
    instructions_file: str | None,
    model: str,
    max_cases: int | None,
) -> float:
    """Run evaluation with the given instructions and print results."""
    dataset = load_relations_dataset(cases_file=cases_file, split=split, max_cases=max_cases)
    instructions = load_instructions(prompt_style=prompt_style, instructions_file=instructions_file)

    if not dataset.cases:
        raise ValueError('Dataset is empty after applying split/max-case filters.')

    print(f'\nRunning evaluation with instructions:\n{instructions[:100]}...')
    print(f'Model: {model}')
    print(f'Dataset: {cases_file} split={split} cases={len(dataset.cases)}')
    print('-' * 60)

    async def evaluate() -> Any:
        with relations_agent.override(instructions=instructions, model=model):
            return await dataset.evaluate(
                extract_relations,
                max_concurrency=5,
                progress=True,
                name=f'prompt={prompt_style}',
            )

    report = asyncio.run(evaluate())

    print('\nEvaluation Results:')
    print('=' * 60)

    total_score = 0.0
    total_cases = len(report.cases) + len(report.failures)
    for case_report in report.cases:
        case_name = cast(str, getattr(case_report, 'name', 'unknown'))
        scores = cast(dict[str, Any], getattr(case_report, 'scores', {}))
        accuracy_result = cast(Any, scores.get('accuracy'))
        expected_count = cast(Any, scores.get('expected_count'))
        output_count = cast(Any, scores.get('output_count'))
        accuracy = float(accuracy_result.value) if accuracy_result else 0.0
        total_score += accuracy
        expected_value = int(expected_count.value) if expected_count else 0
        output_value = int(output_count.value) if output_count else 0
        print(f'  {case_name}: accuracy={accuracy:.2f} expected={expected_value} output={output_value}')

    for failure in report.failures:
        failure_name = cast(str, getattr(failure, 'name', 'unknown'))
        print(f'  {failure_name}: failed')

    avg_score = total_score / total_cases if total_cases else 0.0
    print('-' * 60)
    print(f'Average accuracy: {avg_score:.2%}')
    return avg_score


def run_optimization(
    *,
    cases_file: str,
    train_split: 'SplitFilter',
    val_split: 'SplitFilter',
    task_model: str,
    prompt_style: InstructionStyle,
    seed_instructions_file: str | None,
    max_metric_calls: int = 50,
    output_file: str | None = None,
    max_train_cases: int | None = None,
    max_val_cases: int | None = None,
) -> str:
    """Run GEPA optimization to improve instructions."""
    train_dataset = load_relations_dataset(cases_file=cases_file, split=train_split, max_cases=max_train_cases)
    val_dataset = load_relations_dataset(cases_file=cases_file, split=val_split, max_cases=max_val_cases)
    if not train_dataset.cases or not val_dataset.cases:
        raise ValueError('Training and validation datasets must both contain at least one case.')

    with logfire.span(
        'optimization {max_metric_calls=}',
        max_metric_calls=max_metric_calls,
        task_model=task_model,
        training_cases=len(train_dataset.cases),
        val_cases=len(val_dataset.cases),
    ):
        adapter = create_adapter(
            dataset=train_dataset,
            task=extract_relations,
            agent=relations_agent,
            task_model=task_model,
            max_concurrency=5,
        )

        seed_instructions = load_instructions(prompt_style=prompt_style, instructions_file=seed_instructions_file)
        seed_candidate = {'instructions': json.dumps(seed_instructions)}

        result = optimize(  # pyright: ignore[reportUnknownVariableType]
            seed_candidate=seed_candidate,
            trainset=train_dataset.cases,
            valset=val_dataset.cases,
            adapter=adapter,
            max_metric_calls=max_metric_calls,
            display_progress_bar=True,
        )

        assert isinstance(result.best_candidate, dict)
        best_instructions = json.loads(result.best_candidate['instructions'])

    print(f'Optimization Complete! Best validation score: {result.val_aggregate_scores[result.best_idx]:.2%}')
    print(f'Optimized Instructions:\n{best_instructions}')

    logfire.info(
        'optimization_complete best score {score:.2%}',
        score=result.val_aggregate_scores[result.best_idx],
        best_instructions=best_instructions,
    )

    if output_file:
        Path(output_file).write_text(best_instructions)
        print(f'\nSaved to: {output_file}')

    return best_instructions


def compare_instructions(
    *,
    cases_file: str,
    split: 'SplitFilter',
    model: str,
    max_cases: int | None,
) -> None:
    """Compare initial vs expert instructions."""
    initial_score = run_evaluation(
        cases_file=cases_file,
        split=split,
        prompt_style='initial',
        instructions_file=None,
        model=model,
        max_cases=max_cases,
    )

    expert_score = run_evaluation(
        cases_file=cases_file,
        split=split,
        prompt_style='expert',
        instructions_file=None,
        model=model,
        max_cases=max_cases,
    )
    print(f'\nDelta: {expert_score - initial_score:+.2%}')


def print_case_summary(cases_file: str) -> None:
    """Print a quick summary of persisted golden cases."""
    records = load_relations_dataset(cases_file=cases_file, split='all').cases
    print(f'Loaded {len(records)} cases from {cases_file}')


# --- Utilities ---


def load_instructions(
    *,
    prompt_style: InstructionStyle,
    instructions_file: str | None,
) -> str:
    """Resolve prompt text from either a preset style or a file."""
    if instructions_file is not None:
        return Path(instructions_file).read_text()
    return get_instructions(style=prompt_style)


if __name__ == '__main__':
    sys.exit(main())
