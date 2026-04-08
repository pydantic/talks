"""GEPA adapter for pydantic-evals integration.

This module provides:
- EvalsGEPAAdapter: Bridges GEPA optimization with pydantic-evals evaluation
- Uses pydantic_ai.Agent.override() to inject candidate instructions

This approach demonstrates prompt optimization WITHOUT using Logfire managed
variables (which are not yet released). Instead, we use the Agent.override()
context manager to inject instructions during evaluation.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from pydantic_ai import Agent
from pydantic_core import to_jsonable_python
from pydantic_evals import Case, Dataset
from pydantic_evals.reporting import ReportCase, ReportCaseFailure

InputsT = TypeVar('InputsT')
OutputT = TypeVar('OutputT')
MetadataT = TypeVar('MetadataT')


@dataclass
class EvalTrajectory(Generic[InputsT, OutputT, MetadataT]):
    """Trajectory data from an evaluation run.

    Contains the evaluation report case for reflection during optimization.
    """

    report_case: ReportCase[InputsT, OutputT, MetadataT] | ReportCaseFailure[InputsT, OutputT, MetadataT]


@dataclass
class EvalsGEPAAdapter(
    GEPAAdapter[
        Case[InputsT, OutputT, MetadataT],
        EvalTrajectory[InputsT, OutputT, MetadataT],
        OutputT | None,
    ],
    Generic[InputsT, OutputT, MetadataT],
):
    """GEPA adapter that uses pydantic-evals for evaluation.

    This adapter:
    - Uses Agent.override(instructions=...) to inject candidate prompts
    - Runs evaluation via pydantic-evals Dataset.evaluate()
    - Extracts scores and trajectories for GEPA optimization

    The candidate dict is expected to have a single key "instructions"
    containing the JSON-serialized instructions string.

    Type Parameters:
        InputsT: The type of evaluation task inputs
        OutputT: The type of evaluation task outputs
        MetadataT: The type of evaluation case metadata
    """

    # The evaluation dataset
    dataset: Dataset[InputsT, OutputT, MetadataT]

    # The task function to evaluate
    task: Callable[[InputsT], Awaitable[OutputT]]

    # The agent whose instructions will be overridden
    agent: Agent[Any, Any]

    # The score key to use from evaluator results (e.g., 'accuracy')
    score_key: str = 'accuracy'

    # Maximum concurrent evaluations
    max_concurrency: int = 5

    # Model to use for the task during evaluation
    task_model: str | None = None

    # Model to use for proposing new instructions
    proposer_model: str = 'openai:gpt-4o'

    # The proposer agent for generating new instruction candidates
    _proposer_agent: Agent[Any, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the proposer agent."""
        # Create proposer agent
        self._proposer_agent = Agent(
            self.proposer_model,
            output_type=str,
            defer_model_check=True,
            instructions="""You are an expert prompt engineer. Your task is to improve system prompts
for AI agents based on evaluation feedback.

You will receive:
1. The current instructions being used
2. Examples of inputs, outputs, and feedback from evaluation

Analyze what went wrong and propose improved instructions that will:
- Increase accuracy on the task
- Handle edge cases better
- Be clear and specific

Return ONLY the improved instructions text, nothing else.""",
        )

        # Required by GEPA protocol
        self.propose_new_texts = self._propose_new_texts_impl

    def evaluate(
        self,
        batch: list[Case[InputsT, OutputT, MetadataT]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[EvalTrajectory[InputsT, OutputT, MetadataT], OutputT | None]:
        """Evaluate a candidate on a batch of cases.

        Args:
            batch: The cases to evaluate
            candidate: Dict with 'instructions' key containing the prompt to test
            capture_traces: Whether to capture trajectories for reflection

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories
        """
        # Parse the candidate instructions
        instructions = json.loads(candidate['instructions'])

        temp_cases = [
            Case(
                name=f'{case.name or "case"}__{index}',
                inputs=case.inputs,
                metadata=case.metadata,
                expected_output=case.expected_output,
                evaluators=tuple(case.evaluators),
            )
            for index, case in enumerate(batch)
        ]

        # Create a temporary dataset with just the batch cases
        temp_dataset: Dataset[InputsT, OutputT, MetadataT] = Dataset(
            name=f'{self.dataset.name or "relations"}_batch',
            cases=temp_cases,
            evaluators=list(self.dataset.evaluators),
        )

        async def evaluate_batch() -> Any:
            return await temp_dataset.evaluate(
                self.task,
                max_concurrency=self.max_concurrency,
                progress=False,
            )

        # Run evaluation with overridden instructions
        if self.task_model is not None:
            with self.agent.override(instructions=instructions, model=self.task_model):
                report = asyncio.run(evaluate_batch())
        else:
            with self.agent.override(instructions=instructions):
                report = asyncio.run(evaluate_batch())

        # Extract results
        outputs: list[OutputT | None] = []
        scores: list[float] = []
        trajectories: list[EvalTrajectory[InputsT, OutputT, MetadataT]] | None = [] if capture_traces else None

        for case_report in report.cases:
            outputs.append(case_report.output)

            # Get optimization score from the evaluator results
            score_result = case_report.scores.get(self.score_key)
            if score_result is not None:
                scores.append(float(score_result.value))
            else:
                # Fallback: check if any score exists
                if case_report.scores:
                    # Use first available score
                    first_score = next(iter(case_report.scores.values()))
                    scores.append(float(first_score.value))
                else:
                    scores.append(0.0)

            if capture_traces and trajectories is not None:
                trajectories.append(EvalTrajectory(report_case=case_report))

        # Handle failures
        for failure in report.failures:
            outputs.append(None)
            scores.append(0.0)

            if capture_traces and trajectories is not None:
                trajectories.append(EvalTrajectory(report_case=failure))

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[EvalTrajectory[InputsT, OutputT, MetadataT], OutputT | None],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build a dataset for the proposer LLM to reflect on.

        Args:
            candidate: The current candidate being evaluated
            eval_batch: Results from evaluation
            components_to_update: Which components to update (we only have 'instructions')

        Returns:
            Dict mapping component name to list of example records
        """
        if eval_batch.trajectories is None:
            return {}

        examples: list[dict[str, Any]] = []

        for traj, score in zip(eval_batch.trajectories, eval_batch.scores):
            case = traj.report_case

            record: dict[str, Any] = {
                'case_name': getattr(case, 'name', 'unknown'),
                'inputs': to_jsonable_python(case.inputs) if hasattr(case, 'inputs') else None,
                'expected_output': to_jsonable_python(case.expected_output)
                if hasattr(case, 'expected_output')
                else None,
                'score': score,
            }

            if isinstance(case, ReportCase):
                record['actual_output'] = to_jsonable_python(case.output)
                if case.scores:
                    record['scores'] = {k: v.value for k, v in case.scores.items()}
                if case.assertions:
                    record['assertions'] = [
                        {'name': a.name, 'passed': a.value, 'reason': a.reason} for a in case.assertions.values()
                    ]
            else:
                # Failure case
                record['error'] = getattr(case, 'error_stacktrace', str(case))

            examples.append(record)

        return {'instructions': examples}

    def _propose_new_texts_impl(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose improved instructions based on reflection.

        Args:
            candidate: Current candidate
            reflective_dataset: Feedback data from evaluation
            components_to_update: Components to update

        Returns:
            New candidate dict with improved instructions
        """
        current_instructions = json.loads(candidate['instructions'])
        examples = reflective_dataset.get('instructions', [])

        if not examples:
            return candidate

        # Build the prompt for the proposer
        examples_text = '\n\n'.join(
            [
                f'Example {i + 1}:\n'
                f'  Input: {json.dumps(ex.get("inputs"))}\n'
                f'  Expected: {json.dumps(ex.get("expected_output"))}\n'
                f'  Actual: {json.dumps(ex.get("actual_output"))}\n'
                f'  Score: {ex.get("score", 0):.2f}\n'
                f'  Feedback: {json.dumps(ex.get("scores", {}))}'
                for i, ex in enumerate(examples[:10])  # Limit to 10 examples
            ]
        )

        prompt = f"""Current Instructions:
{current_instructions}

Evaluation Results:
{examples_text}

Based on this feedback, propose improved instructions that will increase accuracy.
Focus on:
- What patterns led to incorrect outputs
- How to make the instructions clearer and more specific
- Edge cases that need to be handled

Respond with ONLY the new instructions text."""

        # Run the proposer agent
        result = self._proposer_agent.run_sync(prompt)
        new_instructions = result.output

        return {'instructions': json.dumps(new_instructions)}


def create_adapter(
    dataset: Dataset[InputsT, OutputT, MetadataT],
    task: Callable[[InputsT], Awaitable[OutputT]],
    agent: Agent[Any, Any],
    score_key: str = 'accuracy',
    task_model: str | None = None,
    proposer_model: str = 'openai:gpt-4o',
    max_concurrency: int = 5,
) -> EvalsGEPAAdapter[InputsT, OutputT, MetadataT]:
    """Create an EvalsGEPAAdapter for prompt optimization.

    Args:
        dataset: The evaluation dataset
        task: The task function to evaluate
        agent: The agent whose instructions will be optimized
        score_key: The key in the evaluator output to use as the optimization score.
            This should match the name of a score returned by your dataset's evaluators.
        task_model: Model to use for the task during evaluation.
        proposer_model: Model for generating new instruction candidates
        max_concurrency: Maximum concurrent evaluations

    Returns:
        Configured adapter for use with gepa.optimize()
    """
    return EvalsGEPAAdapter(
        dataset=dataset,
        task=task,
        agent=agent,
        score_key=score_key,
        task_model=task_model,
        max_concurrency=max_concurrency,
        proposer_model=proposer_model,
    )
