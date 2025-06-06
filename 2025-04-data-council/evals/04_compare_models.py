from pathlib import Path
from types import NoneType

import logfire
from pydantic_evals import Dataset

from custom_evaluators import CUSTOM_EVALUATOR_TYPES
from agent import time_range_agent, infer_time_range, TimeRangeInputs, TimeRangeResponse

logfire.configure(environment='development', service_name='evals')
logfire.instrument_pydantic_ai()

dataset_path = Path(__file__).parent / 'datasets' / 'time_range_v2.yaml'
dataset = Dataset[TimeRangeInputs, TimeRangeResponse, NoneType].from_file(
    dataset_path, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES
)
with logfire.span('Comparing different models for time_range_agent'):
    with time_range_agent.override(model='openai:gpt-4o'):
        dataset.evaluate_sync(infer_time_range, name='openai:gpt-4o')
    with time_range_agent.override(model='anthropic:claude-3-7-sonnet-latest'):
        dataset.evaluate_sync(infer_time_range, name='anthropic:claude-3-7-sonnet-latest')
