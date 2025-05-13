from pathlib import Path
from types import NoneType

import logfire
from pydantic_evals import Dataset

from agent import infer_time_range, TimeRangeInputs, TimeRangeResponse
from custom_evaluators import CUSTOM_EVALUATOR_TYPES

logfire.configure(environment='development', service_name='evals', service_version='0.0.1')
logfire.instrument_pydantic_ai()

dataset_path = Path(__file__).parent / 'datasets' / 'time_range_v2.yaml'
dataset = Dataset[TimeRangeInputs, TimeRangeResponse, NoneType].from_file(
    dataset_path, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES
)
report = dataset.evaluate_sync(infer_time_range)
print(report)

assertion_pass_rate = report.averages().assertions
assert assertion_pass_rate is not None, 'There should be at least one assertion'
assert assertion_pass_rate > 0.9, f'The assertion pass rate was {assertion_pass_rate:.1%}; it should be above 90%.'
