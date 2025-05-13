from pathlib import Path
from types import NoneType

import logfire
from agent import TimeRangeInputs, TimeRangeResponse, infer_time_range
from custom_evaluators import CUSTOM_EVALUATOR_TYPES
from pydantic_evals import Dataset

logfire.configure(
    environment='development', service_name='evals', service_version='0.0.1'
)
logfire.instrument_pydantic_ai()

dataset_path = Path(__file__).parent / 'datasets' / 'time_range_v2.yaml'
dataset = Dataset[TimeRangeInputs, TimeRangeResponse, NoneType].from_file(
    dataset_path, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES
)
report = dataset.evaluate_sync(infer_time_range)
print(report)

assertion_pass_rate = report.averages().assertions
assert assertion_pass_rate is not None, 'There should be at least one assertion'
assert assertion_pass_rate > 0.8, (
    f'The assertion pass rate was {assertion_pass_rate:.1%}; it should be above 80%.'
)
