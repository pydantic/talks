from pathlib import Path
from types import NoneType

from pydantic_evals import Dataset

from custom_evaluators import CUSTOM_EVALUATOR_TYPES, AgentCalledTool, UserMessageIsConcise, ValidateTimeRange
from agent import TimeRangeInputs, TimeRangeResponse


def main():
    dataset_path = Path(__file__).parent / 'datasets' / 'time_range_v1.yaml'
    dataset = Dataset[TimeRangeInputs, TimeRangeResponse, NoneType].from_file(dataset_path)
    dataset.add_evaluator(ValidateTimeRange())
    dataset.add_evaluator(UserMessageIsConcise())
    dataset.add_evaluator(
        AgentCalledTool('time_range_agent', 'get_current_time'),
        specific_case='Single time point',
    )
    dataset.to_file(
        Path(__file__).parent / 'datasets' / 'time_range_v2.yaml',
        custom_evaluator_types=CUSTOM_EVALUATOR_TYPES,
    )


if __name__ == '__main__':
    main()
