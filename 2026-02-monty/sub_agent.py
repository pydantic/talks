from __future__ import annotations

from dataclasses import field

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from pydantic_ai import Agent, ModelRetry, RunContext


class ModelInfo(BaseModel, use_attribute_docstrings=True):
    unique_id: str
    """Unique identifier for the model."""
    name: str
    """Name of the model."""
    description: str | None = None
    """Description of the model."""
    input_mtok: float
    """Input tokens per million tokens."""
    output_mtok: float
    """Output tokens per million tokens."""
    attributes: dict[str, float | str] | None = None
    """Any other attributes of the model."""


agent = Agent(
    'gateway/anthropic:claude-sonnet-4-5',
    output_type=ModelInfo | str,
    instructions="Try to coerce the input into a ModelInfo object, if you're unable to do so, return a string describing the error.",
    name='extraction-sub-agent',
)


@dataclass
class ModelResults:
    models: dict[str, ModelInfo] = field(default_factory=dict)


async def record_model_info(ctx: RunContext[ModelResults], model_information: str) -> str:
    """Record information about a model.

    The input can be plain text, but should provide the following information:
    - unique_id(str): Unique identifier for the model. (REQUIRED)
    - name(str): Name of the model. (REQUIRED)
    - description(str | None): Description of the model. (OPTIONAL)
    - input_mtok(float): Input tokens per million tokens. (REQUIRED)
    - output_mtok(float): Output tokens per million tokens. (REQUIRED)
    - attributes(dict[str, float | str]): Any other attributes of the model. (OPTIONAL)
    """
    try:
        result = await agent.run(model_information)
    except ModelRetry:
        return 'Error, unable to validate model information'
    else:
        output = result.output
        if isinstance(output, str):
            return output
        else:
            ctx.deps.models[output.unique_id] = output
            return f'Model information recorded successfully for {output.unique_id}'
