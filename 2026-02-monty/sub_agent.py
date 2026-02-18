from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic.dataclasses import dataclass
from pydantic_ai import Agent, ModelRetry, RunContext, format_as_xml


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


async def record_model_info(ctx: RunContext[ModelResults], model_information: dict[str, Any]) -> str:
    """Record information about a model.

    The input should have the following schema:

    ```json
    {
        "properties": {
        "unique_id": {
            "description": "Unique identifier for the model.",
            "title": "Unique Id",
            "type": "string"
        },
        "name": {
            "description": "Name of the model.",
            "title": "Name",
            "type": "string"
        },
        "description": {
            "anyOf": [
            {
                "type": "string"
            },
            {
                "type": "null"
            }
            ],
            "default": null,
            "description": "Description of the model.",
            "title": "Description"
        },
        "input_mtok": {
            "description": "Input tokens per million tokens.",
            "title": "Input Mtok",
            "type": "number"
        },
        "output_mtok": {
            "description": "Output tokens per million tokens.",
            "title": "Output Mtok",
            "type": "number"
        },
        "attributes": {
            "anyOf": [
            {
                "additionalProperties": {
                "anyOf": [
                    {
                    "type": "number"
                    },
                    {
                    "type": "string"
                    }
                ]
                },
                "type": "object"
            },
            {
                "type": "null"
            }
            ],
            "default": null,
            "description": "Any other attributes of the model.",
            "title": "Attributes"
        }
        },
        "required": [
        "unique_id",
        "name",
        "input_mtok",
        "output_mtok"
        ],
        "title": "ModelInfo",
        "type": "object"
    }
    ```
    """

    try:
        output = ModelInfo.model_validate(model_information)
    except ValidationError:
        pass
    else:
        ctx.deps.models[output.unique_id] = output
        return f'Model information recorded successfully for {output.unique_id}'

    try:
        result = await agent.run(format_as_xml(model_information))
    except ModelRetry:
        return 'Error, unable to validate model information'
    else:
        output = result.output
        if isinstance(output, str):
            return output
        else:
            ctx.deps.models[output.unique_id] = output
            return f'Model information recorded successfully for {output.unique_id}'
