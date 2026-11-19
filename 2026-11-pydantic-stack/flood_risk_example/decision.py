"""The underwriting decision, made by a decision model.

A decision model answers typed questions about a text, each with a confidence, instead of writing text.
Every field of `Decision` is one question and the fact sheet is the text. It is fast and cheap enough to
call once per property from a loop in the sandbox, and anything it is unsure of goes to a language model.
"""

import os
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, BoolCriteria, ModelAPIError, ModelResponse, UseEnumMemberDocstrings
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel

from .policy import Facts

DECISION_MODEL = os.getenv('DECISION_MODEL', 'typesafe:jev-latest')
LANGUAGE_MODEL = 'gateway/anthropic:claude-sonnet-5-5'

# How sure each answer has to be before it stands; below this the language model decides instead.
CONFIDENCE_BARS = {'outcome': 0.6, 'flood_excess': 0.5}


class Outcome(UseEnumMemberDocstrings, StrEnum):
    accept = 'accept'
    """Offer cover. The property is outside every mapped flood zone and at least 2 m above the nearest
    water, or it is in the 0.2% annual chance zone and at least 3 m above the nearest water."""

    refer = 'refer'
    """Send to a senior underwriter. The property is in the 1% annual chance zone and 2 m or more above
    the nearest water, or in the 0.2% zone and less than 3 m above it, or outside every mapped zone but
    less than 2 m above it."""

    decline = 'decline'
    """Do not offer cover. The property is in the regulatory floodway, or in the 1% annual chance zone
    and less than 2 m above the nearest water."""


class Decision(BaseModel):
    """Decide whether to offer buildings insurance for a property, from its flood risk."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    outcome: Outcome
    """What should the underwriter do with this property?"""

    flood_excess: Annotated[
        bool,
        BoolCriteria(
            true='The property is in the floodway, the 1% zone or the 0.2% zone.',
            false='The property is outside every mapped flood zone.',
        ),
    ]
    """Should a higher excess apply to flood claims?"""


class DecisionResult(BaseModel):
    property_id: str
    outcome: Outcome
    flood_excess: bool
    confidence: dict[str, float] | None
    """Confidence in each answer, from 0 to 1. None when a language model made the decision."""
    decided_by: str
    """The model that made the decision."""


def unsure(response: ModelResponse) -> bool:
    confidence = (response.provider_details or {}).get('confidence', {})
    return any(value < CONFIDENCE_BARS.get(field, 0.5) for field, value in confidence.items())


def decision_model(name: str = DECISION_MODEL) -> Model | str:
    """A decision model backed by a language model for the answers it is unsure of.

    A language model name is used as it is, which is how the offline evals compare the two.
    """
    if name.startswith(('typesafe:', 'system-one:')):
        return FallbackModel(name, LANGUAGE_MODEL, fallback_on=[ModelAPIError, unsure])
    return name


decision_agent = Agent(output_type=Decision, name='decision_agent')


async def decide(facts: Facts, model: Model | str | None = None) -> DecisionResult:
    result = await decision_agent.run(facts.fact_sheet(), model=model or decision_model())
    details = result.response.provider_details or {}
    return DecisionResult(
        property_id=facts.property_id,
        outcome=result.output.outcome,
        flood_excess=result.output.flood_excess,
        confidence=details.get('confidence'),
        decided_by=result.response.model_name or 'unknown',
    )
