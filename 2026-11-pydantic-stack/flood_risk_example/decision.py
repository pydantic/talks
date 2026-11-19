"""The underwriting decision, made by a decision model.

A decision model answers typed questions about a text, each with a confidence, instead of writing text.
Every field of `Decision` is one question and the fact sheet is the text. It is fast and cheap enough to
call once per property from a loop in the sandbox, and anything it is unsure of goes to a language model.
"""

import os
from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, ModelAPIError, ModelResponse, UseEnumMemberDocstrings
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel

from .policy import GUIDE, Facts

DECISION_MODEL = os.getenv('DECISION_MODEL', 'typesafe:jev-latest')
LANGUAGE_MODEL = 'gateway/anthropic:claude-sonnet-5-5'

# How sure each answer has to be before it stands; below this the language model decides instead.
CONFIDENCE_BARS = {'outcome': 0.6, 'flood_excess': 0.5}


class Outcome(UseEnumMemberDocstrings, StrEnum):
    accept = 'accept'
    """Offer cover: none of the guide's decline or refer conditions is true."""

    refer = 'refer'
    """Send to a senior underwriter: one of the guide's refer conditions is true, and no decline condition."""

    decline = 'decline'
    """Do not offer cover: one of the guide's decline conditions is true."""


class Decision(BaseModel):
    """Decide whether to offer buildings insurance for a property, following the underwriting guide."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    outcome: Outcome
    """What does the underwriting guide say to do with this property?"""

    flood_excess: bool
    """Does the underwriting guide apply a higher flood excess to this property?"""


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


decision_agent = Agent(output_type=Decision, name='decision_agent', instructions=f'Underwriting guide:\n\n{GUIDE}')


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
