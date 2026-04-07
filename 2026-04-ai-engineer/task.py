"""The task being optimized: extracting contact information from text.

This module defines:
- ContactInfo: The structured output schema for extraction
- extract_contact_info: The task function that runs the agent
- contact_agent: The pydantic-ai agent performing the extraction
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent


class ContactInfo(BaseModel, use_attribute_docstrings=True):
    """Extracted contact information from text."""

    name: str | None = None
    """The person's full name"""
    email: str | None = None
    """Email address"""
    phone: str | None = None
    """Phone number"""
    company: str | None = None
    """Company or organization name"""
    title: str | None = None
    """Job title or role"""


# The base agent with minimal instructions
# The actual instructions will be overridden during optimization
contact_agent = Agent(
    'gateway/openai:gpt-4o-mini',
    output_type=ContactInfo,
    instructions='Extract contact information from the provided text.',
    instrument=True,
    defer_model_check=True,  # Defer model validation to runtime
)


@dataclass
class TaskInput:
    """Input to the contact extraction task."""

    text: str


async def extract_contact_info(input: TaskInput) -> ContactInfo:
    """Run the contact extraction agent on the input text.

    This is the task function that will be evaluated and optimized.
    The agent's instructions can be overridden via agent.override() to test
    different prompts during optimization.
    """
    result = await contact_agent.run(input.text)
    return result.output
