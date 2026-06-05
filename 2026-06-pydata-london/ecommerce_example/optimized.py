"""Run with `uv run -m ecommerce_example.optimized`."""

from pathlib import Path

import logfire

from .agent import agent

variables = logfire.VariablesOptions()
logfire.configure(service_name='ecommerce_example_optimized', variables=variables)
logfire.instrument_pydantic_ai()

monty_ecommerce_agent_instructions = logfire.var(
    'monty_ecommerce_agent_instructions',
    default=(Path(__file__).parent / 'instructions.md').read_text(),
    description='Instructions for the Monty Ecommerce Agent.',
)

with monty_ecommerce_agent_instructions.get() as resolve:
    logfire.info(f'managed prompt variable {resolve.label}')
    result = agent.run_sync(
        'Which acquisition channel brings in the highest-value customers (by spend)?',
        instructions=resolve.value,
    )
    result = agent.run_sync(
        'Which customer segment drives the most revenue, and how does that break down by region? Tell marketing where to focus.',
        instructions=resolve.value,
    )
    result = agent.run_sync(
        'Do our customers buy differently on weekdays vs weekends? Look at order volume and average order size by day of week.',
        instructions=resolve.value,
    )
