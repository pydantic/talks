"""task.py, but with a managed variable for the prompt."""

from __future__ import annotations

import logfire
from bs4 import BeautifulSoup
from pydantic_ai import Agent

from task import PoliticalRelation, TaskInput, pages_dir

instructions_var = logfire.var(
    name='mp_relations_instructions',
    type=str,
    default="""
Inspect the supplied Wikipedia page text for a UK MP and extract only ancestor or parent-generation
relatives who held political roles.
""",
)


# The base agent with minimal instructions
# The actual instructions will be overridden during optimization
relations_agent = Agent(
    'gateway/openai:gpt-4.1',
    output_type=list[PoliticalRelation],
    instrument=True,
    defer_model_check=True,
)


@relations_agent.instructions
def get_runtime_instructions() -> str:
    """Resolve the latest instructions from the managed variable provider."""
    return instructions_var.get().value


async def extract_relations(input: TaskInput) -> list[PoliticalRelation]:
    """Run the political relations extraction agent on an MP's Wikipedia page.

    This is the task function that will be evaluated and optimized.
    The agent's instructions can be overridden via agent.override() to test
    different prompts during optimization.
    """
    page_file = pages_dir / f'{input.mp.id}.html'
    html = page_file.read_text()
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.find(id='mw-content-text')
    assert body is not None, f'Could not find body element for {input.mp.name}'
    result = await relations_agent.run(body.text)
    return result.output


if __name__ == '__main__':
    print('pushing variables...')
    logfire.configure()
    logfire.variables_push()
