"""Task definitions for extracting political relations from MP Wikipedia pages."""

from __future__ import annotations

import asyncio
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup
from httpx import Client
from pydantic import BaseModel, TypeAdapter, computed_field
from pydantic_ai import Agent

data_dir = Path('mps')
mps_list_file = data_dir / 'list.json'
pages_dir = data_dir / 'pages'
DATA_URL = 'https://files.scolvin.com/mps.tar.gz'
InstructionStyle = Literal['initial', 'expert']
RelationScope = Literal['ancestor', 'same_generation', 'descendant', 'spouse', 'other']
DEFAULT_TASK_MODEL = 'gateway/anthropic:claude-haiku-4-5'


class MP(BaseModel):
    id: int
    name: str
    url: str
    raw_party: str

    @computed_field
    @property
    def party(self) -> Literal['Conservative', 'Labour', 'Liberal Democrat', 'Other']:
        raw_party = self.raw_party.lower()
        if 'conservative' in raw_party:
            return 'Conservative'
        elif 'labour' in raw_party:
            return 'Labour'
        elif 'liberal democrat' in raw_party:
            return 'Liberal Democrat'
        else:
            return 'Other'


mps_ta = TypeAdapter(list[MP])


class PoliticalRelation(BaseModel, use_attribute_docstrings=True):
    """Family member who was either a member of parliament, a local councilor, or otherwise a politician."""

    name: str
    """Name of the family member"""
    role: str
    """Political role of the family member"""
    relation: str
    """Relationship of the family member to the politician"""
    party: str | None = None
    """Political party of the family member"""


@dataclass
class TaskInput:
    """Input to the political relations extraction task."""

    mp: MP


ANCESTORS_INITIAL_INSTRUCTIONS = """
Inspect the supplied Wikipedia page text for a UK MP and extract only ancestor or parent-generation
relatives who held political roles.
"""

ANCESTORS_EXPERT_INSTRUCTIONS = """
Extract only political ancestors or parent-generation relatives from the supplied Wikipedia page text
for a UK MP.

Include parents, grandparents, great-grandparents, aunts, uncles, and similar older-generation relatives
when they held an elected office, government office, party leadership role, or another clearly political
public role.

Rules:
1. Exclude spouses, partners, siblings, cousins, children, and grandchildren even if they are political.
2. Use the relationship stated on the page as the `relation` value.
3. Keep `role` short and specific, focused on the relative's political role.
4. Include `party` only when the page states it or makes it clear.
5. Use only information supported by the provided text. If uncertain, omit the relative.
6. Never include the MP themselves.
7. If no qualifying relatives are found, return an empty list.
"""


def get_instructions(*, style: InstructionStyle) -> str:
    """Return the prompt text for a prompt style and evaluation focus."""
    return ANCESTORS_EXPERT_INSTRUCTIONS if style == 'expert' else ANCESTORS_INITIAL_INSTRUCTIONS


# The base agent with minimal instructions
# The actual instructions will be overridden during optimization
relations_agent = Agent(
    DEFAULT_TASK_MODEL,
    output_type=list[PoliticalRelation],
    instructions=get_instructions(style='initial'),
    instrument=True,
    defer_model_check=True,
)


async def extract_relations(input: TaskInput) -> list[PoliticalRelation]:
    """Run the political relations extraction agent on an MP's Wikipedia page.

    This is the task function that will be evaluated and optimized.
    The agent's instructions can be overridden via agent.override() to test
    different prompts during optimization.
    """
    ensure_data()
    page_file = pages_dir / f'{input.mp.id}.html'
    html = page_file.read_text()
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.find(id='mw-content-text')
    assert body is not None, f'Could not find body element for {input.mp.name}'
    result = await relations_agent.run(body.text)
    return result.output


def get_mps() -> list[MP]:
    """Load the list of MPs from the data directory."""
    ensure_data()
    return mps_ta.validate_json(mps_list_file.read_bytes())


# --- Utilities ---


def ensure_data() -> None:
    """Download and extract the MP data archive if not already present."""
    if data_dir.exists():
        return
    print('Downloading MP data...')
    with Client() as client:
        r = client.get(DATA_URL, follow_redirects=True)
        r.raise_for_status()
    archive_path = Path('mps.tar.gz')
    archive_path.write_bytes(r.content)
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall()
    archive_path.unlink()
    print('Download complete.')


if __name__ == '__main__':
    mp_id = int(sys.argv[1])
    mps = get_mps()
    mp = next((m for m in mps if m.id == mp_id), None)
    if mp is None:
        print(f'MP with id {mp_id} not found')
        sys.exit(1)
    print(f'Extracting relations for {mp.name} ({mp.party})...')
    relations = asyncio.run(extract_relations(TaskInput(mp=mp)))
    for r in relations:
        print(f'  {r.relation}: {r.name} — {r.role}')
    if not relations:
        print('  No political relations found')
