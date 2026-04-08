from __future__ import annotations

import re
import sys
from pathlib import Path

import logfire
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pydantic_ai import Agent

from task import MP, TaskInput, ensure_data, extract_relations, get_mps

logfire.configure()

app = FastAPI()
logfire.instrument_fastapi(app)
logfire.instrument_pydantic_ai()

ensure_data()
_mps = get_mps()
_mp_lookup: dict[int, MP] = {mp.id: mp for mp in _mps}
_pages_dir = Path('mps/pages')


class AgentConfig(BaseModel):
    instructions: str
    model: str
    max_tokens: int


agent_config = logfire.var(
    name='mp_search_agent_config',
    type=AgentConfig,
    default=AgentConfig(
        instructions=(
            'You are a helpful assistant that answers questions about UK Members of Parliament. '
            'Use the mp_search tool to find information. Provide concise, factual answers based on the search results.'
        ),
        model='gateway/anthropic:claude-sonnet-4-5',
        max_tokens=1024,
    ),
)

search_agent = Agent(defer_model_check=True)


@search_agent.tool_plain
def mp_search(query: str) -> str:
    """Search MP Wikipedia pages for information matching a regex pattern.

    Args:
        query: A regex pattern to search for across all MP pages.
    """
    pattern = re.compile(query, re.IGNORECASE)
    results: list[str] = []
    for mp in _mps:
        page_file = _pages_dir / f'{mp.id}.html'
        if not page_file.exists():
            continue
        html = page_file.read_text()
        if not pattern.search(html):
            continue
        soup = BeautifulSoup(html, 'html.parser')
        body = soup.find(id='mw-content-text')
        if body is None:
            continue
        text = body.text
        if not pattern.search(text):
            continue
        for match in pattern.finditer(text):
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 200)
            snippet = text[start:end].strip()
            results.append(f'**{mp.name}** ({mp.party}): ...{snippet}...')
            if len(results) >= 20:
                break
    if not results:
        return 'No results found.'
    return '\n\n'.join(results)


@search_agent.tool_plain
async def extract_political_relations(mp_name: str) -> str:
    """Extract political family relations for a specific MP using their Wikipedia page.

    Args:
        mp_name: The name of the MP to look up. Must be an exact or close match.
    """
    name_lower = mp_name.lower()
    mp = next((m for m in _mps if name_lower in m.name.lower()), None)
    if mp is None:
        return f'MP "{mp_name}" not found.'
    relations = await extract_relations(TaskInput(mp=mp))
    if not relations:
        return f'{mp.name} ({mp.party}): No political family relations found.'
    lines = [f'{mp.name} ({mp.party}) political family relations:']
    for r in relations:
        party = f' [{r.party}]' if r.party else ''
        lines.append(f'  - {r.relation}: {r.name} — {r.role}{party}')
    return '\n'.join(lines)


@app.get('/', response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (Path(__file__).parent / 'index.html').read_text()
    return HTMLResponse(html)


class FormInput(BaseModel):
    query: str


@app.post('/form')
async def form(input: FormInput) -> dict[str, str]:
    with agent_config.get() as config:
        result = await search_agent.run(
            input.query,
            model=config.value.model,
            instructions=config.value.instructions,
            model_settings={
                'max_tokens': config.value.max_tokens,
            },
        )
    return {'result': result.output}


if __name__ == '__main__' and 'push-variables' in sys.argv:
    print('pushing variables...')
    logfire.variables_push()
