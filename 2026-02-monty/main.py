from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import logfire
import pydantic_core
import tiktoken
from devtools import debug
from playwright.async_api import async_playwright
from pydantic import BaseModel
from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.code_execution import CodeExecutionToolset

from bs import tools
from sub_agent import ModelInfo, RunDeps, record_model_info


async def get_html(url: str) -> str:
    """Get the HTML content of a URL."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until='networkidle')
        html = await page.content()
        await browser.close()
    return html


class OutputData(BaseModel, use_attribute_docstrings=True):
    run_summary: str
    """Summary of how you performed the task and any issues encountered."""
    optimal_code: str
    """Optimal python code for getting prices data, to use in future runs.

    This should NOT contain any model names or model-specific details, just the code needed to get and extract
    the data.
    """


_tiktoken_encoding = tiktoken.get_encoding('cl100k_base')


class TruncateCodeExecutionToolset(CodeExecutionToolset[AgentDepsT]):
    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        output = await super().call_tool(*args, **kwargs)
        json_output = pydantic_core.to_json(output).decode()
        tokens = _tiktoken_encoding.encode(json_output)
        if len(tokens) > 10_000:
            logfire.warn('Output truncated, {total_tokens=}', total_tokens=len(tokens))
            truncated = _tiktoken_encoding.decode(tokens[:10_000])
            return f'{truncated}... (WARNING: output truncated to 10k tokens, total tokens: {len(tokens)})'
        else:
            return output


toolset = FunctionToolset(tools=[get_html, record_model_info, *tools])
prices_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-5',
    toolsets=[TruncateCodeExecutionToolset(toolset=toolset)],
    output_type=OutputData,
    deps_type=RunDeps,
    instructions="""
Get structured information including pricing data for all models from the URL provided.

The HTML returned from this URL is too big for context, so make sure to process it with beautiful_soup
or return a small snippet of the HTML to process.

Ignore any deprecated models.

Do not use `print` in code, you can't see the output.

You should record information about each model by calling `record_model_info`.
""",
)

urls = {
    'openai': 'https://developers.openai.com/api/docs/pricing',
    'anthropic': 'https://platform.claude.com/docs/en/about-claude/pricing',
    'groq': 'https://groq.com/pricing',
}


@prices_agent.instructions
async def add_optimal_code(ctx: RunContext[RunDeps]) -> str | None:
    if ctx.deps.previous_code is not None:
        return f'Optimal code from previous run:\n\n```python\n{ctx.deps.previous_code}\n```'


async def get_prices(provider: str, allow_code_reuse: bool = False) -> dict[str, ModelInfo]:
    prev_code = Path(f'{provider}_previous_code.py')
    previous_code = None
    if allow_code_reuse and prev_code.exists():
        previous_code = prev_code.read_text()

    with logfire.span(
        'getting prices for {provider} {existing_code=}',
        provider=provider,
        existing_code=previous_code is not None,
    ):
        deps = RunDeps(previous_code=previous_code)
        r = await prices_agent.run(urls[provider], deps=deps)

        if not prev_code.exists():
            prev_code.write_text(r.output.optimal_code)

        logfire.info('{models=}', models=deps.models)
        prices = Path(f'{provider}_prices.json')
        if not prices.exists():
            prices.write_bytes(pydantic_core.to_json(deps.models, indent=2))
        return deps.models


if __name__ == '__main__':
    logfire.configure(service_name='llm-prices-run')
    logfire.instrument_pydantic_ai()
    models = asyncio.run(get_prices('anthropic', allow_code_reuse=True))
    debug(models)
