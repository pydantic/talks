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
from pydantic_ai import Agent
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.code_execution import CodeExecutionToolset
from pydantic_ai.toolsets.function import FunctionToolset

from bs import tools
from sub_agent import ModelResults, record_model_info

logfire.configure()
logfire.instrument_pydantic_ai()


async def get_html(url: str) -> str:
    """Get the HTML content of a URL."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until='networkidle')
        html = await page.content()
        await browser.close()
    return html


toolset = FunctionToolset(tools=[get_html, record_model_info, *tools])


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


prices_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-5',
    toolsets=[TruncateCodeExecutionToolset(toolset=toolset)],
    output_type=OutputData,
    deps_type=ModelResults,
    instructions="""
Get structured information including pricing data for all models from the URL provided

The HTML returned from this URL is too big for context, so make sure to process it with beautiful_soup
or return a small snippet of the HTML to process.

Do not use `print` in code, you can't see the output.

You should record information about each model by calling `record_model_info`.
""",
)

urls = {
    'openai': 'https://developers.openai.com/api/docs/pricing',
    'anthropic': 'https://platform.claude.com/docs/en/about-claude/pricing',
}
provider = 'anthropic'
previous_code_file = Path(f'{provider}_previous_code.py')


@prices_agent.instructions
async def add_optimal_code() -> str | None:
    if previous_code_file.exists():
        code = previous_code_file.read_text()
        return f'Optimal code from previous run:\n\n```python\n{code}\n```'


async def main():
    with logfire.span(
        'getting prices for {provider} {existing_code=}', provider=provider, existing_code=previous_code_file.exists()
    ):
        model_results = ModelResults()
        r = await prices_agent.run(urls[provider], deps=model_results)
        debug(r.output, model_results)
        previous_code_file.write_text(r.output.optimal_code)
        Path(f'{provider}_prices.json').write_bytes(pydantic_core.to_json(model_results.models, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
