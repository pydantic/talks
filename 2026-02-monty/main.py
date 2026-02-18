from __future__ import annotations

import asyncio
from typing import Any

import logfire
import pydantic_core
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
    optimial_code: str
    """Optimial python code for getting prices data, to use in future runs.

    This should NOT contain any model names or model-specific details, just the code needed to get and extract
    the data.
    """


class TruncateCodeExecutionToolset(CodeExecutionToolset[AgentDepsT]):
    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        output = await super().call_tool(*args, **kwargs)
        json_output = pydantic_core.to_json(output)
        if len(json_output) > 5_000:
            logfire.warn('Output truncated, {total_length=}', total_length=len(json_output))
            return f'{json_output[:5_000]}... (WARNING: output truncated, total length: {len(json_output)})'
        else:
            return output


agent = Agent(
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

openai_url = 'https://developers.openai.com/api/docs/pricing'
anthropic_url = 'https://platform.claude.com/docs/en/about-claude/pricing'


async def main():
    model_results = ModelResults()
    r = await agent.run(anthropic_url, deps=model_results)
    debug(r.output, model_results)


if __name__ == '__main__':
    asyncio.run(main())
