from __future__ import annotations

import asyncio

import logfire
from devtools import debug
from playwright.async_api import async_playwright
from pydantic import BaseModel
from pydantic_ai import Agent, WebSearchTool
from pydantic_ai.toolsets.code_execution import CodeExecutionToolset
from pydantic_ai.toolsets.function import FunctionToolset

from bs import tools

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


toolset = FunctionToolset(tools=[get_html, *tools])


class ModelInfo(BaseModel, use_attribute_docstrings=True):
    name: str
    """Name of the model."""
    description: str | None = None
    """Description of the model."""
    prices: dict[str, float]
    """Consumption prices per million tokens, e.g. "input tokens", "output tokens", "cached tokens" etc."""
    attributes: dict[str, float | str]
    """Any other attributes of the model."""


class OutputData(BaseModel, use_attribute_docstrings=True):
    models: list[ModelInfo]
    """List of models with information including prices."""
    run_summary: str
    """Summary of how you performed the task and any issues encountered."""
    optimial_code: str
    """Optimial python code for getting prices data, to use in future runs."""


agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    builtin_tools=[WebSearchTool()],
    toolsets=[CodeExecutionToolset(toolset=toolset)],
    output_type=OutputData,
)

# Get prices for all anthropic models in markdown.
prompt = """
Get structured information including pricing data for all anthropic models from

https://platform.claude.com/docs/en/about-claude/pricing.

NOTE: the HTML returned from this URL is too big for context, so make sure to process it with beautiful_soup
or return a small snippet of the HTML to process.

Do not use `print` in code, you can't see the output.
"""


async def main():
    r = await agent.run(prompt)
    debug(r.output)


if __name__ == '__main__':
    asyncio.run(main())
