from __future__ import annotations

import asyncio

import logfire
from playwright.async_api import async_playwright
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


PRICES_YAML = 'anthropic.yml'


def read_prices_yaml(start_line: int | None = None, end_line: int | None = None) -> str:
    """Read the current anthropic model prices data from a YAML file.

    This file might be large, so use the `start_line` and `end_line` parameters
    to read only a portion of the file.

    Args:
        start_line: 1-indexed first line to return. Defaults to the start of the file.
        end_line: 1-indexed last line to return (inclusive). Defaults to the end of the file.

    Returns:
        The file contents (or the requested slice).
    """
    with open(PRICES_YAML) as f:
        lines = f.readlines()
    if start_line is not None or end_line is not None:
        lines = lines[(start_line or 1) - 1 : end_line]
    return ''.join(lines)


def update_prices_yaml(old: str, new: str) -> str:
    """Update the anthropic model prices YAML file by replacing the first occurrence of a string.

    Args:
        old: The string to find in the file.
        new: The string to replace it with.

    Returns:
        A confirmation message, or an error if `old` was not found.
    """
    with open(PRICES_YAML) as f:
        content = f.read()
    if old not in content:
        return f'Error: string not found in {PRICES_YAML}'
    content = content.replace(old, new, 1)
    with open(PRICES_YAML, 'w') as f:
        f.write(content)
    return f'Successfully updated {PRICES_YAML}'


toolset = FunctionToolset(tools=[get_html, read_prices_yaml, update_prices_yaml, *tools])


agent = Agent(
    'gateway/anthropic:claude-opus-4-6',
    builtin_tools=[WebSearchTool()],
    toolsets=[CodeExecutionToolset(toolset=toolset)],
)

# Get prices for all anthropic models in markdown.
prompt = """
Get pricing data for all anthropic models from https://platform.claude.com/docs/en/about-claude/pricing.

Then update the yaml file with the new pricing data.

Return instructions on how best to query the HTML for future runs.

NOTE: the HTML returned from this URL is too big for context, so make sure to process it with beautiful_soup
or return a small snippet of the HTML to process.
"""


async def main():
    r = await agent.run(prompt)
    print(r.output)


if __name__ == '__main__':
    asyncio.run(main())
