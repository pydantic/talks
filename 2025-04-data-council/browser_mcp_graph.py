from __future__ import annotations as _
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_graph import Graph, BaseNode, GraphRunContext, End

import logfire

logfire.configure(scrubbing=False, service_name='browse-graph')
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

server = MCPServerStdio(
    'npx',
    args=[
        '-Y',
        '@playwright/mcp@latest',
    ]
)

browser_agent = Agent(
    'anthropic:claude-3-7-sonnet-latest',
    mcp_servers=[server],
    system_prompt='Find the page requested by the user and return the URL only. Nothing else.'
)


@dataclass
class FindBlog(BaseNode):
    url: str

    async def run(self, ctx: GraphRunContext) -> FindLatestPosts:
        result = await browser_agent.run(f"Find the page with a list of blog posts at {self.url}.")
        return FindLatestPosts(result.output)


@dataclass
class FindLatestPosts(BaseNode):
    url: str

    async def run(self, ctx: GraphRunContext) -> SummariesContent:
        result = await browser_agent.run(f"Find the latest blog post at {self.url}")
        return SummariesContent(result.output)


summary_agent = Agent(
    'anthropic:claude-3-7-sonnet-latest',
    system_prompt='Summarise the content of the blog post page as markdown'
)

@dataclass
class SummariesContent(BaseNode[None, None, str]):
    content: str

    async def run(self, ctx: GraphRunContext) -> End[str]:
        result = await summary_agent.run(self.content)
        return End(result.output)


graph = Graph(nodes=[FindBlog, FindLatestPosts, SummariesContent])

async def main():
    async with browser_agent.run_mcp_servers():
        result = await graph.run(FindBlog(url='pydantic.dev'))
        print(result.output)


if __name__ == '__main__':
    with open('browser.mermaid', 'w') as f:
        f.write(graph.mermaid_code())
    # import asyncio
    # asyncio.run(main())
