import asyncio
from pathlib import Path
from textwrap import dedent
from typing import Annotated

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Configure logfire instrumentation
logfire.configure(scrubbing=False, service_name='playwright-browser')
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

class MCPBotResponse(BaseModel):
    answer: str
    reasoning: str
    services_used: list[str] = []
    confidence_percentage: Annotated[int, Field(ge=0, le=100)]

SYSTEM_PROMPT = dedent(
    """
    You're a helpful AI assistant with access to browser automation and Linear project management capabilities.
    
    Browser capabilities (via Playwright):
    - Navigate to websites, interact with web pages, take screenshots, and extract information
    - Be thorough in web navigation and information extraction
    - Take screenshots when helpful for verification
    - Extract relevant information clearly and accurately
    
    Linear capabilities:
    - Find, create, and update Linear issues, projects, and comments
    - Access Linear workspace data and project management information
    - Help with issue tracking and project organization
    
    When working with these services:
    - Explain what you're doing clearly
    - Be mindful of website terms of service and respectful browsing practices
    - Follow best practices for project management workflows
    
    Give a confidence percentage for your answer, from 0 to 100.
    List any services you used (e.g., "playwright", "linear") in the services_used field.
    """
)

# Set up MCP servers
browser_mcp = MCPServerStdio('npx', args=['-Y', '@playwright/mcp@latest'], tool_prefix='browser')
linear_mcp = MCPServerStdio('npx', args=['-y', 'mcp-remote', 'https://mcp.linear.app/sse'], tool_prefix='linear')

# Create the agent with both MCP servers
agent = Agent(
    'openai:gpt-4o',
    output_type=MCPBotResponse,
    system_prompt=SYSTEM_PROMPT,
    mcp_servers=[browser_mcp, linear_mcp],
    instrument=True,
)

async def answer_mcp_question(question: str) -> MCPBotResponse:
    """Run a question through the MCP-enabled browser agent."""
    async with agent.run_mcp_servers():
        result = await agent.run(user_prompt=question)
        return result.output

async def main():
    """Example usage of the browser and Linear agent."""
    async with agent.run_mcp_servers():
        result = await agent.run(
            'Help me with project management: First, check what Linear workspaces and projects are available. '
            'Then navigate to pydantic.dev to get information about their latest announcement and '
            'create a Linear issue to track following up on any interesting developments you find.'
        )
    print(result.output)

if __name__ == '__main__':
    asyncio.run(main()) 