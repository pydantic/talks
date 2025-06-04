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
    websites_accessed: list[str] = []
    confidence_percentage: Annotated[int, Field(ge=0, le=100)]

SYSTEM_PROMPT = dedent(
    """
    You're a helpful AI assistant with access to browser automation capabilities through Playwright.
    You can navigate to websites, interact with web pages, take screenshots, and extract information.
    
    When working with web pages:
    - Be thorough in your web navigation and information extraction
    - Take screenshots when helpful for verification
    - Extract relevant information clearly and accurately
    - Explain what you're doing with the browser
    - Be mindful of website terms of service and respectful browsing practices
    
    Give a confidence percentage for your answer, from 0 to 100.
    List any websites you accessed in the websites_accessed field.
    """
)

# Set up Playwright MCP server
browser_mcp = MCPServerStdio('npx', args=['-Y', '@playwright/mcp@latest'])

# Create the agent with MCP server integration
agent = Agent(
    'openai:gpt-4o',
    output_type=MCPBotResponse,
    system_prompt=SYSTEM_PROMPT,
    mcp_servers=[browser_mcp],
    instrument=True,
)

async def answer_mcp_question(question: str) -> MCPBotResponse:
    """Run a question through the MCP-enabled browser agent."""
    async with agent.run_mcp_servers():
        result = await agent.run(user_prompt=question)
        return result.output

async def main():
    """Example usage of the browser agent."""
    async with agent.run_mcp_servers():
        result = await agent.run(
            'Navigate to pydantic.dev and get information about their latest blog post or announcement. '
            'Summarize what you find.'
        )
    print(result.output)

if __name__ == '__main__':
    asyncio.run(main()) 