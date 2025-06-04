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
    You're a helpful AI assistant with access to browser automation and Logfire telemetry analysis capabilities.
    
    Browser capabilities (via Playwright):
    - Navigate to websites, interact with web pages, take screenshots, and extract information
    - Be thorough in web navigation and information extraction
    - Take screenshots when helpful for verification
    - Extract relevant information clearly and accurately
    
    Logfire capabilities:
    - Find and analyze exceptions in OpenTelemetry traces grouped by file
    - Get detailed trace information about exceptions in specific files
    - Run custom SQL queries on traces and metrics data
    - Access OpenTelemetry schema information for query building
    - Analyze application performance and error patterns
    
    When working with these services:
    - Explain what you're doing clearly
    - Be mindful of website terms of service and respectful browsing practices
    - Use appropriate time ranges for telemetry queries (max 7 days)
    - Help identify patterns in application behavior and errors
    
    Give a confidence percentage for your answer, from 0 to 100.
    List any services you used (e.g., "playwright", "logfire") in the services_used field.
    """
)

# Set up MCP servers with correct command syntax
browser_mcp = MCPServerStdio('npx', args=['--yes', '@playwright/mcp@latest'], tool_prefix='browser')
logfire_mcp = MCPServerStdio('uvx', args=['logfire-mcp'], tool_prefix='logfire')

# Create the agent with both MCP servers
agent = Agent(
    'openai:gpt-4o',
    output_type=MCPBotResponse,
    system_prompt=SYSTEM_PROMPT,
    mcp_servers=[browser_mcp, logfire_mcp],
    instrument=True,
)

async def answer_mcp_question(question: str) -> MCPBotResponse:
    """Run a question through the MCP-enabled agent."""
    async with agent.run_mcp_servers():
        result = await agent.run(user_prompt=question)
        return result.output

async def main():
    """Example usage of the browser and Logfire telemetry agent."""
    question = ('Help me analyze my application: First, check for any exceptions in traces from the last hour using Logfire. '
               'Then navigate to the Logfire documentation to get information about best practices for error monitoring. '
               'Finally, provide recommendations based on what you find.')
    
    result = await answer_mcp_question(question)
    print(result)

if __name__ == '__main__':
    asyncio.run(main()) 