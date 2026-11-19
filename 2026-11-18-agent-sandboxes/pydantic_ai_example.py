import logfire
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai_harness import CodeMode

logfire.configure(service_name='pydantic-ai-example')
logfire.instrument_pydantic_ai()

agent = Agent(
    'gateway/openai:gpt-6.1-sol',
    toolsets=[MCPToolset('https://teamchat-mcp.pydantic.workers.dev/mcp')],
    capabilities=[CodeMode(monty_sandbox=logfire.agent_sandboxes)],
)
result = agent.run_sync("""\
In the last 14 days, which threads in #help-* channels went more than 4 hours
without a reply from anyone on the Platform team? Group them by channel, show who
eventually answered, and say when the slowdown started.
""")
print(result.output)
