import logfire

from .agent import agent

logfire.configure(service_name='world_cup_example')
logfire.instrument_pydantic_ai()

agent.to_cli_sync()
