import logfire

from .agent import agent

logfire.configure(service_name='ecommerce_example')
logfire.instrument_pydantic_ai()

# result = agent.run_sync('Which customer segment drives the most revenue, broken down by region?')
result = agent.run_sync('Which acquisition channel brings in the highest-value customers (by spend)?')
logfire.info(f'{result=}')
