# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic-monty>=1.4"]
# ///
from pydantic_monty import Monty


def get_weather(city: str) -> str:
    return f'raining in {city}'


code = """
weather = get_weather('London')
f'Hello World, it is {weather}'
"""

with Monty() as monty:
    with monty.checkout() as session:
        output = session.feed_run(code, external_lookup={'get_weather': get_weather})
        print(output)
