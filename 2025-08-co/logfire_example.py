from dataclasses import dataclass
from time import sleep

import logfire

logfire.configure(service_name='logfire-example')


@dataclass
class Place:
    name: str
    population: int


logfire.info('hello {place=}', place=Place('World', 8_336_817_000))

with logfire.span('this records {thing}', thing='duration'):
    sleep(1)
    with logfire.span('can be nested'):
        sleep(1)
