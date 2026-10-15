"""Pending spans: run this while watching the live view, the trace appears before it finishes.

Eight nested spans, one second apart, all in a single trace.
"""

import time
from datetime import UTC, datetime

import logfire
from pydantic import BaseModel


class User(BaseModel):
    id: int
    email: str
    signed_up: datetime


logfire.configure(service_name='hello-world')

user = User(id=1, email='x@y.z', signed_up=datetime(2026, 1, 1, 9, 30, tzinfo=UTC))

with logfire.span('hello {name}', name=user):
    for step in range(1, 8):
        with logfire.span('step {step} of 7', step=step):
            time.sleep(1)
    logfire.info('done after {steps} steps', steps=7)
