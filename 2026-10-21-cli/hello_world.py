"""Message templates and Python-looking values."""

from datetime import UTC, datetime

import logfire
from pydantic import BaseModel


class User(BaseModel):
    id: int
    email: str
    signed_up: datetime


logfire.configure(service_name='hello-world')

user = User(id=1, email='x@y.z', signed_up=datetime(2026, 1, 1, 9, 30, tzinfo=UTC))

logfire.info('hello {name}', name=user)

with logfire.span('checking {user.email}', user=user):
    logfire.info(f'user {user.id} has been around for {datetime.now(UTC) - user.signed_up}')
