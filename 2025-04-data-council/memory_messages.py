from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import asyncpg
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

# hack to get around asyncpg's poor typing support
if TYPE_CHECKING:
    DbConn = asyncpg.Connection[asyncpg.Record]
else:
    DbConn = asyncpg.Connection


import logfire

logfire.configure(service_name='mem-msgs')
logfire.instrument_pydantic_ai()
logfire.instrument_asyncpg()


@asynccontextmanager
async def db() -> AsyncIterator[DbConn]:
    conn = await asyncpg.connect('postgresql://postgres@localhost:5432')
    await conn.execute("""
        create table if not exists messages(
            id serial primary key,
            ts timestamp not null default now(),
            user_id integer not null,
            messages json not null
        )
    """)

    try:
        yield conn
    finally:
        await conn.close()


@dataclass
class Deps:
    user_id: int
    conn: DbConn


agent = Agent(
    'openai:gpt-4o',
    instructions='You are a helpful assistant.',
)


@logfire.instrument
async def run_agent(prompt: str, user_id: int):
    async with db() as conn:
        with logfire.span('retrieve messages'):
            messages: list[ModelMessage] = []
            for row in await conn.fetch('SELECT messages FROM messages WHERE user_id = $1 order by ts', user_id):
                messages += ModelMessagesTypeAdapter.validate_json(row[0])

        result = await agent.run(prompt, message_history=messages)
        print(result.output)

        with logfire.span('record messages'):
            msgs = result.new_messages_json().decode()
            await conn.execute('INSERT INTO messages(user_id, messages) VALUES($1, $2)', user_id, msgs)


@logfire.instrument
async def memory_messages():
    await run_agent('My name is Samuel.', 123)

    await run_agent('What is my name?', 123)


if __name__ == '__main__':
    asyncio.run(memory_messages())
