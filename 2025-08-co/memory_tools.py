from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import asyncpg
from pydantic_ai import Agent, RunContext

# hack to get around asyncpg's poor typing support
if TYPE_CHECKING:
    DbConn = asyncpg.Connection[asyncpg.Record]
else:
    DbConn = asyncpg.Connection


import logfire

logfire.configure(service_name='mem-tool')
logfire.instrument_pydantic_ai()
logfire.instrument_asyncpg()
# run postgres with:
# `docker run -e POSTGRES_HOST_AUTH_METHOD=trust --rm -it --name pg -p 5432:5432 -d postgres`


@asynccontextmanager
async def db(reset: bool = False) -> AsyncIterator[DbConn]:
    conn = await asyncpg.connect('postgresql://postgres@localhost:5432')
    if reset:
        await conn.execute('drop table if exists memory')
    await conn.execute("""
        create table if not exists memory(
            id serial primary key,
            user_id integer not null,
            value text not null,
            unique(user_id, value)
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
    deps_type=Deps,
    instructions='You are a helpful assistant.',
)


@agent.tool
async def record_memory(ctx: RunContext[Deps], value: str) -> str:
    """Use this tool to store information in memory."""
    await ctx.deps.conn.execute(
        'insert into memory(user_id, value) values($1, $2) on conflict do nothing',
        ctx.deps.user_id,
        value,
    )
    return 'Value added to memory.'


@agent.tool
async def retrieve_memories(ctx: RunContext[Deps], memory_contains: str) -> str:
    """Get all memories about the user."""
    rows = await ctx.deps.conn.fetch(
        'select value from memory where user_id = $1 and value ilike $2',
        ctx.deps.user_id,
        f'%{memory_contains}%',
    )
    return '\n'.join(row[0] for row in rows)


async def memory_tools():
    async with db(True) as conn:
        deps = Deps(123, conn)
        result = await agent.run('My name is Samuel.', deps=deps)
        print(result.output)

    # time goes by...

    async with db() as conn:
        deps = Deps(123, conn)
        result = await agent.run('What is my name?', deps=deps)
        print(result.output)


if __name__ == '__main__':
    asyncio.run(memory_tools())
