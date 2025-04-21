import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Self, Annotated
from urllib.parse import urlparse

import logfire
from fastapi import Request, Depends

import asyncpg
from asyncpg.connection import Connection

__all__ = ('Database',)


@dataclass
class _Database:
    """
    Wrapper for asyncpg with some utilities, also usable as a fastapi dependency.
    """

    _pool: asyncpg.Pool

    @classmethod
    @asynccontextmanager
    async def create(cls, dsn: str, prepare_db: bool = False, create_database: bool = False) -> AsyncIterator[Self]:
        if prepare_db:
            await _prepare_db(dsn, create_database)
        pool = await asyncpg.create_pool(dsn)
        try:
            yield cls(_pool=pool)
        finally:
            await asyncio.wait_for(pool.close(), timeout=2.0)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Connection]:
        con = await self._pool.acquire()
        try:
            yield con
        finally:
            await self._pool.release(con)

    @asynccontextmanager
    async def acquire_trans(self) -> AsyncIterator[Connection]:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn


def _get_db(request: Request) -> _Database:
    return request.app.state.db


Database = Annotated[_Database, Depends(_get_db)]


@logfire.instrument()
async def _prepare_db(dsn: str, create_database: bool) -> None:
    x = {'foobar': 123, 'baz': 'qux'}
    logfire.info(f'Preparing database {x}')
    if create_database:
        parse_result = urlparse(dsn)
        database = parse_result.path.lstrip('/')
        server_dsn = dsn[: dsn.rindex('/')]
        conn = await asyncpg.connect(server_dsn)
        try:
            db_exists = await conn.fetchval('SELECT 1 FROM pg_database WHERE datname = $1', database)
            if not db_exists:
                await conn.execute(f'CREATE DATABASE {database}')
        finally:
            await conn.close()

    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction():
            await _create_schema(conn)
    finally:
        await conn.close()


async def _create_schema(conn: Connection) -> None:
    await conn.execute("""
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP NOT NULL DEFAULT NOW(),
    prompt TEXT NOT NULL,
    url TEXT NOT NULL
);
-- CREATE INDEX IF NOT EXISTS images_ts_idx ON images (ts DESC);
""")
    from .images import create_images

    await create_images(conn)
