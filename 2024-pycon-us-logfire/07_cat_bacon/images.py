from datetime import datetime, timedelta
from typing import Annotated

from asyncpg import Connection
from openai import AsyncClient

from pydantic import BaseModel, Field, TypeAdapter, BeforeValidator


class Image(BaseModel):
    # model_config = config_record
    id: int = Field(title='ID')
    ts: datetime = Field(title='Timestamp')
    prompt: str = Field(title='Prompt')
    url: str = Field(title='URL')


images_adapter = TypeAdapter(list[Annotated[Image, BeforeValidator(dict)]])
PAGE_LIMIT = 50


async def list_images(conn: Connection, page: int) -> tuple[list[Image], int]:
    offset = (page - 1) * PAGE_LIMIT
    rows = await conn.fetch('SELECT * FROM images ORDER BY ts desc OFFSET $1 LIMIT $2', offset, PAGE_LIMIT)
    # ids = await conn.fetch('SELECT id FROM images ORDER BY ts desc OFFSET $1 LIMIT $2', offset, PAGE_LIMIT)
    # rows = []
    # for row in ids:
    #     rows.append(await conn.fetchrow('SELECT * FROM images where id=$1', row['id']))
    images = images_adapter.validate_python(rows)
    total = await conn.fetchval('SELECT COUNT(*) FROM images')
    return images, total


async def get_image(conn: Connection, image_id: int) -> Image:
    row = await conn.fetchrow('SELECT ts, prompt, url from images where id=$1', image_id)
    return Image(id=image_id, **row)


async def create_image(conn: Connection, openai_client: AsyncClient, animal: str) -> int:
    prompt = f'Create an image of a {animal} in the style of Francis Bacon'
    response = await openai_client.images.generate(prompt=prompt, model='dall-e-3')
    url = response.data[0].url
    return await conn.fetchval('INSERT INTO images (prompt, url) VALUES ($1, $2) RETURNING id', prompt, url)


async def create_images(conn: Connection) -> None:
    image_count = await conn.fetchval('SELECT count(*) from images')
    if image_count > 100_000:
        return

    ts = datetime(2024, 1, 1)
    images = []
    for _ in range(100_000):
        ts = ts + timedelta(seconds=1)
        images.append((ts, 'test data', 'https://example.com'))
    await conn.executemany(f'INSERT INTO images (ts, prompt, url) VALUES ($1, $2, $3)', images)
