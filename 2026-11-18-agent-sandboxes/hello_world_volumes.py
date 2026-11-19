import asyncio

import logfire

# the ID of a sandbox profile with a volume mounted at /data
PROFILE = '01a1ef98-405d-7384-b76c-8b576c8df66d'

write = """
from pathlib import Path

Path('/data/hello.txt').write_text('hello from the first session')
"""

read = "Path('/data/hello.txt').read_text()"


async def main() -> None:
    async with logfire.agent_sandboxes() as pool:
        async with pool.checkout(profile=PROFILE) as session:
            print('writing to file')
            await session.feed_run(write)

            print('reading from volume')
            output = await session.feed_run(read)
            print(output)


if __name__ == '__main__':
    asyncio.run(main())
