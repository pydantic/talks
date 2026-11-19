import asyncio

import logfire


def get_name() -> str:
    return 'world'


async def main() -> None:
    async with logfire.agent_sandboxes() as pool:
        async with pool.checkout() as session:
            await session.feed_run(
                'print("hello", get_name())',
                external_lookup={'get_name': get_name},
            )


if __name__ == '__main__':
    asyncio.run(main())
