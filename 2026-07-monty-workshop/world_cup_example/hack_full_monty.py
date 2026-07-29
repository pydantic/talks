import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import AsyncContextManager, Literal, overload

import httpx2
from dotenv import load_dotenv
from pydantic_monty import AsyncMontySession, AsyncMontyWebsocket

load_dotenv()

__all__ = 'sandbox', 'run_code_instructions'


@overload
def sandbox(
    *,
    provider: Literal['modal'],
    dependencies: list[str] | None = None,
    type_check: bool = False,
    type_check_stubs: str | None = None,
) -> AsyncContextManager[AsyncMontySession]: ...


@overload
def sandbox(
    *, provider: Literal['monty'], type_check: bool = False, type_check_stubs: str | None = None
) -> AsyncContextManager[AsyncMontySession]: ...


@overload
def sandbox(
    *,
    provider: Literal['modal', 'monty'],
    dependencies: list[str] | None = None,
    type_check: bool = False,
    type_check_stubs: str | None = None,
) -> AsyncContextManager[AsyncMontySession]: ...


@asynccontextmanager
async def sandbox(
    *,
    provider: Literal['modal', 'monty'],
    dependencies: list[str] | None = None,
    type_check: bool = False,
    type_check_stubs: str | None = None,
) -> AsyncIterator[AsyncMontySession]:
    if provider == 'modal':
        ws_url = await _create_modal_sandbox()

    else:
        assert provider == 'monty'
        assert dependencies is None, 'dependencies are not supported for monty provider'
        ws_url = 'ws://localhost:8000'

    # `install_dependencies` is a worker turn too, and a real `uv pip install` easily
    # exceeds the default 10s request timeout -- allow much longer when installing.
    request_timeout = 300.0 if dependencies else 10.0
    async with AsyncMontyWebsocket(ws_url, request_timeout=request_timeout) as pool:
        async with pool.checkout(type_check=type_check, type_check_stubs=type_check_stubs) as session:
            if dependencies:
                await session.install_dependencies(dependencies)
            yield session


def run_code_instructions(provider: Literal['modal', 'monty'], dependencies: list[str] | None = None) -> str | None:
    """Provider-specific base prose for the `run_code` tool description.

    Returns `None` for `monty`, where the harness's default description (Monty subset
    of Python, stdlib allowlist) is already accurate. For `modal` returns replacement
    prose describing the full-CPython worker and its installed packages.
    """
    if provider == 'monty':
        return None
    if dependencies:
        packages = ', '.join(f'`{d}`' for d in dependencies)
        packages_line = f'- **Installed third-party packages**: {packages}. Other packages are not installed.'
    else:
        packages_line = '- **No third-party packages are installed**: only the standard library is available.'
    return f"""\
Write and run Python code in a sandboxed environment.

The sandbox runs full CPython in an isolated container. Key notes:
- **Full standard library**: all standard library modules are available and behave normally.
{packages_line}
- **Host filesystem access**: you must import `Path` from `host_pathlib` to write files the host can see."""


async def _create_modal_sandbox() -> str:
    async with httpx2.AsyncClient() as client:
        r = await client.post(
            'https://samuel-1--sandbox-control-web.modal.run/sandboxes',
            json={
                'timeout': 60,
                'image_version': '2afdf0ea',  # v0.0.19 with host_pathlib
                'call_direct': True,
            },
            headers={'authorization': f'Bearer {os.environ["MONTY_SANDBOX_TOKEN"]}'},
            timeout=30,
        )
        r.raise_for_status()
        ws_url = r.json()['parent_url']
    return ws_url
