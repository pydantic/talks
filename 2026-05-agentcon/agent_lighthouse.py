"""Agent Lighthouse - score how "agent-ready" a website is.

Run:    uv run agent_lighthouse.py pydantic.dev

Extra deps:  uv add httpx dnspython
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dns.asyncresolver
import httpx
import logfire
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness import CodeMode
from rich.console import Console
from rich.markdown import Markdown
from typing_extensions import TypedDict

logfire.configure(service_name='agent-lighthouse')
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()


class FetchResult(TypedDict):
    url: str
    status: int
    final_url: str
    content_type: str
    body: str
    headers: dict[str, str]
    error: str | None


@dataclass
class Deps:
    http: httpx.AsyncClient


auditor = Agent(
    'gateway/openai:gpt-5.4-mini',
    deps_type=Deps,
    # capabilities=[CodeMode(max_retries=10)],
    instructions=(Path(__file__).parent / 'lighthouse_instructions.md').read_text(),
)


@auditor.tool
async def fetch(ctx: RunContext[Deps], url: str, accept: str | None = None) -> FetchResult:
    """HTTP GET a URL.

    Args:
        url: full URL.  http(s) only.
        accept: optional Accept header value (e.g. 'text/markdown').

    Returns a FetchResult.  On error, `status` is 0 and `error` is set.
    The body is truncated to 100 KB.  Header keys are lowercased.
    """
    headers: dict[str, str] = {}
    if accept:
        headers['Accept'] = accept
    try:
        r = await ctx.deps.http.get(url, headers=headers)
    except Exception as e:
        return FetchResult(
            url=url,
            status=0,
            final_url=url,
            content_type='',
            body='',
            headers={},
            error=f'{type(e).__name__}: {e}',
        )
    return FetchResult(
        url=url,
        status=r.status_code,
        final_url=str(r.url),
        content_type=r.headers.get('content-type', ''),
        body=r.text[:100_000],
        headers={k.lower(): v for k, v in r.headers.items()},
        error=None,
    )


RecordType = Literal['A', 'AAAA', 'MX', 'TXT', 'CNAME', 'NS']


@auditor.tool_plain
async def dns_lookup(domain: str, record_type: RecordType) -> list[str]:
    """DNS lookup for a domain.

    Returns each record as a string.  Empty list on NXDOMAIN, timeout, or no
    records of that type.  TXT records are joined and decoded as utf-8.
    """
    try:
        answer = await dns.asyncresolver.resolve(domain, record_type, lifetime=5.0)
    except Exception:
        return []
    if record_type == 'TXT':
        return [b''.join(rd.strings).decode('utf-8', 'replace') for rd in answer]
    return [str(rd) for rd in answer]


async def main(domain: str) -> None:
    prompt = f'Audit the agent-readiness of `{domain}` and produce the markdown report.'
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=10.0,
        headers={'User-Agent': 'pydantic-agent-lighthouse/0.1'},
    ) as http:
        with logfire.span('audit', domain=domain):
            result = await auditor.run(prompt, deps=Deps(http=http))
    Console().print(Markdown(result.output))


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else 'pydantic.dev'))
