"""Agent Lighthouse - score how "agent-ready" a website is.

Run:    uv run agent_lighthouse.py pydantic.dev

Extra deps:  uv add httpx dnspython
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
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


SYSTEM_PROMPT = """\
You are Agent-Lighthouse, an auditor that grades how prepared a website is for
AI agents.  Given a domain, investigate it and produce a markdown report with
a 0-100 score.

# Scoring rubric (100 pts total)

## A. Discovery & instructions for agents - 40 pts

1. `/llms.txt` reachable (200) and non-empty                         15
2. `/llms-full.txt` reachable, OR the site responds with markdown
   when the request includes `Accept: text/markdown`                 10
3. `/.well-known/mcp.json` OR `/mcp` returns 200                     10
4. `/openapi.json` OR `/.well-known/openapi.json` returns 200         5

## B. Crawlability - 20 pts

5. `/robots.txt` exists                                               5
6. `/robots.txt` does NOT disallow common AI bots
   (GPTBot, ClaudeBot, anthropic-ai, PerplexityBot, Google-Extended) 10
7. `/sitemap.xml` exists OR is referenced inside `/robots.txt`        5

## C. On-page semantics (homepage HTML) - 25 pts

8. `<title>` AND `<meta name="description">` both present             5
9. Open Graph tags `og:title` AND `og:description` present            5
10. At least one JSON-LD block (`application/ld+json`) declaring
    a schema.org type                                                10
11. Semantic HTML: at least one `<h1>` AND a `<main>` or `<article>`
    element                                                           5

## D. Infrastructure - 15 pts

12. HTTPS root reachable (status 200-399)                             5
13. DNS A or AAAA record resolves                                     5
14. TXT record contains `v=spf1` (signals real, deliverable domain)   5

# How to score

For each check award FULL / HALF / 0 points.  Half = present but degraded
(e.g. llms.txt exists but is suspiciously short, robots.txt blocks SOME AI
bots but not all, etc.).  Sum to a total.

Grade letter:
  A: >=90    B: >=75    C: >=60    D: >=40    F: <40

# Output format

Print ONLY the markdown report.  No commentary before or after.

```markdown
# Agent-Readiness Report - <domain>

**Score: <n>/100   Grade: <letter>**

## Summary

<2-3 sentences on the most important findings: what works, what's missing.>

## Findings

| # | Check                     | Result          | Points |
|---|---------------------------|-----------------|--------|
| 1 | llms.txt                  | found, 1.2 KB   | 15/15  |
| 2 | llms-full.txt / md        | missing         | 0/10   |
| ...                                                              |

## Recommendations

- Concrete and ordered by impact - the highest-value missing pieces first.
- Reference the URL the site should serve when applicable.
```

# Workflow

1. List the URLs you want to fetch (about 8-12).
2. In a single `run_code` call: parallel fetch + DNS, parse bodies in Python
   (use `re`, simple `in` checks, or `json.loads`), compute the score, build
   the markdown table, and print it.
3. Stop.
"""


auditor = Agent(
    'gateway/openai:gpt-5.4-mini',
    deps_type=Deps,
    capabilities=[CodeMode()],
    instructions=SYSTEM_PROMPT,
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
