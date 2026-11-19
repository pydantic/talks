"""The web UI: a chat on the left and a map on the right.

Run with `make app` (`uv run uvicorn flood_risk_example.app:app --port 8000`), with the geo risk service
already running (`make serve`). Set `FLOOD_SANDBOX=1` to run the agent's code in Logfire Agent Sandboxes
and `FLOOD_ONLINE_EVALS=1` to evaluate every run.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
import logfire
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from .agent import build_agent
from .tools import Deps, http_client

STATIC_DIR = Path(__file__).parent / 'static'
SANDBOX = os.getenv('FLOOD_SANDBOX') == '1'
ONLINE_EVALS = os.getenv('FLOOD_ONLINE_EVALS') == '1'

logfire.configure(service_name='flood-risk-app')
logfire.instrument_pydantic_ai()
logfire.instrument_monty()
logfire.instrument_httpx()

agent = build_agent(sandbox=SANDBOX, online_evals=ONLINE_EVALS)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with http_client() as http:
        app.state.http = http
        # One conversation, shared by whoever has the page open: this is a single-user demo.
        app.state.messages = []
        yield


app = FastAPI(title='Flood risk assistant', lifespan=lifespan)
logfire.instrument_fastapi(app)
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


class ChatRequest(BaseModel):
    message: str


@app.get('/')
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / 'index.html')


@app.get('/api/config')
async def config() -> dict[str, bool]:
    return {'sandbox': SANDBOX, 'online_evals': ONLINE_EVALS}


@app.get('/api/properties')
async def properties(request: Request) -> Response:
    return await _proxy(request, '/properties')


@app.get('/api/map/{layer}')
async def map_layer(request: Request, layer: Literal['flood-zones', 'water']) -> Response:
    return await _proxy(request, f'/map/{layer}')


@app.get('/api/image/{property_id}')
async def image(request: Request, property_id: str) -> Response:
    return await _proxy(request, f'/properties/{property_id}/aerial-image')


async def _proxy(request: Request, path: str) -> Response:
    """The browser only talks to this app; it fetches from the geo risk service on the browser's behalf."""
    upstream = await request.app.state.http.get(path)
    return Response(upstream.content, upstream.status_code, media_type=upstream.headers.get('content-type'))


@app.post('/api/reset')
async def reset(request: Request) -> dict[str, bool]:
    request.app.state.messages = []
    return {'ok': True}


@app.post('/api/chat')
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    state = request.app.state

    # Count requests to the geo risk service, so the page can show the fan-out as it happens.
    api_requests = 0

    async def count_request(_: httpx.Response) -> None:
        nonlocal api_requests
        api_requests += 1
        if api_requests % 20 == 0:
            await events.put({'type': 'requests', 'count': api_requests})

    async def run() -> None:
        try:
            async with http_client() as http:
                http.event_hooks['response'].append(count_request)
                deps = Deps(http=http, events=events)
                state.messages = await _run_agent(agent, body.message, state.messages, deps, events)
            await events.put({'type': 'requests', 'count': api_requests})
        except Exception as e:  # noqa: BLE001 - report any failure to the page instead of a dead stream
            logfire.exception('chat failed')
            await events.put({'type': 'error', 'message': f'{type(e).__name__}: {e}'})
        finally:
            await events.put({'type': 'done'})

    async def stream() -> AsyncIterator[str]:
        task = asyncio.create_task(run())
        try:
            while True:
                event = await events.get()
                yield f'data: {json.dumps(event)}\n\n'
                if event['type'] == 'done':
                    break
        finally:
            await task

    return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache'})


async def _run_agent(
    agent: Agent[Deps, str],
    prompt: str,
    messages: list[ModelMessage],
    deps: Deps,
    events: asyncio.Queue[dict[str, Any]],
) -> list[ModelMessage]:
    """Run one turn, sending the text as it streams, and return the new message history."""
    segment = 0
    async with agent.iter(prompt, message_history=messages, deps=deps) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for content in stream.stream_output(debounce_by=None):
                        await events.put({'type': 'text', 'segment': segment, 'text': str(content)})
                segment += 1
            elif Agent.is_call_tools_node(node):
                await events.put({'type': 'working'})
        assert run.result is not None
        return run.result.all_messages()
