from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from typing import Literal, cast

import fastapi
import httpx
import logfire
from fastapi import Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pydantic.alias_generators import to_camel
from pydantic_ai.builtin_tools import AbstractBuiltinTool, CodeExecutionTool, ImageGenerationTool, WebSearchTool
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

from .agent import agent

logfire.configure(service_name='ai-chat')
logfire.instrument_pydantic_ai()
logfire.instrument_mcp()
logfire.instrument_httpx(capture_all=True)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    async with agent:
        async with httpx.AsyncClient() as client:
            yield {'client': client}


app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


@app.options('/api/chat')
def options_chat():
    pass


AIModelID = Literal[
    'gateway/anthropic:claude-sonnet-4-5',
    'gateway/responses:gpt-5',
    'gateway/gemini:gemini-3-pro-preview',
]
BuiltinToolID = Literal['web_search', 'image_generation', 'code_execution']


class AIModel(BaseModel):
    id: str
    name: str
    builtin_tools: list[BuiltinToolID]


class BuiltinTool(BaseModel):
    id: BuiltinToolID
    name: str


BUILTIN_TOOL_DEFS: list[BuiltinTool] = [
    BuiltinTool(id='web_search', name='Web Search'),
    BuiltinTool(id='code_execution', name='Code Execution'),
    BuiltinTool(id='image_generation', name='Image Generation'),
]

BUILTIN_TOOLS: dict[BuiltinToolID, AbstractBuiltinTool] = {
    'web_search': WebSearchTool(),
    'code_execution': CodeExecutionTool(),
    'image_generation': ImageGenerationTool(),
}

AI_MODELS: list[AIModel] = [
    AIModel(
        id='gateway/anthropic:claude-sonnet-4-5',
        name='Claude Sonnet 4.5',
        builtin_tools=[
            'web_search',
            'code_execution',
        ],
    ),
    AIModel(
        id='gateway/responses:gpt-5',
        name='GPT 5',
        builtin_tools=[
            'web_search',
            'code_execution',
            'image_generation',
        ],
    ),
    AIModel(
        id='gateway/gemini:gemini-3-pro-preview',
        name='Gemini 3 Pro',
        builtin_tools=[
            'web_search',
            'code_execution',
        ],
    ),
]


class ConfigureFrontend(BaseModel, alias_generator=to_camel, populate_by_name=True):
    models: list[AIModel]
    builtin_tools: list[BuiltinTool]


@app.get('/api/configure')
async def configure_frontend() -> ConfigureFrontend:
    return ConfigureFrontend(
        models=AI_MODELS,
        builtin_tools=BUILTIN_TOOL_DEFS,
    )


class ChatRequestExtra(BaseModel, extra='ignore', alias_generator=to_camel):
    model: AIModelID | None = None
    builtin_tools: list[BuiltinToolID] = []


@app.post('/api/chat')
async def post_chat(request: Request) -> Response:
    run_input = VercelAIAdapter.build_run_input(await request.body())
    extra_data = ChatRequestExtra.model_validate(run_input.__pydantic_extra__)
    logfire.info(f'{extra_data=}')
    client = cast(httpx.AsyncClient, request.state.client)
    return await VercelAIAdapter[httpx.AsyncClient].dispatch_request(
        request,
        agent=agent,
        deps=client,
        model=extra_data.model,
        builtin_tools=[BUILTIN_TOOLS[tool_id] for tool_id in extra_data.builtin_tools],
    )


@app.get('/')
@app.get('/{id}')
async def index(request: Request):
    client = cast(httpx.AsyncClient, request.state.client)
    response = await client.get('https://cdn.jsdelivr.net/npm/@pydantic/ai-chat-ui@0.0.2/dist/index.html')
    return HTMLResponse(content=response.content, status_code=response.status_code)
